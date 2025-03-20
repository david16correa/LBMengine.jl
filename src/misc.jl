#= ==========================================================================================
=============================================================================================
wall and fluid nodes functions
=============================================================================================
========================================================================================== =#

function wallNodes(massDensity::Array{Float64};
    walledDimensions = :default
)
    # the size, dimensions, and side length of the density field are saved
    sizeM = size(massDensity)
    dims, lens = length(sizeM), sizeM;

    # the wallMap is initialized as an boolean array filled with zeroes,
    # and the indices of the density field are saved.
    wallMap = sizeM |> zeros .|> Bool
    indices = [1:i for i in sizeM];

    # by default, all dimensions are walled
    (walledDimensions == :default) ? (walledDimensions = eachindex(indices)) : nothing

    # for each dimension, a padding will be added. To do this, a set of auxiliary indices will be needed.
    auxIndices = copy(indices);

    # the padding is added in every dimension
    for id in walledDimensions
        for paddingRange in (1:1, lens[id]:lens[id])
            auxIndices[id] = paddingRange;
            wallMap[auxIndices...] .= 1;
            auxIndices = copy(indices)
        end
    end

    #  the final wall map is returned as a sparse matrix
    (dims > 2) ? (return wallMap |> BitArray) : (return wallMap |> sparse)
    #  SparseArrays.jl only works for 1D and 2D. Look into SparseArrayKit.jl for higher dimensional compatibility!!
end

#= ==========================================================================================
=============================================================================================
output management
=============================================================================================
========================================================================================== =#

function mkFigDirs()
    !isdir("figs") && mkdir("figs")
    !isdir("figs/$(today())") && mkdir("figs/$(today())")
end

function save_jpg(name::String, fig::Figure)
    nameJPG = name*".jpg"
    save(".output.png", fig)
    if Sys.islinux()
        run(`convert .output.png $nameJPG`)
        run(`rm .output.png`);
    elseif Sys.isapple()
        run(`magick .output.png $nameJPG`)
        run(`rm .output.png`);
    elseif Sys.iswindows()
        save(name*".png", fig)
        rm(".output.png")
    end
end

#= ==========================================================================================
=============================================================================================
graphics stuff
=============================================================================================
========================================================================================== =#

"the fluid velocity plot is generated and saved."
function plotFluidVelocity(model::LBMmodel;
    saveFig = true,
    fluidVelocity = :default, 
    maximumFluidSpeed = :default
)
    lbs = []
    ubs = []
    for id in 1:model.spaceTime.dims
        lb, ub = model.spaceTime.coordinates[id] |> V -> (minimum(V), maximum(V));
        append!(lbs, [lb])
        append!(ubs, [ub])
    end

    t = model.time

    fluidVelocity = model.fluidVelocity

    if maximumFluidSpeed == :default
        maximumFluidSpeed = (model.fluidVelocity .|> norm) |> maximum
    end

    #----------------------------------heatmap and colorbar---------------------------------
    fig, ax, hm = heatmap(model.spaceTime.coordinates[1], model.spaceTime.coordinates[2], norm.(fluidVelocity)/model.fluidParams.c_s, alpha = 0.7,
        colorrange = (0, maximumFluidSpeed/model.fluidParams.c_s), 
        highclip = :red, # truncate the colormap 
        axis=(
            title = "fluid velocity, t = $(t |> x -> round(x; digits = 2))",
            aspect = ubs - lbs |> v -> v[1]/v[2]
        ),
    );
    ax.xlabel = "x"; ax.ylabel = "y";
    Colorbar(fig[:, end+1], hm, label = "Mach number (M = u/cₛ)"
        #=ticks = (-1:0.5:1, ["$i" for i ∈ -1:0.5:1]),=#
    );
    #--------------------------------------vector field---------------------------------------
    indices_x = range(1, stop = length(model.spaceTime.coordinates[1]), length = 11) |> collect .|> round .|> Int64
    indices_y = range(1, stop = length(model.spaceTime.coordinates[2]), length = 11) |> collect .|> round .|> Int64
    vectorFieldX = model.spaceTime.coordinates[1][indices_x];
    vectorFieldY = model.spaceTime.coordinates[2][indices_y];
    pos = [Point2(i,j) for i ∈ vectorFieldX for j ∈ vectorFieldY];
    vec = [fluidVelocity[i,j] for i ∈ eachindex(model.spaceTime.coordinates[1])[indices_x] for j ∈ eachindex(model.spaceTime.coordinates[2])[indices_y]];
    vec = 0.07 .* vec ./ maximumFluidSpeed;
    nonZeroVec = (vec .|> norm) .> 0.0007
    arrows!(fig[1,1], pos[nonZeroVec], vec[nonZeroVec],
        arrowsize = 10,
        align = :center
    );
    xlims!(lbs[1], ubs[1]);
    ylims!(lbs[2], ubs[2]);

    if saveFig
        mkFigDirs()
        save_jpg("figs/$(today())/LBM figure $(Time(now()))", fig)
        return nothing
    else
        return fig, ax
    end
end

"the momentum density plot is generated and saved."
function plotMomentumDensity(model::LBMmodel;
    saveFig = true, 
    momentumDensity = :default, 
    maximumMomentumDensity = :default
)
    lbs = []
    ubs = []
    for id in 1:model.spaceTime.dims
        lb, ub = model.spaceTime.coordinates[id] |> V -> (minimum(V), maximum(V));
        append!(lbs, [lb])
        append!(ubs, [ub])
    end

    t = model.time

    momentumDensity = model.momentumDensity

    if maximumMomentumDensity == :default
        maximumMomentumDensity = (model.momentumDensity .|> norm) |> maximum
    end

    #----------------------------------heatmap and colorbar---------------------------------
    fig, ax, hm = heatmap(model.spaceTime.coordinates[1], model.spaceTime.coordinates[2], norm.(momentumDensity), alpha = 0.7,
        colorrange = (0, maximumMomentumDensity), 
        highclip = :red, # truncate the colormap 
        axis=(
            title = "momentum density, t = $(t |> x -> round(x; digits = 2))",
            aspect = ubs - lbs |> v -> v[1]/v[2]
        ),
    );
    ax.xlabel = "x"; ax.ylabel = "y";
    Colorbar(fig[:, end+1], hm,
        #=ticks = (-1:0.5:1, ["$i" for i ∈ -1:0.5:1]),=#
    );
    #--------------------------------------vector field---------------------------------------
    indices_x = range(1, stop = length(model.spaceTime.coordinates[1]), length = 11) |> collect .|> round .|> Int64
    indices_y = range(1, stop = length(model.spaceTime.coordinates[2]), length = 11) |> collect .|> round .|> Int64
    vectorFieldX = model.spaceTime.coordinates[1][indices_x];
    vectorFieldY = model.spaceTime.coordinates[2][indices_y];
    pos = [Point2(i,j) for i ∈ vectorFieldX for j ∈ vectorFieldY];
    vec = [momentumDensity[i,j] for i ∈ eachindex(model.spaceTime.coordinates[1])[indices_x] for j ∈ eachindex(model.spaceTime.coordinates[2])[indices_y]];
    vec = 0.07 .* vec ./ maximumMomentumDensity;
    nonZeroVec = (vec .|> norm) .> 0.007
    arrows!(fig[1,1], pos[nonZeroVec], vec[nonZeroVec],
        arrowsize = 10, 
        align = :center
    );
    xlims!(lbs[1], ubs[1]);
    ylims!(lbs[2], ubs[2]);

    if saveFig
        mkFigDirs()
        save_jpg("figs/$(today())/LBM figure $(Time(now()))", fig)
        return nothing
    else
        return fig, ax
    end
end


"the mass density plot is generated and saved."
function plotMassDensity(model::LBMmodel;
    saveFig = true,
    massDensity = :default, 
    maximumMassDensity = :default,
    minimumMassDensity = :default
)
    lbs = []
    ubs = []
    for id in 1:model.spaceTime.dims
        lb, ub = model.spaceTime.coordinates[id] |> V -> (minimum(V), maximum(V));
        append!(lbs, [lb])
        append!(ubs, [ub])
    end

    t = model.time

    massDensity = model.massDensity

    if maximumMassDensity == :default
        maximumMassDensity = massDensity |> maximum
    end

    if minimumMassDensity == :default
        minimumMassDensity = massDensity[model.boundaryConditionsParams.wallRegion .|> b -> !b] |> minimum |> x -> maximum([0, x])
    end

    minimumMassDensity ≈ maximumMassDensity && (minimumMassDensity = 0);
    maximumMassDensity ≈ minimumMassDensity && (maximumMassDensity = 1);

    #----------------------------------heatmap and colorbar---------------------------------
    fig, ax, hm = heatmap(model.spaceTime.coordinates[1], model.spaceTime.coordinates[2], massDensity,
        colorrange = (minimumMassDensity, maximumMassDensity), 
        lowclip = :black, # truncate the colormap 
        axis=(
            title = "mass density, t = $(t |> x -> round(x; digits = 2))",
            aspect = ubs - lbs |> v -> v[1]/v[2]
        ),
    );
    ax.xlabel = "x"; ax.ylabel = "y";
    Colorbar(fig[:, end+1], hm,
        #=ticks = (-1:0.5:1, ["$i" for i ∈ -1:0.5:1]),=#
    );
    xlims!(lbs[1], ubs[1]);
    ylims!(lbs[2], ubs[2]);

    if saveFig
        mkFigDirs()
        save_jpg("figs/$(today())/LBM figure $(Time(now()))", fig)
        return nothing
    else
        return fig, ax
    end
end

#= ==========================================================================================
=============================================================================================
some velocity sets
=============================================================================================
========================================================================================== =#

cs = [
    [[0]]
    #======#
    [[p] for p in [-1, 1]]
]
ws = [
    2/3
    #======#
    fill(1/6, 2)
]
D1Q3 = [LBMvelocity(cs[i], ws[i]) for i in eachindex(cs)];

cs = [
    [[0,0]]
    #======#
    [[p,0] for p in [-1, 1]]
    [[0,p] for p in [-1, 1]]
    #======#
    [[p,q] for p in [-1,1], q in [-1,1]]|>vec
]
ws = [
    4/9
    #======#
    fill(1/9, 4)
    #======#
    fill(1/36, 4)
]
D2Q9 = [LBMvelocity(cs[i], ws[i]) for i in eachindex(cs)]

cs = [
    [[0,0,0]]
    #======#
    [[p,0,0] for p in [-1, 1]]
    [[0,p,0] for p in [-1, 1]]
    [[0,0,p] for p in [-1, 1]]
    #======#
    [[p,q,r] for p in [-1,1], q in [-1,1], r in [-1,1]]|>vec
]
ws = [
    2/9
    #======#
    fill(1/9, 6)
    #======#
    fill(1/72, 8)
]
D3Q15 = [LBMvelocity(cs[i], ws[i]) for i in eachindex(cs)]

cs = [
    [[0,0,0]]
    #======#
    [[p,0,0] for p in [-1, 1]]
    [[0,p,0] for p in [-1, 1]]
    [[0,0,p] for p in [-1, 1]]
    #======#
    [[0,p,q] for p in [-1,1], q in [-1,1]]|>vec
    [[p,0,q] for p in [-1,1], q in [-1,1]]|>vec
    [[p,q,0] for p in [-1,1], q in [-1,1]]|>vec
]
ws = [
    1/3
    #======#
    fill(1/18, 6)
    #======#
    fill(1/36, 12)
];
D3Q19 = [LBMvelocity(cs[i], ws[i]) for i in eachindex(cs)];

cs = [
    [[0,0,0]]
    #======#
    [[p,0,0] for p in [-1, 1]]
    [[0,p,0] for p in [-1, 1]]
    [[0,0,p] for p in [-1, 1]]
    #======#
    [[0,p,q] for p in [-1,1], q in [-1,1]]|>vec
    [[p,0,q] for p in [-1,1], q in [-1,1]]|>vec
    [[p,q,0] for p in [-1,1], q in [-1,1]]|>vec
    #======#
    [[p,q,r] for p in [-1,1], q in [-1,1], r in [-1,1]]|>vec
]
ws = [
    8/27
    #======#
    fill(2/27, 6)
    #======#
    fill(1/54, 12)
    #======#
    fill(1/216, 8)
]
D3Q27 = [LBMvelocity(cs[i], ws[i]) for i in eachindex(cs)];
