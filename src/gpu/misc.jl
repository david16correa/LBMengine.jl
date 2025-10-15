#= ==========================================================================================
=============================================================================================
wall and fluid nodes functions
=============================================================================================
========================================================================================== =#

function wallNodes(modelSize, walledDimensions)
    # the size, dimensions, and side length of the density field are saved
    dims, lens = length(modelSize), modelSize;

    # the wallMap is initialized as an boolean array filled with zeroes,
    # and the indices of the density field are saved.
    wallMap = fillCuArray(false, modelSize)
    indices = [1:i for i in modelSize];

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

    #  the final wall map is returned
    return wallMap
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

    fluidVelocity = model.fluidVelocity |> Array |> U -> [U[i,j,:] for i in 1:size(U,1), j in 1:size(U,2)];

    if maximumFluidSpeed == :default
        maximumFluidSpeed = (fluidVelocity .|> norm) |> maximum
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
    # we only need the normalized vector field; places where the fluid velocity is zero are taken care of
    vec = [
        fluidVelocity[i,j] |> v -> norm(v) > 0 ? v/norm(v) : [0,0]
        for i ∈ eachindex(model.spaceTime.coordinates[1])[indices_x] for j ∈ eachindex(model.spaceTime.coordinates[2])[indices_y]
    ];
    vec = (model.spaceTime.coordinates[1] |> v -> v[end] - v[1]) / 20 .* vec

    arrows2d!(fig[1,1], pos, vec,
        shaftlength = 0.05,
        shaftwidth = 0.01,
        tiplength = 0.05,
        tipwidth = 0.05,
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

    massDensity = model.massDensity |> Array

    if maximumMassDensity == :default
        maximumMassDensity = massDensity |> maximum
    end

    if minimumMassDensity == :default
        try
            minimumMassDensity = massDensity[model.boundaryConditionsParams.wallRegion |> Array .|> b -> !b] |> minimum |> x -> maximum([0, x])
        catch
            minimumMassDensity = massDensity |> minimum
        end
    end

    minimumMassDensity ≈ maximumMassDensity && (minimumMassDensity = 0; maximumMassDensity = 1);

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
D2Q9 = (; cs, ws)

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
D3Q15 = (; cs, ws)

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
]
D3Q19 = (; cs, ws);

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
D3Q27 = (; cs, ws);
