#= ==========================================================================================
=============================================================================================
misc functions
=============================================================================================
========================================================================================== =#

mean(v) = sum(v) / length(v)

#= ==========================================================================================
=============================================================================================
scalar and vector fild arithmetic auxilary functions -
=============================================================================================
========================================================================================== =#

dot(v::Vector, w::Vector) = v .* w |> sum

norm(T) = √(sum(el for el ∈ T.*T))

function scalarFieldTimesVector(a::Array, V::Vector)
    return [a * V for a in a]
end

function vectorFieldDotVector(F::Array, v::Vector)
    return [dot(F, v) for F in F]
end

function vectorFieldDotVectorField(V::Array, W::Array)
    return size(V) |> sizeV -> [dot(V[id], W[id]) for id in eachindex(IndexCartesian(), V)]
end

#= ==========================================================================================
=============================================================================================
shift auxilary functions 
=============================================================================================
========================================================================================== =#

function pbcIndexShift(indices::UnitRange{Int64}, Δ::Int64)
    if Δ > 0
        return [indices[end-Δ+1:end]; indices[1:end-Δ]]
    elseif Δ < 0
        # originalmente era [indices[(Δ+1):end]; indices[1:Δ]] con un shift positivo, pero Δ < 0
        return [indices[(-Δ+1):end]; indices[1:-Δ]]
    else
        return indices
    end
end

function pbcMatrixShift(M::Union{Array{Float64}, SparseMatrixCSC, BitArray}, Δ::Vector{Int64})
    return size(M) |> sizeM -> [pbcIndexShift(1:sizeM[i], Δ[i]) for i in eachindex(sizeM)] |> shiftedIndices -> M[shiftedIndices...]
end

#= ==========================================================================================
=============================================================================================
wall and fluid nodes functions 
=============================================================================================
========================================================================================== =#

function wallNodes(ρ::Array{Float64};
    walledDimensions = :default
)
    # the size, dimensions, and side length of the density field are saved
    sizeM = size(ρ)
    dims, len = length(sizeM), sizeM[1];

    # the wallMap is initialized as an boolean array filled with zeroes,
    # and the indices of the density field are saved.
    wallMap = sizeM |> zeros .|> Bool
    indices = [1:i for i in sizeM];

    # by default, all dimensions are walled
    (walledDimensions == :default) ? (walledDimensions = eachindex(indices)) : nothing

    # for each dimension, a padding will be added. To do this, a set of auxilary indices will be needed.
    auxIndices = copy(indices);
    paddingRanges = (1:1, len:len);

    # the padding is added in every dimension
    for id in walledDimensions, paddingRange in paddingRanges
        auxIndices[id] = paddingRange;
        wallMap[auxIndices...] .= 1;
        auxIndices = copy(indices)
    end

    #  the final wall map is returned as a sparse matrix
    (dims > 2) ? (return wallMap) : (return wallMap |> sparse)
    #  SparseArrays.jl only works for 1D and 2D. Look into SparseArrayKit.jl for higher dimensional compatibility!!
end

#= ==========================================================================================
=============================================================================================
bounce-back boundary conditions
=============================================================================================
========================================================================================== =#

function bounceBackPrep(wallRegion::Union{SparseMatrixCSC, BitArray}, velocities::Vector{LBMvelocity})
    cs = [velocity.c for velocity in velocities];

    streamingInvasionRegions = [(pbcMatrixShift(wallRegion, -c) .|| wallRegion) .⊻ wallRegion for c in cs]

    oppositeVectorId = [findfirst(x -> x == -c, cs) for c in cs]

    return streamingInvasionRegions, oppositeVectorId
end

#= ==========================================================================================
=============================================================================================
graphics stuff
=============================================================================================
========================================================================================== =#

function save_jpg(name::String, fig::Figure)
    namePNG = name*".png"
    nameJPG = name*".jpg"
    save(namePNG, fig) 
    run(`magick $namePNG $nameJPG`)
    run(`rm $namePNG`)
end

"the fluid velocity plot is generated and saved."
function plotFluidVelocity(model::LBMmodel;
    saveFig = true, 
    t = :default,
    u = :default, 
    maximumFluidSpeed = :default
)
    xlb, xub = model.spaceTime.x |> V -> (minimum(V), maximum(V));

    (t == :default) ? (t = model.time[end]) : nothing

    (u == :default) ? (u = model.u) : nothing

    if maximumFluidSpeed == :default
        maximumFluidSpeed = (model.u .|> norm) |> maximum
    end

    #----------------------------------heatmap and colorbar---------------------------------
    fig, ax, hm = heatmap(model.spaceTime.x, model.spaceTime.x, norm.(u)/model.fluidParams.c_s, alpha = 0.7,
        colorrange = (0, maximumFluidSpeed/model.fluidParams.c_s), 
        highclip = :red, # truncate the colormap 
        axis=(
            title = "fluid velocity, t = $(t |> x -> round(x; digits = 2))",
            aspect = 1,
        ),
    );
    ax.xlabel = "x"; ax.ylabel = "y";
    Colorbar(fig[:, end+1], hm, label = "Mach number (M = u/cₛ)"
        #=ticks = (-1:0.5:1, ["$i" for i ∈ -1:0.5:1]),=#
    );
    #--------------------------------------vector field---------------------------------------
    indices = range(1, stop = length(model.spaceTime.x), length = 10) |> collect .|> round .|> Int64
    vectorFieldX = model.spaceTime.x[indices];
    pos = [Point2(i,j) for i ∈ vectorFieldX for j ∈ vectorFieldX];
    vec = [u[i,j] for i ∈ eachindex(model.spaceTime.x)[indices] for j ∈ eachindex(model.spaceTime.x)[indices]];
    vec = 0.07 .* vec ./ maximumFluidSpeed;
    arrows!(fig[1,1], pos, vec,
        arrowsize = 10,
        align = :center
    );
    xlims!(xlb, xub);
    ylims!(xlb, xub);

    if saveFig
        save_jpg("figs/$(today())/LBM figure $(Time(now()))", fig)
    else
        return fig, ax
    end
end

"the momentum density plot is generated and saved."
function plotMomentumDensity(model::LBMmodel;
    saveFig = true, 
    t = :default,
    ρu = :default, 
    maximumMomentumDensity = :default
)
    xlb, xub = model.spaceTime.x |> V -> (minimum(V), maximum(V));

    (t == :default) ? (t = model.time[end]) : nothing

    (ρu == :default) ? (ρu = model.ρu) : nothing

    if maximumMomentumDensity == :default
        maximumMomentumDensity = (model.ρu .|> norm) |> maximum
    end

    #----------------------------------heatmap and colorbar---------------------------------
    fig, ax, hm = heatmap(model.spaceTime.x, model.spaceTime.x, norm.(ρu), alpha = 0.7,
        colorrange = (0, maximumMomentumDensity), 
        highclip = :red, # truncate the colormap 
        axis=(
            title = "momentum density, t = $(t |> x -> round(x; digits = 2))",
            aspect = 1,
        ),
    );
    ax.xlabel = "x"; ax.ylabel = "y";
    Colorbar(fig[:, end+1], hm,
        #=ticks = (-1:0.5:1, ["$i" for i ∈ -1:0.5:1]),=#
    );
    #--------------------------------------vector field---------------------------------------
    indices = range(1, stop = length(model.spaceTime.x), length = 10) |> collect .|> round .|> Int64
    vectorFieldX = model.spaceTime.x[indices];
    pos = [Point2(i,j) for i ∈ vectorFieldX for j ∈ vectorFieldX];
    vec = [ρu[i,j] for i ∈ eachindex(model.spaceTime.x)[indices] for j ∈ eachindex(model.spaceTime.x)[indices]];
    vec = 0.07 .* vec ./ maximumMomentumDensity;
    arrows!(fig[1,1], pos, vec, 
        arrowsize = 10, 
        align = :center
    );
    xlims!(xlb, xub);
    ylims!(xlb, xub);

    if saveFig
        save_jpg("figs/$(today())/LBM figure $(Time(now()))", fig)
    else
        return fig, ax
    end
end


"the mass density plot is generated and saved."
function plotMassDensity(model::LBMmodel;
    saveFig = true,
    t = :default,
    ρ = :default, 
    maximumMassDensity = :default,
    minimumMassDensity = :default
)
    xlb, xub = model.spaceTime.x |> V -> (minimum(V), maximum(V));

    (t == :default) ? (t = model.time[end]) : nothing

    (ρ == :default) ? (ρ = model.ρ) : nothing

    if maximumMassDensity == :default
        maximumMassDensity = ρ |> maximum
    end

    if minimumMassDensity == :default
        minimumMassDensity = ρ |> minimum |> x -> maximum([0, x])
    end

    #----------------------------------heatmap and colorbar---------------------------------
    fig, ax, hm = heatmap(model.spaceTime.x, model.spaceTime.x, ρ, 
        colorrange = (minimumMassDensity, maximumMassDensity), 
        lowclip = :black, # truncate the colormap 
        axis=(
            title = "mass density, t = $(t |> x -> round(x; digits = 2))",
            aspect = 1,
        ),
    );
    ax.xlabel = "x"; ax.ylabel = "y";
    Colorbar(fig[:, end+1], hm,
        #=ticks = (-1:0.5:1, ["$i" for i ∈ -1:0.5:1]),=#
    );
    xlims!(xlb, xub);
    ylims!(xlb, xub);

    if saveFig
        save_jpg("figs/$(today())/LBM figure $(Time(now()))", fig)
    else
        return fig, ax
    end
end

"The animation of the fluid velocity evolution is created."
function anim8fluidVelocity(model::LBMmodel)
    xlb, xub = model.spaceTime.x |> V -> (minimum(V), maximum(V));

    maximumFluidSpeed = 1.;

    mkdir("tmp")

    us = [] |> Vector{Matrix{Vector{Float64}}};
    for t in eachindex(model.time)
        hydroVariablesUpdate!(model; time = t);
        append!(us, [model.u]);
    end

    maximumFluidSpeed = (us .|> M -> norm.(M)) .|> maximum |> maximum

    for t in eachindex(model.time)
        animationFig, animationAx = plotFluidVelocity(model; 
            saveFig = false, 
            t = model.time[t],
            u = us[t],
            maximumFluidSpeed = maximumFluidSpeed
        )
        save("tmp/$(t).png", animationFig)
    end

    run(`./createAnim.sh`)
    name = "anims/$(today())/LBM simulation $(Time(now())).mp4"
    run(`mv anims/output.mp4 $(name)`)
end

"The animation of the fluid velocity evolution is created."
function anim8momentumDensity(model::LBMmodel)
    xlb, xub = model.spaceTime.x |> V -> (minimum(V), maximum(V));

    maximumMomentumDensity = 1.;

    mkdir("tmp")

    ρus = [] |> Vector{Matrix{Vector{Float64}}};
    for t in eachindex(model.time)
        hydroVariablesUpdate!(model; time = t);
        append!(ρus, [model.ρu]);
    end

    maximumMomentumDensity = (ρus .|> M -> norm.(M)) .|> maximum |> maximum

    for t in eachindex(model.time)
        animationFig, animationAx = plotMomentumDensity(model; 
            saveFig = false,
            t = model.time[t],
            ρu = ρus[t],
            maximumMomentumDensity = maximumMomentumDensity
        )
        save("tmp/$(t).png", animationFig)
    end

    run(`./createAnim.sh`)
    name = "anims/$(today())/LBM simulation $(Time(now())).mp4"
    run(`mv anims/output.mp4 $(name)`)
end

"The animation of the mass density evolution is created."
function anim8massDensity(model::LBMmodel)
    xlb, xub = model.spaceTime.x |> V -> (minimum(V), maximum(V));

    ρs = [massDensity(model; time = t) for t in eachindex(model.time)]

    maximumMassDensity = (ρs .|> maximum) |> maximum
    minimumMassDensity = [ρ[model.boundaryConditionsParams.wallRegion .|> b -> !b] |> minimum for ρ in ρs] |> minimum |> x -> maximum([0, x])

    mkdir("tmp")

    for t in eachindex(model.time)
        animationFig, animationAx = plotMassDensity(model; 
            saveFig = false,
            t = model.time[t],
            ρ = ρs[t],
            maximumMassDensity = maximumMassDensity,
            minimumMassDensity = minimumMassDensity
        )
        save("tmp/$(t).png", animationFig)
    end

    run(`./createAnim.sh`)
    name = "anims/$(today())/LBM simulation $(Time(now())).mp4"
    run(`mv anims/output.mp4 $(name)`)
end


#= ==========================================================================================
=============================================================================================
some velocity sets 
=============================================================================================
========================================================================================== =#

cs = [
      [0],
      #======#
      [1],
      [-1]
];
ws = [
      2/3,
      #======#
      1/6,
      1/6
];
D1Q3 = [LBMvelocity(cs[i], ws[i]) for i in eachindex(cs)];

cs = [
      [0,0],
      #======#
      [1,0],
      [-1,0],
      [0,1],
      [0,-1],
      #======#
      [1,1],
      [-1,1],
      [1,-1],
      [-1,-1]
];
ws = [
      4/9,
      #======#
      1/9,
      1/9,
      1/9,
      1/9,
      #======#
      1/36,
      1/36,
      1/36,
      1/36
];
D2Q9 = [LBMvelocity(cs[i], ws[i]) for i in eachindex(cs)];

cs = [
      [0,0,0], 
      #======#
      [1,0,0],
      [0,1,0],
      [0,0,1],
      [-1,0,0],
      [0,-1,0],
      [0,0,-1],
      #======#
      [1,1,0],
      [1,-1,0],
      [-1,1,0],
      [-1,-1,0],
      [0,1,1],
      [0,1,-1],
      [0,-1,1],
      [0,-1,-1],
      [1,0,1],
      [1,0,-1],
      [-1,0,1],
      [-1,0,-1],
      #======#
      [1,1,1],
      [1,1,-1],
      [1,-1,1],
      [1,-1,-1],
      [-1,1,1],
      [-1,1,-1],
      [-1,-1,1],
      [-1,-1,-1]
];
ws = [
      8/27,
      #======#
      2/27,
      2/27,
      2/27,
      2/27,
      2/27,
      2/27,
      #======#
      1/54,
      1/54,
      1/54,
      1/54,
      1/54,
      1/54,
      1/54,
      1/54,
      1/54,
      1/54,
      1/54,
      1/54,
      #======#
      1/216,
      1/216,
      1/216,
      1/216,
      1/216,
      1/216,
      1/216,
      1/216
];
D3Q27 = [LBMvelocity(cs[i], ws[i]) for i in eachindex(cs)];
