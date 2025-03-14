#= ==========================================================================================
=============================================================================================
misc functions
=============================================================================================
========================================================================================== =#

mean(v) = sum(v) / length(v)

#= ==========================================================================================
=============================================================================================
scalar and vector fild arithmetic auxiliary functions
=============================================================================================
========================================================================================== =#

function scalarFieldTimesVector(a::Array, V::Vector)
    return [a * V for a in a]
end

function vectorFieldDotVector(F::Array, v::Vector)
    return [dot(F, v) for F in F]
end

function vectorFieldDotVectorField(V::Array, W::Array)
    return [dot(V[id], W[id]) for id in eachindex(IndexCartesian(), V)]
end

function cross(v::Vector, w::Vector)
    dim = length(v) # dimension consistency is not checked bc I trust I won't mess things up
    dim == 2 && (return v[1]*w[2] - v[2]*w[1]) # in two dimensions, vectors are assumed to have zero z component, and only the z component is returned
    dim == 3 && (return [v[2]*w[3] - v[3]*w[2], -v[1]*w[3] + v[3]*w[1], v[1]*w[2] - v[2]*w[1]])
end

#= I know this method is highly questionable, but it was born out of the need to compute the tangential
velocity using the position vector and the angular velocity in two dimensions. Ω happens to be a scalar
in two dimensions, but momentarily using three dimensions results in a simpler algorithm. =#
cross(omega::Real, V::Vector) = cross([0; 0; omega], [V; 0])[1:2]

function vectorCrossVectorField(V::Vector, W = Array)
    return [cross(V, W) for W in W]
end

function vectorFieldCrossVectorField(V::Array, W = Array)
    return [cross(V[id], W[id]) for id in eachindex(IndexCartesian(), V)]
end

vectorFieldCrossVector(V::Array, W::Vector) = - vectorCrossVectorField(W, V)

#= ==========================================================================================
=============================================================================================
calculus
=============================================================================================
========================================================================================== =#

function boundaryDerivative(f::Array, model::LBMmodel; targetDim = 1, h = 1)
    # the derivative is initialized
    Df = fill(zero(f[1]), size(f));
    # the derivative with respect to either x or y is found; cbc is not implemented for 3 dimensions yet!
    if targetDim == 1 # derivative with respect to x
        if 1 in model.boundaryConditionsParams.walledDimensions
            # forward difference for left boundary
            Df[1,:] = -1.5*f[1,:] + 2*f[2,:] - 0.5*f[3,:];
            # backward difference for right boundary
            Df[end,:] = 1.5*f[end,:] - 2*f[end-1,:] + 0.5*f[end-2,:];
        else
            # central difference for left and right boundaries considering periodic boundary conditions
            Df[[1;end], :] = 0.5(f[[2;1], :] - f[[end;end-1], :]);
        end
        # central difference for top and bottom boundaries
        Df[2:end-1,[1;end]] = 0.5(f[3:end, [1;end]] - f[1:end-2, [1;end]]);
    else # derivative with respect to y
        if 2 in model.boundaryConditionsParams.walledDimensions
            # forward difference for bottom boundary
            Df[:,1] = -1.5*f[:,1] + 2*f[:,2] - 0.5*f[:,3];
            # backward difference for top boundary
            Df[:,end] = 1.5*f[:,end] - 2*f[:,end-1] + 0.5*f[:,end-2];
        else
            # central difference for bottom and top boundaries considering periodic boundary conditions
            Df[:, [1;end]] = 0.5(f[:, [2;1]] - f[:, [end;end-1]]);
        end
        # central difference for left and right boundaries
        Df[[1;end], 2:end-1] = 0.5(f[[1;end], 3:end] - f[[1;end], 1:end-2]);

        # # forward difference for left boundary
        # Df[:, 1] = 0.5(f[:, 2] - f[:, end]);
        # # central difference for top and bottom boundaries
        # Df[[1;end], 2:end-1] = 0.5(f[[1;end], 3:end] - f[[1;end], 1:end-2]);
        # # backward difference for right boundary
        # Df[:, end] = 0.5(f[:, 1] - f[:, end-1]);

    end
    # derivative is returned 
    return Df ./ h
end

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
    (dims > 2) ? (return wallMap) : (return wallMap |> sparse)
    #  SparseArrays.jl only works for 1D and 2D. Look into SparseArrayKit.jl for higher dimensional compatibility!!
end

#= ==========================================================================================
=============================================================================================
bounce-back boundary conditions
=============================================================================================
========================================================================================== =#

function bounceBackPrep(wallRegion::Union{SparseMatrixCSC, BitArray}, velocities::Vector{LBMvelocity}; returnStreamingInvasionRegions = false)
    cs = [velocity.c |> Tuple for velocity in velocities];

    streamingInvasionRegions = [(circshift(wallRegion, -1 .* c) .|| wallRegion) .⊻ wallRegion for c in cs]

    returnStreamingInvasionRegions && return streamingInvasionRegions

    oppositeVectorId = [findfirst(x -> x == -1 .* c, cs) for c in cs]

    return streamingInvasionRegions, oppositeVectorId
end

#= ==========================================================================================
=============================================================================================
saving data
=============================================================================================
========================================================================================== =#

function writeTrajectories(model::LBMmodel)
    if model.spaceTime.dims == 2
        fluidDf = DataFrame(
            tick = model.tick,
            time = model.time,
            id_x = [coordinate[1] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            id_y = [coordinate[2] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            coordinate_x = [coordinate[1] for coordinate in model.spaceTime.X] |> vec,
            coordinate_y = [coordinate[2] for coordinate in model.spaceTime.X] |> vec,
            massDensity = model.massDensity |> vec,
            fluidVelocity_x = [velocity[1] for velocity in model.fluidVelocity] |> vec,
            fluidVelocity_y = [velocity[2] for velocity in model.fluidVelocity] |> vec
        ) # keyword argument constructor
    elseif model.spaceTime.dims == 3
        fluidDf = DataFrame(
            tick = model.tick,
            time = model.time,
            id_x = [coordinate[1] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            id_y = [coordinate[2] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            id_z = [coordinate[3] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            coordinate_x = [coordinate[1] for coordinate in model.spaceTime.X] |> vec,
            coordinate_y = [coordinate[2] for coordinate in model.spaceTime.X] |> vec,
            coordinate_z = [coordinate[3] for coordinate in model.spaceTime.X] |> vec,
            massDensity = model.massDensity |> vec,
            fluidVelocity_x = [velocity[1] for velocity in model.fluidVelocity] |> vec,
            fluidVelocity_y = [velocity[2] for velocity in model.fluidVelocity] |> vec,
            fluidVelocity_z = [velocity[3] for velocity in model.fluidVelocity] |> vec
        ) # keyword argument constructor
    end
    distributionsDf = DataFrame(
        vec.(model.distributions), ["f$(i)" for i in 1:length(model.distributions)]
    ) # vector of vectors constructor

    CSV.write("output.lbm/fluidTrj_$(model.tick).csv", [fluidDf distributionsDf])
end

function writeParticleTrajectory(particle::LBMparticle, model::LBMmodel)
    if model.spaceTime.dims == 2
        particleDf = DataFrame(
            tick = model.tick,
            time = model.time,
            particleId = particle.id,
            position_x = particle.position[1],
            position_y = particle.position[2],
            velocity_x = particle.velocity[1],
            velocity_y = particle.velocity[2],
            angularVelocity = particle.angularVelocity
        )
    elseif model.spaceTime.dims == 3
        particleDf = DataFrame(
            tick = model.tick,
            time = model.time,
            particleId = particle.id,
            position_x = particle.position[1],
            position_y = particle.position[2],
            position_z = particle.position[3],
            velocity_x = particle.velocity[1],
            velocity_y = particle.velocity[2],
            velocity_z = particle.velocity[3],
            angularVelocity_x = particle.angularVelocity[1],
            angularVelocity_y = particle.angularVelocity[2],
            angularVelocity_z = particle.angularVelocity[3],
        )
    end

    if !isfile("output.lbm/particlesTrj.csv")
        CSV.write("output.lbm/particlesTrj.csv", particleDf)
    else
        CSV.write("output.lbm/particlesTrj.csv", particleDf, append = true)
    end
end

function writeParticlesTrajectories(model::LBMmodel)
    for particle in model.particles
        writeParticleTrajectory(particle, model)
    end
end

function writeTensor(model::LBMmodel, T::Array, name::String)
    mkOutputDirs()

    if model.spaceTime.dims == 2
        metadataDf = DataFrame(
            tick = model.tick,
            time = model.time,
            id_x = [coordinate[1] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            id_y = [coordinate[2] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            coordinate_x = [coordinate[1] for coordinate in model.spaceTime.X] |> vec,
            coordinate_y = [coordinate[2] for coordinate in model.spaceTime.X] |> vec,
        ) # keyword argument constructor
        componentStrings = ["x", "y"]
    elseif model.spaceTime.dims == 3
        metadataDf = DataFrame(
            tick = model.tick,
            time = model.time,
            id_x = [coordinate[1] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            id_y = [coordinate[2] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            id_z = [coordinate[3] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            coordinate_x = [coordinate[1] for coordinate in model.spaceTime.X] |> vec,
            coordinate_y = [coordinate[2] for coordinate in model.spaceTime.X] |> vec,
            coordinate_z = [coordinate[3] for coordinate in model.spaceTime.X] |> vec,
        ) # keyword argument constructor
        componentStrings = ["x", "y", "z"]
    end

    componentLabels = ["component_"*i*j for i in componentStrings, j in componentStrings] |> vec
    tensorDf = T |> vec |> v -> DataFrame(
        vec.(v), componentLabels
    ) # vector of vectors constructor

    CSV.write("output.lbm/$name.csv", [metadataDf tensorDf])

    return nothing
end

#= ==========================================================================================
=============================================================================================
output management
=============================================================================================
========================================================================================== =#

function mkOutputDirs()
    !isdir("output.lbm") && mkdir("output.lbm")
end

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
        namePNG = name*".png"
        run(`mv .output.png $nameJPG`)
        run(`rm .output.png $namePNG`);
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
