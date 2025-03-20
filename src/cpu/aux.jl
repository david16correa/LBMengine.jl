#= ==========================================================================================
=============================================================================================
scalar and vector field arithmetic auxiliary functions
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

function vectorCrossVectorField(V::Vector, W::Array)
    return [cross(V, W) for W in W]
end

function vectorFieldCrossVectorField(V::Array, W::Array)
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
    mkpath("output.lbm")

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
