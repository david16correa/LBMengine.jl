#= ==========================================================================================
=============================================================================================
general methods
=============================================================================================
========================================================================================== =#

function eulerStep!(id::Int64, model::LBMmodel)
    # the particle is saved locally for readibility
    particle = model.particles[id]
    # a simple euler scheme is used
    particle.position += model.spaceTime.Δt * particle.velocity

    # if the particle has spherical symmetry, then we're done
    :spherical in particle.particleParams.symmetries && (return nothing)

    # rotations must be added if non spherical objects are implemetned!!
    @warn "Methods for non-spherical objects have not yet been implemetned! Particles will not rotate!"
end

function moveParticles!(id::Int64, model::LBMmodel; initialSetup = false)
    # the particle is named locally for readibility
    particle = model.particles[id]
    # node veloicity must change if: there was linear or angular momentum input, or if the particle is being setup
    nodeVelocityMustChange = (particle.momentumInput |> v -> v != zero(v)) || (particle.angularMomentumInput |> v -> v != zero(v)) || initialSetup
    # normally, particleMoved would be initialized as false; it'd be later turned into true if the particle moved (go figure).
    # This variable helps us skip unnecessary analyses. However, these are necessary during initial setup. Hence this initialization.
    particleMoved = initialSetup

    # the velocity and angular velocity are updated, and the particle is moved (this is not necessary in the initial setup)
    if !initialSetup
        particle.velocity += particle.particleParams.inverseMass * particle.momentumInput
        particle.angularVelocity += particle.particleParams.inverseMomentOfInertia * particle.angularMomentumInput
        # the particle will be moved only if the velocity is nonzero! (there are no methods for rotating particles)
        (particle.velocity |> v -> v != zero(v)) && (eulerStep!(id, model); particleMoved = true)
    end

    # the inputs are reset
    particle.momentumInput = particle.momentumInput |> zero
    particle.angularMomentumInput = particle.angularMomentumInput |> zero

    # if the particle didn't move, we're done
    !particleMoved && return

    # the particle discretisation on the lattice is updated
    solidRegion = [particle.particleParams.solidRegionGenerator(x - particle.position) for x in model.spaceTime.X]
    #= solidRegion != particle.boundaryConditionsParams.solidRegion && println("ups") =#
    model.spaceTime.dims < 3 && (solidRegion = sparse(solidRegion))
    # the new streaming invasion regions are found
    streamingInvasionRegions = bounceBackPrep(solidRegion, model.velocities; returnStreamingInvasionRegions = true)
    interiorStreamingInvasionRegions = bounceBackPrep(solidRegion .|> b -> !b, model.velocities; returnStreamingInvasionRegions = true)
    for id in eachindex(model.boundaryConditionsParams.oppositeVectorId)
        streamingInvasionRegions[id] = streamingInvasionRegions[id] .|| interiorStreamingInvasionRegions[id]
    end

    # the solid velocity (momentum density / mass density) is found
    if nodeVelocityMustChange
        particle.nodeVelocity = [
            (solidRegion[findfirst(v -> v == x, model.spaceTime.X)]) ? (particle.velocity + cross(particle.angularVelocity, x - particle.position)) : [0. for _ in 1:model.spaceTime.dims]
        for x in model.spaceTime.X]
    end

    # everything is stored in the original particle
    particle.boundaryConditionsParams = (; solidRegion, streamingInvasionRegions);
end

function moveParticles!(model::LBMmodel)
    for id in eachindex(model.particles)
        moveParticles!(id, model);
    end
end

