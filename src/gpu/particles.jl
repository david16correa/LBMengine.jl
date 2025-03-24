#= ==========================================================================================
=============================================================================================
all things particles
=============================================================================================
========================================================================================== =#

function eulerStep!(id::Int64, model::LBMmodel)
    # the particle is saved locally for readibility
    particle = model.particles[id]
    # a simple euler scheme is used
    particle.position += model.spaceTime.timeStep * particle.velocity

    # if the particle has spherical symmetry, then we're done
    :spherical in particle.particleParams.properties && (return nothing)

    # rotations must be added if non spherical objects are implemetned!!
    @warn "Methods for non-spherical objects have not yet been implemetned! Particles will not rotate!"
end

function moveParticles!(id::Int64, model::LBMmodel; initialSetup = false)
    # the particle is named locally for readibility
    particle = model.particles[id]
    # these are used to avoid unnecessary analyses (these analyses, however, are necessary during initial setup)
    particleMoved = initialSetup
    nodeVelocityMustBeFound = initialSetup

    # the velocity and angular velocity are updated, and the particle is moved (this is not necessary in the initial setup)
    if !initialSetup
        deltaP, deltaL = particle.momentumInput, particle.angularMomentumInput
        particle.velocity += particle.particleParams.inverseMass * deltaP
        particle.angularVelocity += particle.particleParams.inverseMomentOfInertia * deltaL
        # the particle will be moved only if the velocity is nonzero! (there are no methods for rotating particles)
        (particle.velocity |> v -> v != zero(v)) && (eulerStep!(id, model); particleMoved = true)
        # the node velocity needs to be found if a) the angular velocity changed, or b) the particle moved
        ((deltaL |> v -> v != zero(v)) || particleMoved) && (nodeVelocityMustBeFound = true)
    end

    # the inputs are reset
    particle.momentumInput = particle.momentumInput |> zero
    particle.angularMomentumInput = particle.angularMomentumInput |> zero

    # the projection of the particle onto the lattice is found, along with its boundary nodes (streaming invasion regions)
    if particleMoved
        # the particle discretisation on the lattice is updated
        solidRegion = getSphere(particle.particleParams.radius, vectorFieldPlusVector(model.spaceTime.X, -particle.position))
        particle.boundaryConditionsParams = (; solidRegion);

        # in gpu modules we only have ladd
        #= if :ladd in model.schemes =#
            # the new exterior boundary and streaming invasion regions are found
            streamingInvasionRegions = bounceBackPrep(particle.boundaryConditionsParams.solidRegion, model.velocities; returnStreamingInvasionRegions = true)
            #= particle.boundaryConditionsParams = merge(particle.boundaryConditionsParams, (; streamingInvasionRegions)); =#
            particle.boundaryConditionsParams = (; particle.boundaryConditionsParams..., streamingInvasionRegions);
        #= end =#

    end

    # the solid velocity (momentum density / mass density) is found
    if nodeVelocityMustBeFound
        particle.nodeVelocity = getNodeVelocity(model; id = id)
    end
end

function moveParticles!(model::LBMmodel)
    for id in eachindex(model.particles)
        moveParticles!(id, model);
    end
end
