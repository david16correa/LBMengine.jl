#= ==========================================================================================
=============================================================================================
all things particles
=============================================================================================
========================================================================================== =#

function eulerStep!(id::Int64, model::LBMmodel)
    # the particle is saved locally for readibility
    particle = model.particles[id]
    # a simple euler scheme is used
    particle.position += model.spaceTime.Δt * particle.velocity

    # if the particle has spherical symmetry, then we're done
    :spherical in particle.particleParams.properties && (return nothing)

    # rotations must be added if non spherical objects are implemetned!!
    @warn "Methods for non-spherical objects have not yet been implemetned! Particles will not rotate!"
end

function bulkVelocity(model::LBMmodel, particle::LBMparticle, X::Vector)
    xMinusR = X - particle.position
    xMinusR_norm = xMinusR |> norm

    # for a spherical particle, the ladd scheme only needs the nodes closest to the interphase!
    (:bead in particle.particleParams.properties) && :ladd in model.schemes && (abs(xMinusR_norm - particle.particleParams.radius) > model.spaceTime.latticeParameter) && return zero(particle.position)

    bulkV = particle.velocity + cross(particle.angularVelocity, xMinusR)

    (:bead in particle.particleParams.properties || xMinusR_norm == 0) && (return bulkV)

    @assert :squirmer in particle.particleParams.properties "As of right now, only beads and squirmers are supported."

    # fancyR := (x - R)/|x - R|, following Griffiths' electrodynamics book. This helps me read.
    fancyR = xMinusR/xMinusR_norm

    e_dot_fancyR = dot(particle.particleParams.swimmingDirection, fancyR);

    firstTerm = particle.particleParams.B1 + particle.particleParams.B2 * e_dot_fancyR
    secondTerm = e_dot_fancyR * fancyR - particle.particleParams.swimmingDirection

    #= return (firstTerm * secondTerm) * xMinusR_norm / particle.particleParams.radius + bulkV =#
    return firstTerm * secondTerm + bulkV
end

function moveParticles!(id::Int64, model::LBMmodel; initialSetup = false)
    # the particle is named locally for readibility
    particle = model.particles[id]
    # these are used to avoid unnecessary analyses (these analyses, however, are necessary during initial setup)
    particleMoved = initialSetup
    nodeVelocityMustBeFound = initialSetup

    # the velocity and angular velocity are updated, and the particle is moved (this is not necessary in the initial setup)
    if !initialSetup
        deltaP = particle.momentumInput |> sum
        deltaL = particle.angularMomentumInput |> sum
        particle.velocity += particle.particleParams.inverseMass * deltaP
        particle.angularVelocity += particle.particleParams.inverseMomentOfInertia * deltaL
        # the particle will be moved only if the velocity is nonzero! (there are no methods for rotating particles)
        (particle.velocity |> v -> v != zero(v)) && (eulerStep!(id, model); particleMoved = true)
        # the node velocity needs to be found if a) the angular velocity changed, or b) the particle moved
        ((deltaL |> v -> v != zero(v)) || particleMoved) && (nodeVelocityMustBeFound = true)
    end

    # the inputs are reset
    particle.momentumInput = particle.momentumInput .|> zero
    particle.angularMomentumInput = particle.angularMomentumInput .|> zero

    # the projection of the particle onto the lattice is found, along with its boundary nodes (streaming invasion regions)
    if particleMoved
        # the particle discretisation on the lattice is updated
        solidRegion = [particle.particleParams.solidRegionGenerator(x - particle.position) for x in model.spaceTime.X]
        #= solidRegion != particle.boundaryConditionsParams.solidRegion && println("ups") =#
        model.spaceTime.dims < 3 ? (solidRegion = sparse(solidRegion)) : (solidRegion = BitArray(solidRegion))

        particle.boundaryConditionsParams = (; solidRegion);

        if :ladd in model.schemes
            # the new exterior boundary and streaming invasion regions are found
            streamingInvasionRegions = bounceBackPrep(solidRegion, model.velocities; returnStreamingInvasionRegions = true)
            particle.boundaryConditionsParams = merge(particle.boundaryConditionsParams, (; streamingInvasionRegions));
        end

    end

    # the solid velocity (momentum density / mass density) is found
    if nodeVelocityMustBeFound
        particle.nodeVelocity = fill(fill(0., model.spaceTime.dims), size(particle.boundaryConditionsParams.solidRegion))
        particle.nodeVelocity[particle.boundaryConditionsParams.solidRegion] = [
            bulkVelocity(model, particle, model.spaceTime.X[id])
        for id in findall(particle.boundaryConditionsParams.solidRegion)]
    end

    particle.gpuTmp = (;
        #= distributions = model.distributions .|> CuArray{Float64}, =#
        boundaryConditionsParams = (;
            streamingInvasionRegions = particle.boundaryConditionsParams.streamingInvasionRegions |> M -> CuArray(cat(M...; dims = model.spaceTime.dims+1)),
            solidRegion = particle.boundaryConditionsParams.solidRegion |> cu,
        ),
        nodeVelocity = [u[k] for u in particle.nodeVelocity, k in 1:model.spaceTime.dims]|>CuArray{Float64}
    )
end

function moveParticles!(model::LBMmodel)
    for id in eachindex(model.particles)
        moveParticles!(id, model);
    end
end
