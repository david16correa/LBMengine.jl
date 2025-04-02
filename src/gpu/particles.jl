#= ==========================================================================================
=============================================================================================
time step
=============================================================================================
========================================================================================== =#

function moveParticles!(id::Int64, model::LBMmodel; initialSetup = false)
    # the particle is named locally for readibility
    particle = model.particles[id]
    # these are used to avoid unnecessary analyses (these analyses, however, are necessary during initial setup)
    particleMoved = initialSetup
    nodeVelocityMustBeFound = initialSetup

    # the velocity and angular velocity are updated, and the particle is moved (this is not necessary in the initial setup)
    if !initialSetup
        # Verlet Integration
        A = particle.forceInput * particle.particleParams.inverseMass
        Alpha = particle.torqueInput * particle.particleParams.inverseMomentOfInertia

        if model.tick == 0
            particle.velocity += 0.5 * model.spaceTime.timeStep * A
            particle.angularVelocity += 0.5 * model.spaceTime.timeStep * Alpha
        else
            particle.velocity += model.spaceTime.timeStep * A
            particle.angularVelocity += model.spaceTime.timeStep * Alpha
        end

        particle.position += model.spaceTime.timeStep * particle.velocity

        (particle.velocity |> v -> v != zero(v)) && (particleMoved = true)
        ((Alpha |> v -> v != zero(v)) || particleMoved) && (nodeVelocityMustBeFound = true)
    end

    # the inputs are reset
    particle.forceInput = particle.forceInput |> zero
    particle.torqueInput = particle.torqueInput |> zero

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

#= ==========================================================================================
=============================================================================================
molecular dynamics
=============================================================================================
========================================================================================== =#

function performBonds!(model)
    for bond in model.particleBonds
        # as of right now, there are only linear bonds
        if bond.type == :linear
            particle1 = model.particles[bond.id1]
            particle2 = model.particles[bond.id2]

            disp21 = particle2.position - particle1.position
            disp = disp21 |> Array |> norm # this is actually quicker than disp21 |> norm, which I find annoying
            fancyR21 = disp21 / disp

            # F21 := force acting on particle 1 by virtue of its interaction with particle 2
            F21 = bond.hookConstant * (disp - bond.equilibriumDisp) * fancyR21

            particle1.forceInput += F21
            particle2.forceInput -= F21 # F12 := -F21
        else
            particle1 = model.particles[bond.id1]
            particle2 = model.particles[bond.id2]
            particle3 = model.particles[bond.id3]

            disp12 = particle1.position - particle2.position
            normDisp12 = disp12 |> Array |> norm
            unitDisp12 = disp12/normDisp12
            disp32 = particle3.position - particle2.position
            normDisp32 = disp32 |> Array |> norm
            unitDisp32 = disp32/normDisp32

            angle123 = sum(unitDisp12 .* unitDisp32) |> acos

            tau = bond.hookConstant * (angle123 - bond.equilibriumAngle) * cross(unitDisp32, unitDisp12)

            F321 = cross(tau, unitDisp32) / normDisp32
            particle1.forceInput += F321
            #= println(F321) =#

            F123 = -cross(tau, unitDisp12) / normDisp12
            particle3.forceInput += F123
            #= println(F123) =#

            # due to conservation of momentum
            particle2.forceInput -= F321 + F123
        end
    end
end

function molecularDynamicsTick!(model::LBMmodel)
    performBonds!(model)
    #
    for id in eachindex(model.particles)
        moveParticles!(id, model);
    end
end
