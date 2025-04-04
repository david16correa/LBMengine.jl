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
            # disp21 := displacement from particle 1 to particle 2
            disp21 = model.particles[bond.id2].position - model.particles[bond.id1].position
            disp = disp21 |> Array |> norm # this is actually quicker than disp21 |> norm, which I find annoying
            unitDisp21 = disp21 / disp

            # force21 := force acting on particle 1 by virtue of its interaction with particle 2
            force21 = bond.hookConstant * (disp - bond.equilibriumDisp) * unitDisp21

            model.particles[bond.id1].forceInput += force21
            model.particles[bond.id2].forceInput -= force21 # force12 := -force21
        else
            # disp12 := displacement from particle 2 to particle 1
            disp12 = model.particles[bond.id1].position - model.particles[bond.id2].position
            normDisp12 = disp12 |> Array |> norm
            unitDisp12 = disp12/normDisp12
            disp32 = model.particles[bond.id3].position - model.particles[bond.id2].position
            normDisp32 = disp32 |> Array |> norm
            unitDisp32 = disp32/normDisp32

            angle123 = sum(unitDisp12 .* unitDisp32) |> acos
            torque = bond.hookConstant * (angle123 - bond.equilibriumAngle) * cross(unitDisp32, unitDisp12)

            # force321 := force acting on particle 1 by virtue of its interaction with particles 2 and 3
            force321 = -cross(torque, unitDisp12) / normDisp12
            model.particles[bond.id1].forceInput += force321

            force123 = cross(torque, unitDisp32) / normDisp32
            model.particles[bond.id3].forceInput += force123

            model.particles[bond.id2].forceInput -= force321 + force123 # conservation of momentum
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
