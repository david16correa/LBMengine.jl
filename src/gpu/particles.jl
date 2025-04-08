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
        # Verlet Integration - translation
        if model.particles[id].particleParams.coupleForces
            A = particle.forceInput * particle.particleParams.inverseMass
            if model.tick == 0
                particle.velocity += 0.5 * model.spaceTime.timeStep * A
            else
                particle.velocity += model.spaceTime.timeStep * A
            end

            particle.position += model.spaceTime.timeStep * particle.velocity

            (particle.velocity |> v -> v != zero(v)) && (particleMoved = true)
        end

        # Verlet Integration - rotation
        if model.particles[id].particleParams.coupleTorques
            Alpha = particle.torqueInput * particle.particleParams.inverseMomentOfInertia
            if model.tick == 0
                particle.angularVelocity += 0.5 * model.spaceTime.timeStep * Alpha
            else
                particle.angularVelocity += model.spaceTime.timeStep * Alpha
            end

            # for squirmers, the swimming direction will rotate around. To achieve this, rodrigues'
            # rotation formula is used - https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
            if :squirmer in particle.particleParams.properties
                # v := vector to be rotated
                v = particle.particleParams.swimmingDirection
                # θ := |ω| Δt; since |ω| will be needed shortly, the time step is not multiplied just yet
                theta = particle.angularVelocity |> norm
                # if the rotation is zero then the entire scheme is skipped
                if !(theta ≈ 0)
                    # k is the unit vector describing the axis of rotation about which v will be rotated
                    k = particle.angularVelocity/theta
                    theta *= model.spaceTime.timeStep
                    # the first two terms of rodrigues' rotation formula can be used both in 2D and 3D
                    vRot = v * cos(theta) + cross(k, v) * sin(theta)
                    if model.spaceTime.dims == 3
                        # this last term will always be zero in 2D, as k̂ = ẑ
                        vRot += k * dot(k,v) * (1 - cos(theta))
                    end
                    # the swimming direction is updated
                    particle.particleParams = (; particle.particleParams..., swimmingDirection = vRot / norm(vRot))
                end
            end

            (Alpha |> v -> v != zero(v)) && (nodeVelocityMustBeFound = true)
        end

        nodeVelocityMustBeFound = particleMoved || nodeVelocityMustBeFound
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

function parformInteractions!(model)
    for interaction in model.particleInteractions
        if interaction.type == :linear
            # disp21 := displacement from particle 1 to particle 2
            disp21 = model.particles[interaction.id2].position - model.particles[interaction.id1].position
            disp = disp21 |> Array |> norm # this is actually quicker than disp21 |> norm, which I find annoying
            unitDisp21 = disp21 / disp

            # force21 := force acting on particle 1 by virtue of its interaction with particle 2
            force21 = interaction.hookConstant * (disp - interaction.equilibriumDisp) * unitDisp21

            model.particles[interaction.id1].forceInput += force21
            model.particles[interaction.id2].forceInput -= force21 # Newton's third law
        elseif interaction.type == :polar
            # disp12 := displacement from particle 2 to particle 1
            disp12 = model.particles[interaction.id1].position - model.particles[interaction.id2].position
            normDisp12 = disp12 |> Array |> norm
            unitDisp12 = disp12/normDisp12
            disp32 = model.particles[interaction.id3].position - model.particles[interaction.id2].position
            normDisp32 = disp32 |> Array |> norm
            unitDisp32 = disp32/normDisp32

            angle123 = sum(unitDisp12 .* unitDisp32) |> x -> (abs(x)>1 ? sign(x) : x) |> acos
            torque = interaction.hookConstant * (angle123 - interaction.equilibriumAngle) * cross(unitDisp32, unitDisp12)

            # force321 := force acting on particle 1 by virtue of its interaction with particles 2 and 3
            force321 = -cross(torque, unitDisp12) / normDisp12
            model.particles[interaction.id1].forceInput += force321

            force123 = cross(torque, unitDisp32) / normDisp32
            model.particles[interaction.id3].forceInput += force123

            model.particles[interaction.id2].forceInput -= force321 + force123 # conservation of momentum
        elseif interaction.type == :dipoleDipole
            B = interaction.B(model.time)
            if !(norm(B) ≈ 0)
                for (id1,id2) in interaction.pairs
                    # disp21 := displacement from particle 1 to particle 2
                    disp21 = model.particles[id2].position - model.particles[id1].position
                    disp = disp21 |> Array |> norm
                    unitDisp21 = disp21 / disp
                    force21 = -interaction.dipoleConstant / disp^4 * (
                        2 * cross(cross(disp21, B), B)
                        - 2 * dot(B,B) * disp21
                        + 5 * disp21 * (cross(disp21,B) |> v -> dot(v,v))
                    )
                    model.particles[id1].forceInput += force21
                    model.particles[id2].forceInput -= force21 # Newton's third law
                end
            end
        elseif interaction.type == :bistable
            # disp21 := displacement from particle 1 to particle 2
            disp21 = model.particles[interaction.id2].position - model.particles[interaction.id1].position
            disp = disp21 |> Array |> norm # this is actually quicker than disp21 |> norm, which I find annoying
            unitDisp21 = disp21 / disp

            # force21 := force acting on particle 1 by virtue of its interaction with particle 2
            force21 = (interaction.fourA * (disp - interaction.trapRadius)^3 - interaction.twoB * (disp - interaction.trapRadius)) * unitDisp21

            model.particles[interaction.id1].forceInput += force21
            model.particles[interaction.id2].forceInput -= force21 # Newton's third law
        end
    end
end

function molecularDynamicsTick!(model::LBMmodel)
    parformInteractions!(model)
    #
    for id in eachindex(model.particles)
        moveParticles!(id, model);
    end
end
