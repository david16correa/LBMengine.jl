#= ==========================================================================================
=============================================================================================
general methods
=============================================================================================
========================================================================================== =#

function massDensityGet(model::LBMmodel)
    return sum(distribution for distribution in model.distributions)
end
 
function momentumDensityGet(model::LBMmodel; useEquilibriumScheme = false)
    # en estep punto se dará por hecho que la fuerza es constante!!
    bareMomentum = sum(scalarFieldTimesVector(model.distributions[id], model.velocities[id].c) for id in eachindex(model.velocities))

    if useEquilibriumScheme
        if :shan in model.schemes
            return bareMomentum + model.fluidParams.relaxationTime * model.spaceTime.Δt^2 * model.forceDensity
        end
        if :guo in model.schemes
            return bareMomentum + 0.5 * model.spaceTime.Δt * model.forceDensity
        end
    end

    if :shan in model.schemes || :guo in model.schemes
        return bareMomentum + 0.5 * model.spaceTime.Δt * model.forceDensity
    end

    return bareMomentum
end

function hydroVariablesUpdate!(model::LBMmodel; useEquilibriumScheme = false)
    model.massDensity = massDensityGet(model)
    model.momentumDensity = momentumDensityGet(model; useEquilibriumScheme = useEquilibriumScheme)
    model.fluidVelocity = [[0.; 0] for _ in model.massDensity]
    fluidIndices = (model.massDensity .≈ 0) .|> b -> !b;
    model.fluidVelocity[fluidIndices] = model.momentumDensity[fluidIndices] ./ model.massDensity[fluidIndices]
end

#= ==========================================================================================
=============================================================================================
lattice Boltzmann dynamics
=============================================================================================
========================================================================================== =#

function equilibriumDistribution(id::Int64, model::LBMmodel; particleId = :default)
    # the quantities to be used are saved separately
    ci = model.velocities[id].c .* model.spaceTime.Δx_Δt
    wi = model.velocities[id].w

    if particleId isa Int64
        u = model.particles[particleId].nodeVelocity
    else
        u = model.fluidVelocity
    end
    # the equilibrium distribution is found step by step and returned
    firstStep = vectorFieldDotVector(u, ci) |> udotci -> udotci/model.fluidParams.c2_s + udotci.^2 / (2 * model.fluidParams.c4_s)
    secondStep = firstStep - vectorFieldDotVectorField(u, u)/(2*model.fluidParams.c2_s)

    model.fluidParams.isFluidCompressible && return wi * ((secondStep .+ 1) .* model.massDensity)

    return wi * model.massDensity + wi * (model.initialConditions.massDensity .* secondStep)
end

#= "calculates Ω at the last recorded time!" =#
#= function collisionOperator(id::Int64, model::LBMmodel) =#
#=     # the Bhatnagar-Gross-Krook collision opeartor is used =#
#=     BGK = -model.distributions[id] + equilibriumDistribution(id, model) |> f -> model.spaceTime.Δt/model.fluidParams.relaxationTime * f =#
#==#
#=     # forcing terms are added =#
#=     if :guo in model.schemes =#
#=         return BGK + guoForcingTerm(id, model) =#
#=     end =#
#==#
#=     return BGK =#
#= end =#

function collisionStep(model::LBMmodel)
    # the equilibrium distributions are found
    equilibriumDistributions = [equilibriumDistribution(id, model) for id in eachindex(model.velocities)]

    # the Bhatnagar-Gross-Krook collision opeartor is used
    Omega = [-model.spaceTime.Δt/model.fluidParams.relaxationTime * (model.distributions[id] - equilibriumDistributions[id]) for id in eachindex(model.velocities)]

    if :psm in model.schemes
        for particle in model.particles
            equilibriumDistributions_solidNodeVelocity = [equilibriumDistribution(id, model; particleId = particle.id) for id in eachindex(model.velocities)]
            E = particle.boundaryConditionsParams.solidRegion |> Array
            B = model.fluidParams.relaxationTime/model.spaceTime.Δt - 0.5 |> tau_minus_half -> E * tau_minus_half ./ ((1 .- E) .+ tau_minus_half)
            OmegaS = [ (model.boundaryConditionsParams.oppositeVectorId[id] |> conjugateId -> 
                (model.distributions[conjugateId] - equilibriumDistributions[conjugateId] - model.distributions[id] + equilibriumDistributions_solidNodeVelocity[id])
            ) for id in eachindex(model.velocities)]
            Omega = [(1 .- B) .* Omega[id] + B .* OmegaS[id] for id in eachindex(model.velocities)]

            # if the solid is coupled to forces or torques, the fluids momentum is transfered to it
            if particle.particleParams.coupleForces || particle.particleParams.coupleTorques
                for id in eachindex(model.velocities)[2:end]
                    ci = model.velocities[id].c .* model.spaceTime.Δx_Δt
                    sumTerm = B .* OmegaS[id]
                    particle.particleParams.coupleForces && (particle.momentumInput -= model.spaceTime.Δx^model.spaceTime.dims * sum(sumTerm) * ci)
                    particle.particleParams.coupleTorques && (particle.angularMomentumInput -= model.spaceTime.Δx^model.spaceTime.dims * cross(
                        sum(sumTerm .* [x - particle.position for x in model.spaceTime.X[conjugateBoundaryNodes]]), ci
                    ))
                end
            end
        end
    end
    # forcing terms are added
    if :guo in model.schemes
        Omega = [Omega[id] + guoForcingTerm(id, model) for id in eachindex(model.velocities)]
    end

    return [model.distributions[id] + Omega[id] for id in eachindex(model.velocities)] 
end

function guoForcingTerm(id::Int64, model::LBMmodel)
    # the quantities to be used are saved separately
    ci = model.velocities[id].c .* model.spaceTime.Δx_Δt
    wi = model.velocities[id].w
    # the forcing term is found
    firstTerm = [ci - u for u in model.fluidVelocity] / model.fluidParams.c2_s
    secondTerm = vectorFieldDotVector(model.fluidVelocity, ci) |> udotci -> [ci * v for v in udotci]/model.fluidParams.c4_s
    intermediateStep = vectorFieldDotVectorField(firstTerm + secondTerm, model.forceDensity)
    return (1 - model.spaceTime.Δt/(2 * model.fluidParams.relaxationTime)) * wi * intermediateStep * model.spaceTime.Δt
end

#= ==========================================================================================
=============================================================================================
time evolution
=============================================================================================
========================================================================================== =#

function tick!(model::LBMmodel)
    # collision (or relaxation)
    collisionedDistributions = collisionStep(model)

    # propagated distributions will be saved in a new vector
    propagatedDistributions = [] |> LBMdistributions ;

    # streaming (or propagation), with streaming invasion exchange
    for id in eachindex(model.velocities)
        # distributions are initially streamed
        streamedDistribution = pbcMatrixShift(collisionedDistributions[id], model.velocities[id].c)

        if :bounceBack in model.schemes
            # the wall region, the invasion region of the fluid, and the conjugate of the fluid with opposite momentum are retrieved
            wallRegion = model.boundaryConditionsParams.wallRegion
            conjugateInvasionRegion, conjugateId = model.boundaryConditionsParams |> params -> (params.streamingInvasionRegions[params.oppositeVectorId[id]], params.oppositeVectorId[id])

            # the wall regions are imposed to vanish
            streamedDistribution[wallRegion] .= 0;

            # the boudnary nodes of all rigid moving particles are considered in the streaming invasion exchange step
            if :ladd in model.schemes
                for particle in model.particles
                    wallRegion = wallRegion .|| particle.boundaryConditionsParams.solidRegion
                    conjugateInvasionRegion = conjugateInvasionRegion .|| particle.boundaryConditionsParams.streamingInvasionRegions[model.boundaryConditionsParams.oppositeVectorId[id]]
                end
            end

            # streaming invasion exchange step is performed
            streamedDistribution[conjugateInvasionRegion] = collisionedDistributions[conjugateId][conjugateInvasionRegion]

            # if any wall is moving, its momentum is transfered to the fluid
            if :movingWalls in model.schemes
                ci = model.velocities[id].c .* model.spaceTime.Δx_Δt
                wi = model.velocities[id].w

                uwdotci = pbcMatrixShift(model.boundaryConditionsParams.solidNodeVelocity, model.velocities[id].c) |> uw -> vectorFieldDotVector(uw,ci)
                streamedDistribution[conjugateInvasionRegion] += (2 * wi / model.fluidParams.c2_s) * model.massDensity[conjugateInvasionRegion] .* uwdotci[conjugateInvasionRegion]
            end

            # if any particle is moving, its momentum is exchanged with the fluid
            if :ladd in model.schemes
                ci = model.velocities[id].c .* model.spaceTime.Δx_Δt
                wi = model.velocities[id].w
                for particle in model.particles
                    conjugateBoundaryNodes = particle.boundaryConditionsParams.streamingInvasionRegions[model.boundaryConditionsParams.oppositeVectorId[id]]

                    # if the fluid did not bump into the particle, then the entire scheme can be skipped
                    sum(conjugateBoundaryNodes) == 0 && break

                    # the solids momentum is transfered to the fluid
                    uw = particle.nodeVelocity + pbcMatrixShift(particle.nodeVelocity, model.velocities[id].c)
                    uwdotci = vectorFieldDotVector(uw,ci)
                    streamedDistribution[conjugateBoundaryNodes] += (2 * wi / model.fluidParams.c2_s) * model.massDensity[conjugateBoundaryNodes] .* uwdotci[conjugateBoundaryNodes]

                    # if the solid is coupled to forces or torques, the fluids momentum is transfered to it
                    if particle.particleParams.coupleForces || particle.particleParams.coupleTorques
                        sumTerm = collisionedDistributions[conjugateId][conjugateBoundaryNodes] + streamedDistribution[conjugateBoundaryNodes]
                        particle.particleParams.coupleForces && (particle.momentumInput -= model.spaceTime.Δx^model.spaceTime.dims * sum(sumTerm) * ci)
                        particle.particleParams.coupleTorques && (particle.angularMomentumInput -= model.spaceTime.Δx^model.spaceTime.dims * cross(
                            sum(sumTerm .* [x - particle.position for x in model.spaceTime.X[conjugateBoundaryNodes]]), ci
                        ))
                    end
                end
            end
        end

        # the resulting propagation is appended to the propagated distributions
        append!(propagatedDistributions, [streamedDistribution]);
    end

    # the new distributions and time are appended
    model.distributions = propagatedDistributions
    model.time += model.spaceTime.Δt
    model.tick += 1

    # Finally, the hydrodynamic variables are updated
    hydroVariablesUpdate!(model; useEquilibriumScheme = true);
    # and particles are moved, if there are any
    :ladd in model.schemes || :psm in model.schemes && (moveParticles!(model));
end

function LBMpropagate!(model::LBMmodel; simulationTime = :default, ticks = :default, verbose = false, ticksBetweenSaves = 10)
    if simulationTime != :default && ticks != :default
        error("simulationTime and ticks cannot be simultaneously chosen, as the time step is defined already in the model!")
    elseif simulationTime == :default && ticks == :default
        time = range(model.spaceTime.Δt, length = 100, step = model.spaceTime.Δt);
    elseif ticks == :default
        time = range(model.spaceTime.Δt, stop = simulationTime::Number, step = model.spaceTime.Δt);
    else
        time = range(model.spaceTime.Δt, length = ticks::Int64, step = model.spaceTime.Δt);
    end

    verbose && (outputTimes = range(1, stop = length(time), length = 50) |> collect .|> round)

    if :saveData in model.schemes
        writingTimes = range(1, stop = length(time), step = ticksBetweenSaves) |> collect
        writingTimes[end] != length(time) && @warn "Simulation time will be $(time[writingTimes[end]]), as this is the last snapshot that will be recorded in the data."
        time = time[1:writingTimes[end]]
    end

    for t in time |> eachindex
        tick!(model);
        verbose && t in outputTimes && print("\r t = $(model.time)")
        :saveData in model.schemes && t in writingTimes && writeTrajectories(model)
    end
    print("\r");

    hydroVariablesUpdate!(model);
end
