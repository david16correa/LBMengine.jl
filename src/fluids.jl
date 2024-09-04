#= ==========================================================================================
=============================================================================================
general methods
=============================================================================================
========================================================================================== =#

function massDensityGet(model::LBMmodel; time = :default)
    if time == :default
        return sum(distribution for distribution in model.distributions[end])
    else
        return sum(distribution for distribution in model.distributions[time])
    end
end
 
function momentumDensityGet(model::LBMmodel; time = :default, useEquilibriumScheme = false)
    # en estep punto se dará por hecho que la fuerza es constante!!
    if time == :default
        bareMomentum = sum(scalarFieldTimesVector(model.distributions[end][id], model.velocities[id].c) for id in eachindex(model.velocities))
    else
        bareMomentum = sum(scalarFieldTimesVector(model.distributions[time][id], model.velocities[id].c) for id in eachindex(model.velocities))
    end

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

function hydroVariablesUpdate!(model::LBMmodel; time = :default, useEquilibriumScheme = false)
    model.massDensity = massDensityGet(model; time = time)
    model.momentumDensity = momentumDensityGet(model; time = time, useEquilibriumScheme = useEquilibriumScheme)
    model.fluidVelocity = [[0.; 0] for _ in model.massDensity]
    fluidIndices = (model.massDensity .≈ 0) .|> b -> !b;
    model.fluidVelocity[fluidIndices] = model.momentumDensity[fluidIndices] ./ model.massDensity[fluidIndices]
end

#= ==========================================================================================
=============================================================================================
lattice Boltzmann dynamics
=============================================================================================
========================================================================================== =#

function equilibriumDistribution(id::Int64, model::LBMmodel)
    # the quantities to be used are saved separately
    ci = model.velocities[id].c .* model.spaceTime.Δx_Δt
    wi = model.velocities[id].w
    # the equilibrium distribution is found step by step and returned
    firstStep = vectorFieldDotVector(model.fluidVelocity, ci) |> udotci -> udotci/model.fluidParams.c2_s + udotci.^2 / (2 * model.fluidParams.c4_s)
    secondStep = firstStep - vectorFieldDotVectorField(model.fluidVelocity, model.fluidVelocity)/(2*model.fluidParams.c2_s)

    model.fluidParams.isFluidCompressible && return wi * ((secondStep .+ 1) .* model.massDensity)

    return wi * model.massDensity + wi * (model.initialConditions.massDensity .* secondStep)
end

"calculates Ω at the last recorded time!"
function collisionOperator(id::Int64, model::LBMmodel)
    # the Bhatnagar-Gross-Krook collision opeartor is used
    BGK = -model.distributions[end][id] + equilibriumDistribution(id, model) |> f -> model.spaceTime.Δt/model.fluidParams.relaxationTime * f

    # forcing terms are added
    if :guo in model.schemes
        return BGK + guoForcingTerm(id, model)
    end

    return BGK
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
    collisionedDistributions = [model.distributions[end][id] .+ collisionOperator(id, model) for id in eachindex(model.velocities)] 

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
                    particleBoundaryNodes = particle.boundaryConditionsParams.streamingInvasionRegions[model.boundaryConditionsParams.oppositeVectorId[id]]

                    # if the fluid did not bump into the particle, then the entire scheme can be skipped
                    sum(particleBoundaryNodes) == 0 && break

                    # the solids momentum is transfered to the fluid
                    uwdotci = pbcMatrixShift(particle.nodeVelocity, model.velocities[id].c) |> uw -> vectorFieldDotVector(uw,ci)
                    streamedDistribution[particleBoundaryNodes] += (2 * wi / model.fluidParams.c2_s) * model.massDensity[particleBoundaryNodes] .* uwdotci[particleBoundaryNodes]

                    # the fluids momentum is transfered to the solid
                    sumTerm = collisionedDistributions[conjugateId][particleBoundaryNodes] + streamedDistribution[particleBoundaryNodes]
                    particle.momentumInput -= model.spaceTime.Δx^model.spaceTime.dims * sum(sumTerm) * ci
                    particle.angularMomentumInput -= model.spaceTime.Δx^model.spaceTime.dims * cross(sum(sumTerm .* [x - particle.position for x in model.spaceTime.X[particleBoundaryNodes]]), ci)
                end
            end
        end

        # the resulting propagation is appended to the propagated distributions
        append!(propagatedDistributions, [streamedDistribution]);
    end

    # the new distributions and time are appended
    append!(model.distributions, [propagatedDistributions]);
    append!(model.time, [model.time[end]+model.spaceTime.Δt]);

    # Finally, the hydrodynamic variables are updated
    hydroVariablesUpdate!(model; useEquilibriumScheme = true);
    # and particles are moved, if there are any
    :ladd in model.schemes && (moveParticles!(model));
end

function LBMpropagate!(model::LBMmodel; simulationTime = :default, ticks = :default, verbose = false)
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

    for t in time |> eachindex
        tick!(model);
        verbose && t in outputTimes && print("\r t = $(model.time[end])")
    end
    print("\r");

    hydroVariablesUpdate!(model);
end
