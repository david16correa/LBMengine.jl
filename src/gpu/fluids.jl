#= ==========================================================================================
=============================================================================================
gpu accelerated LBM
=============================================================================================
========================================================================================== =#

function massDensityGet(model::LBMmodel)
    # sum will output an array of the same dimensions as its input; the last dimension of the
    # input array corresponds to the auxiliary fluids, which makes it have one dimension more than
    # the desired mass density. Rang solves this issue.
    rang = (((1:s for s in size(model.distributions)[1:end-1]))..., 1)
    return sum(model.distributions, dims=model.spaceTime.dims+1)[rang...]
end

function momentumDensityGet(model::LBMmodel; useEquilibriumScheme = false)
    # en estep punto se dará por hecho que la fuerza es constante!!
    rang(id) = (((1:s for s in size(model.distributions)[1:end-1]))..., id)
    bareMomentum = [scalarFieldTimesVector(model.distributions[rang(id)...], model.velocities.cs[id]) for id in eachindex(model.velocities.cs)] |> sum

    if :guo in model.schemes
        return CUDA.@. bareMomentum + 0.5 * model.spaceTime.Δt * model.forceDensity
    end

    if useEquilibriumScheme
        if :shan in model.schemes
            return CUDA.@. bareMomentum + model.fluidParams.relaxationTime * model.spaceTime.Δt^2 * model.forceDensity
        end
    end

    if :shan in model.schemes
        return CUDA.@. bareMomentum + 0.5 * model.spaceTime.Δt * model.forceDensity
    end

    return bareMomentum
end

function hydroVariablesUpdate!(model::LBMmodel; useEquilibriumScheme = false)
    model.massDensity = massDensityGet(model)
    model.momentumDensity = momentumDensityGet(model; useEquilibriumScheme = useEquilibriumScheme)
    model.fluidVelocity = findFluidVelocity(model.massDensity, model.momentumDensity)

    return nothing
end

function equilibriumDistribution(id::Int64, model::LBMmodel)
    # the quantities to be used are saved separately
    ci = model.velocities.cs[id] .* model.spaceTime.Δx_Δt
    wi = model.velocities.ws[id]

    u = model.fluidVelocity

    # the equilibrium distribution is found step by step and returned
    firstStep = vectorFieldDotVector(u, ci) |> udotci -> udotci*model.fluidParams.invC2_s + udotci.^2 * (0.5 * model.fluidParams.invC4_s)
    secondStep = firstStep - vectorFieldDotVectorField(u, u)*(0.5*model.fluidParams.invC2_s)

    model.fluidParams.isFluidCompressible && return wi * ((secondStep .+ 1) .* model.massDensity)
    return wi * model.massDensity + wi * (model.initialConditions.massDensity .* secondStep)
end

function nonEquilibriumDistribution(id::Int64, model::LBMmodel)
    rang = (((1:s for s in size(model.distributions)[1:end-1]))..., id)
    return model.distributions[rang...] - equilibriumDistribution(id, model)
end

function collisionStep(model::LBMmodel)
    # preliminary preparation
    rowsCols = size(model.distributions)
    rang(id) = (((1:s for s in rowsCols[1:end-1]))..., id)

    # the equilibrium distributions are found
    equilibriumDistributions = CuArray{eltype(model.distributions)}(undef, rowsCols) # Output matrix
    for id in eachindex(model.velocities.cs)
        equilibriumDistributions[rang(id)...] = equilibriumDistribution(id, model)
    end

    # the collision operator is found
    if :bgk in model.schemes
        # the Bhatnagar-Gross-Krook collision opeartor is used
        Omega = -model.spaceTime.Δt/model.fluidParams.relaxationTime * (model.distributions - equilibriumDistributions)
    elseif :trt in model.schemes
        # the two-relaxation-time collision opeartor is used
        conjugateIds = model.boundaryConditionsParams.oppositeVectorId

        conjugateDistributions = model.distributions[rang(conjugateIds)...]
        distributionsPlus = 0.5 * (model.distributions + conjugateDistributions)
        distributionsMinus = 0.5 * (model.distributions - conjugateDistributions)

        conjugateDistributions = equilibriumDistributions[rang(conjugateIds)...]
        equilibriumDistributionsPlus = 0.5 * (equilibriumDistributions + conjugateDistributions)
        equilibriumDistributionsMinus = 0.5 * (equilibriumDistributions - conjugateDistributions)

        OmegaPlus = -model.spaceTime.Δt/model.fluidParams.relaxationTimePlus * (distributionsPlus - equilibriumDistributionsPlus)
        OmegaMinus = -model.spaceTime.Δt/model.fluidParams.relaxationTimeMinus * (distributionsMinus - equilibriumDistributionsMinus)

        Omega = OmegaPlus + OmegaMinus
    end

    # forcing terms are added
    if :guo in model.schemes
        for id in eachindex(model.velocities.cs)
            Omega[rang(id)...] += guoForcingTerm(id, model)
        end
    end

    # the collision step is performed
    Omega += model.distributions

    return Omega
end

function guoForcingTerm(id::Int64, model::LBMmodel)
    # the quantities to be used are saved separately
    ci = model.velocities.cs[id] .* model.spaceTime.Δx_Δt
    invC2_s = model.fluidParams.invC2_s
    wi = model.velocities.ws[id]
    U = model.fluidVelocity
    F = model.forceDensity
    Δt = model.spaceTime.Δt

    # the forcing term is found (terms common for both collision methods are found first)
    secondTerm = vectorFieldDotVector(U, ci) |> udotci -> scalarFieldTimesVector(udotci, ci)*model.fluidParams.invC4_s
    if :bgk in model.schemes
        firstTerm = vectorFieldPlusVector(-U, ci) * invC2_s
        intermediateStep = vectorFieldDotVectorField(firstTerm + secondTerm, F)
        return wi * Δt * (1 - Δt/(2 * model.fluidParams.relaxationTime)) * intermediateStep
    elseif :trt in model.schemes
        intermediateStep = vectorFieldDotVectorField(-U*invC2_s + secondTerm, F)
        plusPart = (1 - Δt/(2 * model.fluidParams.relaxationTimePlus)) * intermediateStep

        intermediateStep = vectorFieldDotVector(F, ci*invC2_s)
        minusPart = (1 - Δt/(2 * model.fluidParams.relaxationTimeMinus)) * intermediateStep

        return wi * Δt * (plusPart + minusPart)
    end
end

#= function cbcBoundaries_gpu!(model::LBMmodel) =#
#= end =#

function tick!(model::LBMmodel)
    # preliminary preparation
    rowsCols = size(model.distributions)
    rang(id) = (((1:s for s in rowsCols[1:end-1]))..., id)

    # collision (or relaxation)
    collisionedDistributions = collisionStep(model);

    # propagated distributions will be saved in a new vector
    propagatedDistributions = CuArray{eltype(collisionedDistributions)}(undef, rowsCols);

    # streaming (or propagation), with streaming invasion exchange
    for id in eachindex(model.velocities.cs)
        # distributions are initially streamed
        streamedDistribution = circshift_gpu(collisionedDistributions[rang(id)...], model.velocities.cs[id])

        if :bounceBack in model.schemes
            # the wall region, the invasion region of the fluid, and the conjugate of the fluid with opposite momentum are retrieved
            wallRegion = model.boundaryConditionsParams.wallRegion;
            conjugateId = model.boundaryConditionsParams.oppositeVectorId[id];
            conjugateInvasionRegion = model.boundaryConditionsParams.streamingInvasionRegions[rang(conjugateId)...];

            # the wall regions are imposed to vanish
            streamedDistribution[wallRegion] .= 0;

            # the boudnary nodes of all rigid moving particles are considered in the streaming invasion exchange step
            if :ladd in model.schemes
                for particle in model.particles
                    wallRegion = wallRegion .|| particle.boundaryConditionsParams.solidRegion
                    conjugateInvasionRegion = conjugateInvasionRegion .|| particle.boundaryConditionsParams.streamingInvasionRegions[rang(conjugateId)...]
                end
            end

            # streaming invasion exchange step is performed
            streamedDistribution[conjugateInvasionRegion] = collisionedDistributions[rang(conjugateId)...][conjugateInvasionRegion]

            # if any wall is moving, its momentum is transfered to the fluid
            if :movingWalls in model.schemes
                ci = model.velocities.cs[id].c .* model.spaceTime.Δx_Δt
                wi = model.velocities.ws[id]

                uwdotci = circshift_gpu(model.boundaryConditionsParams.solidNodeVelocity, model.velocities.cs[id]) |> uw -> vectorFieldDotVector(uw,ci)
                streamedDistribution[conjugateInvasionRegion] += (2 * wi * model.fluidParams.invC2_s) * model.massDensity[conjugateInvasionRegion] .* uwdotci[conjugateInvasionRegion]
            end

            # if any particle is moving, its momentum is exchanged with the fluid
            if :ladd in model.schemes
                ci = model.velocities.cs[id] .* model.spaceTime.Δx_Δt
                wi = model.velocities.ws[id]
                for particle in model.particles
                    conjugateBoundaryNodes = particle.boundaryConditionsParams.streamingInvasionRegions[rang(conjugateId)...]

                    # if the fluid did not bump into the particle, then the entire scheme can be skipped
                    sum(conjugateBoundaryNodes) == 0 && break

                    # the solids momentum is transfered to the fluid
                    uw = particle.nodeVelocity + circshift_gpu(particle.nodeVelocity, model.velocities.cs[id])
                    uwdotci = vectorFieldDotVector(uw,ci)
                    streamedDistribution[conjugateBoundaryNodes] += (2 * wi * model.fluidParams.invC2_s) * model.massDensity[conjugateBoundaryNodes] .* uwdotci[conjugateBoundaryNodes]

                    # if the solid is coupled to forces or torques, the fluids momentum is transfered to it
                    if particle.particleParams.coupleForces || particle.particleParams.coupleTorques
                        sumTerm = collisionedDistributions[rang(conjugateId)...][conjugateBoundaryNodes] + streamedDistribution[conjugateBoundaryNodes]
                        particle.particleParams.coupleForces && (particle.momentumInput -= model.spaceTime.latticeParameter^model.spaceTime.dims * sum(sumTerm) * ci |> Array)
                        particle.particleParams.coupleTorques && (particle.angularMomentumInput -= model.spaceTime.latticeParameter^model.spaceTime.dims * cross(
                            sum(sumTerm .* [x - particle.position for x in model.spaceTime.X[conjugateBoundaryNodes]]), ci
                        )|> Array)
                    end
                end
            end
        end

        # the resulting propagation is appended to the propagated distributions
        propagatedDistributions[rang(id)...] = streamedDistribution
    end

    # the distributions and time are updated
    model.distributions = propagatedDistributions
    model.time += model.spaceTime.Δt
    model.tick += 1

    # characteristic boundary conditions
    :cbc in model.schemes && cbcBoundaries!(model)

    # Finally, the hydrodynamic variables are updated
    hydroVariablesUpdate!(model; useEquilibriumScheme = true);
    # and particles are moved, if there are any
    (:ladd in model.schemes || :psm in model.schemes) && (moveParticles!(model));
end

function LBMpropagate!(model::LBMmodel; simulationTime = :default, ticks = :default, verbose = false, ticksSaved = :default)
    verbose && (println("CUDA.functional() = true"))

    @assert any(x -> x == :default, [simulationTime, ticks]) "simulationTime and ticks cannot be simultaneously chosen, as the time step is defined already in the model!"
    if simulationTime == :default && ticks == :default
        time = range(model.spaceTime.Δt, length = 100, step = model.spaceTime.Δt);
    elseif ticks == :default
        time = range(model.spaceTime.Δt, stop = simulationTime::Number, step = model.spaceTime.Δt);
    else
        time = range(model.spaceTime.Δt, length = ticks::Int64, step = model.spaceTime.Δt);
    end

    simulationTime = time[end];
    ticks = length(time);

    if :saveData in model.schemes
        (ticksSaved == :default) && (ticksSaved = 100);
        totalTicks = floor(simulationTime/model.spaceTime.Δt)
        checkPoints = range(0, stop=totalTicks, length=ticksSaved) |> collect .|> round .|> Int64
    end

    verbose && (outputTimes = range(1, stop = length(time), length = 50) |> collect .|> round)

    for t in time |> eachindex
        tick!(model);
        verbose && (t in outputTimes) && (print("\r t = $(round(model.time; digits = 2))"); flush(stdout))
        if :saveData in model.schemes
            (model.tick in checkPoints) && writeTrajectories(model)
            # if there are particles in the system, their trajectories are stored as well;
            # since storage is not an issue here, all ticks are saved
            writeParticlesTrajectories(model)
        end
    end
    print("\r");

    return nothing
end

#= ==========================================================================================
=============================================================================================
rheology
=============================================================================================
========================================================================================== =#

function viscousStressTensor(model)
    # the non-equilibrium distributions are found
    nonEquilibriumDistributions = [nonEquilibriumDistribution(id, model) for id in eachindex(model.velocities.cs)]

    # the (1 - Δt/2τ) factor is found beforehand
    coeff = 1 - 0.5*model.spaceTime.Δt/model.fluidParams.relaxationTime

    # the viscous stress tensor is returned
    return [
        -coeff * sum(model.velocities.cs[id][alpha] * model.velocities.cs[id][beta] * nonEquilibriumDistributions[id] for id in eachindex(model.velocities.cs))
    for alpha in 1:model.spaceTime.dims, beta in 1:model.spaceTime.dims]

    return nothing
end
