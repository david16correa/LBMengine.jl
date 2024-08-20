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

"Time evolution"
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
            # the wall regions are imposed to vanish
            streamedDistribution[model.boundaryConditionsParams.wallRegion] .= 0;
            # the invasion region of the fluid with opposite momentum is retrieved
            conjugateInvasionRegion, conjugateId = model.boundaryConditionsParams |> params -> (params.streamingInvasionRegions[params.oppositeVectorId[id]], params.oppositeVectorId[id])
            # streaming invasion exchange step is performed
            streamedDistribution[conjugateInvasionRegion] = collisionedDistributions[conjugateId][conjugateInvasionRegion]

            if :movingWalls in model.schemes
                ci = model.velocities[id].c .* model.spaceTime.Δx_Δt
                wi = model.velocities[id].w
                uwdotci = pbcMatrixShift(model.boundaryConditionsParams.solidNodeVelocity, model.velocities[id].c) |> uw -> vectorFieldDotVector(uw,ci)

                streamedDistribution[conjugateInvasionRegion] += (2 * wi / model.fluidParams.c2_s) * model.massDensity[conjugateInvasionRegion] .* uwdotci[conjugateInvasionRegion]
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

#= ==========================================================================================
=============================================================================================
init methods
=============================================================================================
========================================================================================== =#

"Initializes f_i to f^eq_i, which is the simplest strategy."
function findInitialConditions(id::Int64, velocities::Vector{LBMvelocity}, fluidParams::NamedTuple, massDensity::Array{Float64}, u::Array{Vector{Float64}}, Δx_Δt::Float64; kwInitialConditions = (; )) 
    # the quantities to be used are saved separately
    ci = velocities[id].c .* Δx_Δt
    wi = velocities[id].w
    if :forceDensity in (kwInitialConditions |> keys)
        consistencyTerm = [[0.; 0] for _ in massDensity]
        fluidIndices = (massDensity .≈ 0) .|> b -> !b;
        consistencyTerm[fluidIndices] = kwInitialConditions.forceDensity[fluidIndices] ./ massDensity[fluidIndices]
        u -= 0.5 * kwInitialConditions.Δt * consistencyTerm
    end
    # the equilibrium distribution is found step by step and returned
    firstStep = vectorFieldDotVector(u, ci) |> udotci -> udotci/fluidParams.c2_s + udotci.^2 / (2 * fluidParams.c4_s)
    secondStep = firstStep - vectorFieldDotVectorField(u, u)/(2*fluidParams.c2_s) .+ 1
    return secondStep .* (wi * massDensity)
end

function modelInit(;
    massDensity = :default, # default: ρ(x) = 1
    fluidVelocity = :default, # default: u(x) = 0
    velocities = :default, # default: chosen by dimensionality (D1Q3, D2Q9, or D3Q27)
    relaxationTimeRatio = 0.8, # τ/Δt > 1 → under-relaxation, τ/Δt = 1 → full relaxation, 0.5 < τ/Δt < 1 → over-relaxation, τ/Δt < 0.5 → unstable
    x = range(0, stop = 1, step = 0.01),
    dims = 2, # default mode must be added!!
    Δt = :default, # default: Δt = Δx
    walledDimensions = :default, # walls around y axis (all non-walled dimensions are periodic!)
    solidNodes = :default, # default: no solid nodes (other than the walls) 
    solidNodeVelocity = :default, # default: static solids - u = [0,0]
    isFluidCompressible = false,
    forceDensity = :default, # default: F(0) = 0
    forcingScheme = :default # {:guo, :shan}, default: Guo, C. Zheng, B. Shi, Phys. Rev. E 65, 46308 (2002)
)
    # the list of schemes is initialized
    schemes = [] |> Vector{Symbol}
    # the keywords for the initial conditions are initialized
    kwInitialConditions = (; )
    boundaryConditionsParams = (; )

    # if default conditions were chosen, ρ is built. Otherwise its dimensions are verified
    if massDensity == :default
        massDensity = [length(x) for _ in 1:dims] |> v -> ones(v...)
    else
        size(massDensity) |> sizeM -> all(x -> x == sizeM[1], sizeM) ? nothing : error("All dimensions must have the same length! size(ρ) = $(sizeM)")
    end

    # the side length is stored
    N = size(massDensity) |> sizeM -> sizeM[1];

    # if default conditions were chosen, u is built. Otherwise its dimensions are verified
    if fluidVelocity == :default
        fluidVelocity = [[0. for _ in 1:dims] for _ in massDensity];
    else
        size(fluidVelocity) |> sizeU -> prod(i == j for i in sizeU for j in sizeU) ? nothing : error("All dimensions must have the same length! size(u) = $(sizeU)")
    end

    #= ------------------------ choosing the velocity set ----------------------- =#
    # if dimensions are too large, and the user did not define a velocity set, then there's an error
    if (dims >= 4) && !(velocities == :default)
        error("for dimensions higher than 3 a velocity set must be defined using a Vector{LBMvelocity}! modelInit(...; velocities = yourInput)")
    # if the user did not define a velocity set, then a preset is chosen
    elseif velocities == :default
        velocities = [[D1Q3]; [D2Q9]; [D3Q27]] |> v -> v[dims]
    # if the user did define a velocity set, its type is verified
    elseif !(velocities isa Vector{LBMvelocity})
        error("please input a velocity set using a Vector{LBMvelocity}!")
    end

    #= ---------------- space and time variables are initialized ---------------- =#
    Δx = step(x);
    # by default Δt = Δx, as this is the most stable
    (Δt == :default) ? (Δt = Δx) : nothing
    # size Δx/Δt is often used, its value is stored to avoid redundant calculations
    Δx_Δt = Δx/Δt |> Float64
    spaceTime = (; x, Δx, Δt, Δx_Δt, dims); 
    time = [0.];

    #= -------------------- fluid parameters are initialized -------------------- =#
    c_s, c2_s, c4_s = Δx_Δt/√3, Δx_Δt^2 / 3, Δx_Δt^4 / 9;
    relaxationTime = Δt * relaxationTimeRatio;
    fluidParams = (; c_s, c2_s, c4_s, relaxationTime, isFluidCompressible);

    #= -------------------- boundary conditions (bounce back) -------------------- =#
    wallRegion = [false for _ in massDensity]
    if walledDimensions != :default
        wallRegion = wallNodes(massDensity; walledDimensions = walledDimensions); 

        append!(schemes, [:bounceBack])
    end
    if solidNodes != :default && size(solidNodes) == size(wallRegion)
        wallRegion = wallRegion .|| solidNodes

        append!(schemes, [:bounceBack])
    end
    dims <= 2 && (wallRegion = sparse(wallRegion))

    if :bounceBack in schemes
        massDensity[wallRegion] .= 0;
        streamingInvasionRegions, oppositeVectorId = bounceBackPrep(wallRegion, velocities);
        boundaryConditionsParams = merge(boundaryConditionsParams, (; wallRegion, streamingInvasionRegions, oppositeVectorId));
    end

    #= -------------------- boundary conditions (moving walls) ------------------- =#

    if solidNodeVelocity isa Array && solidNodeVelocity[1] isa Vector && size(solidNodeVelocity) == size(massDensity)
        maskedArray = [[0.,0] for _ in solidNodeVelocity]
        maskedArray[wallRegion] = solidNodeVelocity[wallRegion]
        boundaryConditionsParams = merge(boundaryConditionsParams, (; solidNodeVelocity = maskedArray));

        append!(schemes, [:movingWalls])
    end

    #= --------------------------- forcing scheme prep --------------------------- =#
    # the default forcing scheme is Guo
    forcingScheme == :default && (forcingScheme = :guo)

    # by defualt, there is no force density
    if forceDensity == :default
        forceDensity = [[0., 0]];
    # if a single vector is defined it is assumed the force denisty is constant
    elseif forceDensity isa Vector && size(forceDensity) == size(fluidVelocity[1])
        forceDensity = [forceDensity |> Vector{Float64} for _ in massDensity];
        forceDensity[wallRegion] = [[0.,0] for _ in forceDensity[wallRegion]]
        kwInitialConditions = merge(kwInitialConditions, (; forceDensity, Δt))

        append!(schemes, [forcingScheme])
    # if a force density field is defined its dimensions are verified
    elseif size(forceDensity) == size(massDensity)
        forceDensity[wallRegion] = [[0.,0] for _ in forceDensity[wallRegion]]
        kwInitialConditions = merge(kwInitialConditions, (; forceDensity, Δt))

        append!(schemes, [forcingScheme])
    # if none of the above, the dimensions must be wrong
    else
        error("force density does not have consistent dimensions!")
    end

    #= --------------------------- initial distributions are found --------------------------- =#
    initialDistributions = [
        findInitialConditions(
            id,
            velocities,
            fluidParams,
            massDensity,
            fluidVelocity,
            Δx_Δt; 
            kwInitialConditions = kwInitialConditions
        ) 
    for id in eachindex(velocities)]

    #= ------------------------- the model is initialized ------------------------ =#
    model = LBMmodel(
        spaceTime, # space step (Δx), time step (Δt), space coordinate (x), Δt/Δx, dimensionality (dims)
        time, # not in spaceTime bc NamedTuple are immutable!
        fluidParams, # speed of sound and its powers (c_s, c2_s, c4_s), relaxation time (τ)
        (; massDensity = massDensity), # ρ₀
        massDensity, # mass density
        massDensity.*fluidVelocity, # momentum density
        fluidVelocity, # fluid velocity
        forceDensity,
        [initialDistributions], # f_i(x, t) for all t
        velocities, # c_i for all i
        boundaryConditionsParams, # stream invasion regions and index j such that c[i] = -c[j]
        unique(schemes)
    );

#= ---------------------------- consistency check ---------------------------- =#
    # to ensure consistency, ρ, ρu and u are all found using the initial conditions of f_i
    hydroVariablesUpdate!(model);
    # if either ρ or u changed, the user is notified
    acceptableError = 0.01;
    fluidRegion = wallRegion .|> b -> !b;
    error_massDensity = (model.massDensity[fluidRegion] - massDensity[fluidRegion] .|> abs)  |> maximum
    error_fluidVelocity = (model.fluidVelocity[fluidRegion] - fluidVelocity[fluidRegion] .|> norm) |> maximum
    if (error_massDensity > acceptableError) || (error_fluidVelocity > acceptableError)
        @warn "the initial conditions for ρ and u could not be met. New ones were defined."
    end

    # the model is returned
    return model
end
