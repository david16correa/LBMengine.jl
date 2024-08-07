#= ==========================================================================================
=============================================================================================
general methods
=============================================================================================
========================================================================================== =#

function massDensity(model::LBMmodel; time = :default)
    if time == :default
        return sum(distribution for distribution in model.distributions[end])
    else
        return sum(distribution for distribution in model.distributions[time])
    end
end

function momentumDensity(model::LBMmodel; time = :default)
    if time == :default
        return sum(scalarFieldTimesVector(model.distributions[end][id], model.velocities[id].c) for id in eachindex(model.velocities))
    else
        return sum(scalarFieldTimesVector(model.distributions[time][id], model.velocities[id].c) for id in eachindex(model.velocities))
    end
end

function hydroVariablesUpdate!(model::LBMmodel; time = :default)
    model.ρ = massDensity(model; time = time)
    model.ρu = momentumDensity(model; time = time)
    model.u = [[0.; 0] for _ in model.ρ]
    fluidIndices = (model.ρ .≈ 0) .|> b -> !b;
    model.u[fluidIndices] = model.ρu[fluidIndices] ./ model.ρ[fluidIndices]
end

function equilibriumDistribution(id::Int64, model::LBMmodel)
    # the quantities to be used are saved separately
    ci = model.velocities[id].c .* model.spaceTime.Δx_Δt
    wi = model.velocities[id].w
    # the equilibrium distribution is found step by step and returned
    firstStep = vectorFieldDotVector(model.u, ci) |> udotci -> udotci/model.fluidParams.c2_s + udotci.^2 / (2 * model.fluidParams.c4_s)
    secondStep = firstStep - vectorFieldDotVectorField(model.u, model.u)/(2*model.fluidParams.c2_s) .+ 1

    if model.fluidParams.fluidIsCompressible
        return wi * (secondStep .* model.ρ)
    else
        return wi * model.ρ + wi * (model.initialConditions.ρ .* (secondStep .- 1))
    end
end

"calculates Ω at the last recorded time!"
function collisionOperator(id::Int64, model::LBMmodel; returnEquilibriumDistribution = false)
    # the Bhatnagar-Gross-Krook collision opeartor is used, returning the equilibrium distribution if needed
    return -model.distributions[end][id] + equilibriumDistribution(id, model) |> f -> model.spaceTime.Δt/model.fluidParams.τ * f
end

"Time evolution (without pressure diff)"
function LBMpropagate!(model::LBMmodel)
    # collision (or relaxation)
    collisionedDistributions = [model.distributions[end][id] .+ collisionOperator(id, model) for id in eachindex(model.velocities)] 

    # propagated distributions will be saved in a new vector
    propagatedDistributions = [] |> LBMdistributions ;

    # streaming (or propagation), with streaming invasion exchange
    for id in eachindex(model.velocities)
        # distributions are initially streamed, and the wall regions are imposed to vanish
        streamedDistribution = pbcMatrixShift(collisionedDistributions[id], model.velocities[id].c)
        streamedDistribution[model.boundaryConditionsParams.wallRegion] .= 0;

        # the invasion region of the fluid with opposite momentum is retrieved
        conjugateInvasionRegion, conjugateId = model.boundaryConditionsParams |> params -> (params.streamingInvasionRegions[params.oppositeVectorId[id]], params.oppositeVectorId[id])

        # streaming invasion exchange step is performed
        streamedDistribution[conjugateInvasionRegion] = collisionedDistributions[conjugateId][conjugateInvasionRegion]

        # the resulting propagation is appended to the propagated distributions
        append!(propagatedDistributions, [streamedDistribution]);
    end

    # the new distributions and time are appended
    append!(model.distributions, [propagatedDistributions]);
    append!(model.time, [model.time[end]+model.spaceTime.Δt]);

    # Finally, the hydrodynamic variables are updated
    hydroVariablesUpdate!(model);
end

#= ==========================================================================================
=============================================================================================
init methods
=============================================================================================
========================================================================================== =#

"Initializes f_i to f^eq_i, which is the simplest strategy."
function findInitialConditions(id::Int64, velocities::Vector{LBMvelocity}, fluidParams::NamedTuple, ρ::Array{Float64}, u::Array{Vector{Float64}}, Δx_Δt::Float64) 
    # the quantities to be used are saved separately
    ci = velocities[id].c .* Δx_Δt
    wi = velocities[id].w
    # the equilibrium distribution is found step by step and returned
    firstStep = vectorFieldDotVector(u, ci) |> udotci -> udotci/fluidParams.c2_s + udotci.^2 / (2 * fluidParams.c4_s)
    secondStep = firstStep - vectorFieldDotVectorField(u, u)/(2*fluidParams.c2_s) .+ 1
    return secondStep .* (wi * ρ)
end

function modelInit(; 
    ρ = :default, # default: ρ(x) = 1
    u = :default, # default: u(x) = 0
    velocities = :default, # default: chosen by dimensionality (D1Q3, D2Q9, or D3Q27)
    τ_Δt = 0.8, # τ/Δt > 1 → under-relaxation, τ/Δt = 1 → full relaxation, 0.5 < τ/Δt < 1 → over-relaxation, τ/Δt < 0.5 → unstable
    x = range(0, stop = 1, step = 0.01),
    Δt = :default, # default: Δt = Δx
    walledDimensions = [2], # walls around y axis (all non-walled dimensions are periodic!)
    solidNodes = :default, # default: no solid nodes (other than the walls)
    fluidIsCompressible = false
)

    # if default conditions were chosen, ρ is built. Otherwise its dimensions are verified
    if ρ == :default
        ρ = [1. for i in x, j in x]
    else
        size(ρ) |> sizeM -> prod(i == j for i in sizeM for j in sizeM) ? nothing : error("All dimensions must have the same length! size(ρ) = $(sizeM)")
    end

    # if default conditions were chosen, u is built. Otherwise its dimensions are verified
    if u == :default
        u = [[0.; 0] for _ in ρ];
    else
        size(u) |> sizeU -> prod(i == j for i in sizeU for j in sizeU) ? nothing : error("All dimensions must have the same length! size(u) = $(sizeU)")
    end

    # the dimensionality and the side length are stored
    dims, N = size(ρ) |> sizeM -> (length(sizeM), sizeM[1]);

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
    τ = Δt * τ_Δt;
    fluidParams = (; c_s, c2_s, c4_s, τ, fluidIsCompressible);
    wallRegion = wallNodes(ρ; walledDimensions = walledDimensions); 
    if solidNodes != :default && size(solidNodes) == size(wallRegion)
        wallRegion = wallRegion .|| solidNodes
    end
    padded_ρ = copy(ρ); padded_ρ[wallRegion] .= 0;
    initialDistributions = [findInitialConditions(id, velocities, fluidParams, padded_ρ, u, Δx_Δt) for id in eachindex(velocities)]

    #= -------------------- boundary conditions (bounce back) -------------------- =#
    streamingInvasionRegions, oppositeVectorId = bounceBackPrep(wallRegion, velocities);
    boundaryConditionsParams = (; wallRegion, streamingInvasionRegions, oppositeVectorId);

    #= ------------------------- the model is initialized ------------------------ =#
    model = LBMmodel(spaceTime, time, fluidParams, (; ρ = padded_ρ), padded_ρ, padded_ρ.*u, u, [initialDistributions], velocities, boundaryConditionsParams);

    #= ---------------------------- consistency check ---------------------------- =#
    # to ensure consistency, ρ, ρu and u are all found using the initial conditions of f_i
    hydroVariablesUpdate!(model);
    # if either ρ or u changed, the user is notified
    acceptableError = 0.01;
    fluidRegion = wallRegion .|> b -> !b;
    error_ρ = (model.ρ[fluidRegion] - ρ[fluidRegion] .|> abs)  |> maximum
    error_u = (model.u[fluidRegion] - u[fluidRegion] .|> norm) |> maximum
    if (error_ρ > acceptableError) || (error_u > acceptableError)
        @warn "the initial conditions for ρ and u could not be met. New ones were defined."
    end

    # the model is returned
    return model
end
