#= ==========================================================================================
=============================================================================================
general methods
=============================================================================================
========================================================================================== =#

function massDensity(model::LBMmodel; time = -1)
    if time == -1
        return sum(distribution for distribution in model.distributions[end])
    else
        return sum(distribution for distribution in model.distributions[time])
    end
end

function momentumDensity(model::LBMmodel; time = -1)
    if time == -1
        return sum(scalarFieldTimesVector(model.distributions[end][id], model.velocities[id].c) for id in eachindex(model.velocities))
    else
        return sum(scalarFieldTimesVector(model.distributions[time][id], model.velocities[id].c) for id in eachindex(model.velocities))
    end
end

function hydroVariablesUpdate!(model::LBMmodel)
    model.ρ = massDensity(model)
    model.ρu = momentumDensity(model)
    model.u = [[0., 0] for _ in model.ρ]
    fluidIndices = (model.ρ .≈ 0) .|> b -> !b;
    model.u[fluidIndices] = model.ρu[fluidIndices] ./ model.ρ[fluidIndices]
end

function equilibrium(id::Int64, model::LBMmodel)
    # the id is checked
    checkIdInModel(id, model)
    # the quantities to be used are saved separately
    ci = model.velocities[id].c
    wi = model.velocities[id].w
    # the equilibrium distribution is found step by step and returned
    firstStep = vectorFieldDotVector(model.u, ci) |> udotci -> udotci/model.fluidParams.c2_s + udotci.^2 / (2 * model.fluidParams.c4_s)
    secondStep = firstStep - vectorFieldDotVectorField(model.u, model.u)/(2*model.fluidParams.c2_s) .+ 1
    return secondStep .* (wi * model.ρ)
end

"calculates Ω at the last recorded time!"
function collisionOperator(id::Int64, model::LBMmodel)
    # the id is checked
    checkIdInModel(id, model)
    # the Bhatnagar-Gross-Krook collision opeartor is used
    return -model.distributions[end][id] + equilibrium(id, model) |> f -> model.spaceTime.Δt/model.fluidParams.τ * f
end

function LBMpropagate!(model::LBMmodel)
    # collision (or relaxation)
    collisionedDistributions = [model.distributions[end][id] .+ collisionOperator(id, model) for id in eachindex(model.velocities)] 

    # propagated distributions will be saved in a new vector
    propagatedDistributions = [] |> LBMdistributions ;

    # streaming (or propagation), with streaming invasion exchange
    for id in eachindex(model.velocities)
        # distributions are initially streamed, and the wall regions are imposed to vanish
        streamedDistribution = pbcMatrixShift(collisionedDistributions[id], model.velocities[id].c * model.spaceTime.Δt_Δx)
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
function initialConditions(id::Int64, velocities::Vector{LBMvelocity}, fluidParams::NamedTuple, ρ::Array{Float64}, u::Array{Vector{Float64}}) 
    # the quantities to be used are saved separately
    ci = velocities[id].c
    wi = velocities[id].w
    # the equilibrium distribution is found step by step and returned
    firstStep = vectorFieldDotVector(u, ci) |> udotci -> udotci/fluidParams.c2_s + udotci.^2 / (2 * fluidParams.c4_s)
    secondStep = firstStep - vectorFieldDotVectorField(u, u)/(2*fluidParams.c2_s) .+ 1
    return secondStep .* (wi * ρ)
end

function modelInit(ρ::Array{Float64}, u::Array{Vector{Float64}}; velocities = "auto", τ_Δt = 0.8, sideLength = 1, walledDimensions = [-1])
    sizeM = size(ρ)
    prod(i == j for i in sizeM for j in sizeM) ? nothing : error("All dimensions must have the same length! size(ρ) = $(sizeM)")

    dims, len = length(sizeM), sizeM[1];

    # if dimensions are too large, and the user did not define a velocity set, then there's an error
    if (dims >= 4) && !(velocities isa Vector{LBMvelocity})
        error("for dimensions higher than 3 a velocity set must be defined using a Vector{LBMvelocity}! modelInit(...; velocities = yourInput)")
    # if the user did not define a velocity set, then a preset is chosen
    elseif !(velocities isa Vector{LBMvelocity})
        velocities = [[D1Q3]; [D2Q9]; [D3Q27]] |> v -> v[dims]
    end

    #= ---------------- space and time variables are initialized ---------------- =#
    # A vector for the coordinates (which are all assumed to be equal) is created, and its step is stored
    x = range(0, stop = sideLength, length = len); Δx = Δt = step(x);
    Δt_Δx = 1; # Δt/Δx
    spaceTime = (; x, Δx, Δt, Δt_Δx, dims); 
    time = [0.];

    #= -------------------- fluid parameters are initialized -------------------- =#
    c_s, c2_s, c4_s = 1/(Δt_Δx * √3), 1/(Δt_Δx^2 * 3), 1/(Δt_Δx^4 * 9);
    τ = Δt * τ_Δt;
    fluidParams = (; c_s, c2_s, c4_s, τ);
    wallRegion = wallNodes(ρ, Δt_Δx; walledDimensions = walledDimensions); 
    padded_ρ = copy(ρ); padded_ρ[wallRegion] .= 0;
    initialDistributions = [initialConditions(id, velocities, fluidParams, padded_ρ, u) for id in eachindex(velocities)]

    #= -------------------- boundary conditions (bounce back) -------------------- =#
    streamingInvasionRegions, oppositeVectorId = bounceBackPrep(wallRegion, velocities);
    boundaryConditionsParams = (; wallRegion, streamingInvasionRegions, oppositeVectorId);

    #= ------------------------ the model is initialized ------------------------ =#
    model = LBMmodel(spaceTime, time, fluidParams, padded_ρ, padded_ρ.*u, u, [initialDistributions], velocities, boundaryConditionsParams);
    # to ensure consitensy, ρ, ρu and u are all found using the initial conditions of f_i
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


