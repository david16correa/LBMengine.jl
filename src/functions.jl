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

function hydroVariablesUpdate!(model::LBMmodel; time = -1)
    model.ρ = massDensity(model; time = time)
    model.ρu = momentumDensity(model; time = time)
    model.u = [[0.; 0] for _ in model.ρ]
    fluidIndices = (model.ρ .≈ 0) .|> b -> !b;
    model.u[fluidIndices] = model.ρu[fluidIndices] ./ model.ρ[fluidIndices]
end

function equilibrium(id::Int64, model::LBMmodel; fluidIsCompressible = true)
    # the id is checked
    checkIdInModel(id, model)
    # the quantities to be used are saved separately
    ci = model.velocities[id].c .* model.spaceTime.Δx_Δt
    wi = model.velocities[id].w
    # the equilibrium distribution is found step by step and returned
    firstStep = vectorFieldDotVector(model.u, ci) |> udotci -> udotci/model.fluidParams.c2_s + udotci.^2 / (2 * model.fluidParams.c4_s)
    secondStep = firstStep - vectorFieldDotVectorField(model.u, model.u)/(2*model.fluidParams.c2_s) .+ 1

    if fluidIsCompressible
        return wi * (secondStep .* model.ρ)
    else
        return wi * model.ρ + wi * (model.initialConditions.ρ .* (secondStep .- 1))
    end
end

function equilibrium(id::Int64, model::LBMmodel, ρ::Array{Float64}, u::Array{Vector{Float64}}; fluidIsCompressible = true)
    # the id is checked
    checkIdInModel(id, model)
    # the quantities to be used are saved separately
    ci = model.velocities[id].c .* model.spaceTime.Δx_Δt
    wi = model.velocities[id].w
    # the equilibrium distribution is found step by step and returned
    firstStep = vectorFieldDotVector(u, ci) |> udotci -> udotci/model.fluidParams.c2_s + udotci.^2 / (2 * model.fluidParams.c4_s)
    secondStep = firstStep - vectorFieldDotVectorField(u, u)/(2*model.fluidParams.c2_s) .+ 1

    if fluidIsCompressible
        return wi * (secondStep .* model.ρ)
    else
        return wi * model.ρ + wi * (model.initialConditions.ρ .* (secondStep .- 1))
    end
end

"calculates Ω at the last recorded time!"
function collisionOperator(id::Int64, model::LBMmodel; returnEquilibriumDistribution = false)
    # the id is checked
    checkIdInModel(id, model)
    # the Bhatnagar-Gross-Krook collision opeartor is used, returning the equilibrium distribution if needed
    if returnEquilibriumDistribution
        equilibriumDistribution = equilibrium(id, model)
        return equilibriumDistribution, -model.distributions[end][id] + equilibriumDistribution |> f -> model.spaceTime.Δt/model.fluidParams.τ * f
    else
        return -model.distributions[end][id] + equilibrium(id, model) |> f -> model.spaceTime.Δt/model.fluidParams.τ * f
    end
end

function LBMpropagate!(model::LBMmodel)
    #------------------------------------------collision step------------------------------------------
    # auxilary nodes are created to deal with pressure differences
    auxNodesCreate!(model);
    # the equilibrium distributions and the collisioned distributions are found; their values at the auxilary nodes are not correct at this point
    equilibriumDistributions, collisionedDistributions = 
        [collisionOperator(id, model; returnEquilibriumDistribution = true) for id in eachindex(model.velocities)] |> 
        feqΩ -> ([T[1] for T in feqΩ], model.distributions[end] .+ [T[2] for T in feqΩ]);

    # non equilibrium distributions are found; these are used to find the collisioned distributions at the auxilary nodes
    nonEquilibriumDistributions = collisionedDistributions .- equilibriumDistributions;
    for id in eachindex(nonEquilibriumDistributions)
        nonEquilibriumDistributions[id][auxNodesId(0, model)...] = nonEquilibriumDistributions[id][auxNodesId(model.boundaryConditionsParams.N, model)...];
        nonEquilibriumDistributions[id][auxNodesId(model.boundaryConditionsParams.N+1, model)...] = nonEquilibriumDistributions[id][auxNodesId(1, model)...];
    end
    # now the collisioned distributions everywhere are known; the rest is standard procedure
    collisionedDistributions = equilibriumDistributions .+ nonEquilibriumDistributions;

    #---------------------------------------------stream step--------------------------------------------
    # propagated distributions will be saved in a new vector
    streamedDistributions = [] |> LBMdistributions ;

    for id in eachindex(model.velocities)
        streamedDistribution = pbcMatrixShift(collisionedDistributions[id], model.velocities[id].c) |> distribution -> auxNodesRemove(distribution, model)
        streamedDistribution[model.boundaryConditionsParams.wallRegion] .= 0;
        append!(streamedDistributions, [streamedDistribution])

        collisionedDistributions[id] = auxNodesRemove(collisionedDistributions[id], model)
    end

    #----------------------------------streaming invasion exchange step---------------------------------
    # propagated distributions will be saved in a new vector
    propagatedDistributions = [] |> LBMdistributions ;

    for id in eachindex(model.velocities)
        # the invasion region of the fluid with opposite momentum is retrieved
        conjugateInvasionRegion, conjugateId = model.boundaryConditionsParams |> params -> (params.streamingInvasionRegions[params.oppositeVectorId[id]], params.oppositeVectorId[id])

        # streaming invasion exchange step is performed
        streamedDistributions[id][conjugateInvasionRegion] = collisionedDistributions[conjugateId][conjugateInvasionRegion]

        # the resulting propagation is appended to the propagated distributions
        append!(propagatedDistributions, [streamedDistributions[id]]);
    end

    # auxilary nodes are removed
    auxNodesRemove!(model);

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

function modelInit(ρ::Array{Float64}, u::Array{Vector{Float64}}; 
    velocities = "auto",
    τ_Δt = 0.8,
    Δt = -1,
    xlb = 0,
    xub = 1,
    walledDimensions = [2], # Poiseuille
    pressurizedDimensions = [1], # Poiseuille
    densityHL = (1.01, 0.99), # Poiseuille
    solidNodes = [-1]
)

    sizeM = size(ρ)
    prod(i == j for i in sizeM for j in sizeM) ? nothing : error("All dimensions must have the same length! size(ρ) = $(sizeM)")

    dims, N = length(sizeM), sizeM[1];

    # if dimensions are too large, and the user did not define a velocity set, then there's an error
    if (dims >= 4) && !(velocities isa Vector{LBMvelocity})
        error("for dimensions higher than 3 a velocity set must be defined using a Vector{LBMvelocity}! modelInit(...; velocities = yourInput)")
    # if the user did not define a velocity set, then a preset is chosen
    elseif !(velocities isa Vector{LBMvelocity})
        velocities = [[D1Q3]; [D2Q9]; [D3Q27]] |> v -> v[dims]
    end

    #= ---------------- space and time variables are initialized ---------------- =#
    # A vector for the coordinates (which are all assumed to be equal) is created, and its step is stored
    x = range(xlb, stop = xub, length = N); Δx  = step(x);
    (Δt == -1) ? (Δt = Δx) : nothing
    Δx_Δt = Δx/Δt |> Float64
    spaceTime = (; x, Δx, Δt, Δx_Δt, dims); 
    time = [0.];

    #= -------------------- fluid parameters are initialized -------------------- =#
    c_s, c2_s, c4_s = Δx_Δt/√3, Δx_Δt^2 / 3, Δx_Δt^4 / 9;
    τ = Δt * τ_Δt;
    fluidParams = (; c_s, c2_s, c4_s, τ);
    wallRegion = wallNodes(ρ, 1; walledDimensions = walledDimensions); 
    if size(solidNodes) == size(wallRegion)
        #=wallRegion .+= solidNodes=#
        wallRegion = wallRegion .|| solidNodes
    end
    padded_ρ = copy(ρ); padded_ρ[wallRegion] .= 0;
    initialDistributions = [findInitialConditions(id, velocities, fluidParams, padded_ρ, u,Δx_Δt) for id in eachindex(velocities)]

    #= -------------------- boundary conditions (bounce back) -------------------- =#
    streamingInvasionRegions, oppositeVectorId = bounceBackPrep(wallRegion, velocities);
    bounceBackParams = (; wallRegion, streamingInvasionRegions, oppositeVectorId);

    #= ---------------- boundary conditions (pressure difference) ---------------- =#
    auxSystemSize, auxSystemMainRegion, auxSystemIds = auxNodesPrep(sizeM, pressurizedDimensions, N)
    ρH, ρL = [densityHL[1] for _ in zeros(sizeM[1:end-1]...)], [densityHL[2] for _ in zeros(sizeM[1:end-1]...)]
    pressureDiffParams = (; auxSystemSize, auxSystemMainRegion, auxSystemIds, pressurizedDimensions, ρH, ρL, N);

    #= ------------------------ the model is initialized ------------------------ =#
    model = LBMmodel(spaceTime, time, fluidParams, (; ρ = padded_ρ), padded_ρ, padded_ρ.*u, u, [initialDistributions], velocities, merge(bounceBackParams, pressureDiffParams));
    model.initialConditions = (; ρ = model.initialConditions.ρ |> M -> auxNodesMassDensityCreate(M, model))
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


