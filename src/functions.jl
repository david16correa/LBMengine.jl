#= ==========================================================================================
=============================================================================================
functions
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
    model.u = model.ρu ./ model.ρ
end

function equilibrium(id::Int64, model::LBMmodel)
    # the id is checked
    checkIdInModel(id, model)
    # the quantities to be used are saved separately
    ci = model.velocities[id].c
    wi = model.velocities[id].w
    # the equilibrium distribution is found step by step and returned
    firstStep = vectorFieldDotVector(model.u, ci) |> udotci -> udotci/model.fluidParamters.c2_s + udotci.^2 / (2 * model.fluidParamters.c4_s)
    secondStep = firstStep - vectorFieldDotVectorField(model.u, model.u)/(2*model.fluidParamters.c2_s) .+ 1
    return secondStep .* (wi * model.ρ)
end

"calculates Ω at the last recorded time!"
function collisionOperator(id::Int64, model::LBMmodel)
    # the id is checked
    checkIdInModel(id, model)
    # the Bhatnagar-Gross-Krook collision opeartor is used
    return -model.distributions[end][id] + equilibrium(id, model) |> f -> model.spaceTime.Δt/model.fluidParamters.τ * f
end

function LBMpropagate!(model::LBMmodel)
    # propagated distributions will be saved in a new vector
    propagatedDistributions = [] |> LBMdistributions ;
    # each propagated distribution is found and saved
    for id in eachindex(model.velocities)
        # collision (or relaxation)
        fnew = model.distributions[end][id] .+ collisionOperator(id, model);
        # streaming (or propagation)
        pbcMatrixShift(fnew, model.velocities[id].c * model.spaceTime.Δt_Δx) |> fshifted -> append!(propagatedDistributions, [fshifted]);
    end
    # the new distributions and time are appended
    append!(model.distributions, [propagatedDistributions]);
    append!(model.time, [model.time[end]+model.spaceTime.Δt]);
    # Finally, the hydrodynamic variables are updated
    hydroVariablesUpdate!(model);
end

"Initializes f_i to f^eq_i, which is the simplest strategy."
function initialConditions(id::Int64, velocities::Vector{LBMvelocity}, fluidParamters::NamedTuple, ρ::Array{Float64}, u::Array{Vector{Float64}}) 
    # the quantities to be used are saved separately
    ci = velocities[id].c
    wi = velocities[id].w
    # the equilibrium distribution is found step by step and returned
    firstStep = vectorFieldDotVector(u, ci) |> udotci -> udotci/fluidParamters.c2_s + udotci.^2 / (2 * fluidParamters.c4_s)
    secondStep = firstStep - vectorFieldDotVectorField(u, u)/(2*fluidParamters.c2_s) .+ 1
    return secondStep .* (wi * ρ)
end


function modelInit(ρ::Array{Float64}, u::Array{Vector{Float64}}; velocities = "auto", Δt = 0.01, τ = 1., sideLength = 1)
    dims, len = ρ |> size |> length, size(ρ, 1);
    # if the velocity set is not defined by the user,
    # it is chosen from the dimensionality of the problem 
    # (no velocity set has been hard coded yet for 4 or more dimensions)
    if velocities isa Vector{LBMvelocity}
        nothing
    elseif dims == 1
        velocities = D1Q3;
    elseif dims == 2
        velocities = D2Q9;
    elseif dims == 3
        velocities = D3Q27;
    else
        error("for dimensions higher than 3 a velocity set must be defined using a Vector{LBMvelocity}! modelInit(...; velocities = yourInput)")
    end
    #= ---------------- space and time variables are initialized ---------------- =#
    # A vector for the coordinates (which are all assumed to be equal) is created, and its step is stored
    x = range(0, stop = sideLength, length = len); Δx = step(x);
    # Δt/Δx is imposed to be an integer; Δt is adjusted if necessary
    Δt_Δx = Δt/Δx |> ceil |> Int64; 
    consistentΔt = Δx*Δt_Δx;
    if consistentΔt != Δt
        @warn "Δt = $(Δt) cannot be used, as Δt/Δt must be an integer; Δt = $(consistentΔt) will be used instead.";
        Δt = consistentΔt;
    end
    spaceTime = (; x, Δx, Δt, Δt_Δx, dims); 
    time = [0.];
    #= -------------------- fluid parameters are initialized -------------------- =#
    c_s = Δx/Δt / √3;
    c2_s = (Δx/Δt)^2 / 3; c4_s = c2_s^2;
    fluidParamters = (; c_s, c2_s, c4_s, τ);
    initialDistributions = [initialConditions(id, velocities, fluidParamters, ρ, u) for id in eachindex(velocities)]
    #= ------------------------ the model is initialized ------------------------ =#
    model = LBMmodel(spaceTime, time, fluidParamters, ρ, ρ.*u, u, [initialDistributions], velocities);
    # to ensure consitensy, ρ, ρu and u are all found using the initial conditions of f_i
    hydroVariablesUpdate!(model);
    # if either ρ or u changed, the user is notified
    acceptableError = 0.01;
    error_ρ = (model.ρ - ρ .|> abs) |> maximum
    error_u = (model.u - u .|> v -> sum(v.*v)) |> maximum
    if (error_ρ > acceptableError) || (error_u > acceptableError)
        @warn "the initial conditions for ρ and u could not be met. New ones were defined."
    end
    # the model is returned
    return model
end


