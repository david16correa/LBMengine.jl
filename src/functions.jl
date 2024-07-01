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
    # the quantities to be used are found
    ci = model.velocities[id].c
    wi = model.velocities[id].w
    # the equilibrium distribution is found and returned
    firstStep = vectorFieldDotVector(model.u, ci) |> udotci -> udotci/model.fluidParamters.c2_s + udotci.^2 / (2 * model.fluidParamters.c4_s)
    secondStep = firstStep - vectorFieldDotVectorField(model.u, model.u)/(2*model.fluidParamters.c2_s) .+ 1
    return secondStep .* (wi * model.ρ)
end

"calculates Ω at the last recorded time!"
function collisionOperator(id::Int64, model::LBMmodel)
    checkIdInModel(id, model)
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

function modelInit(velocities::Vector{LBMvelocity}; dims = 2, Δx = 0.01, Δt = 0.01, τ = 1., sideLength = 1)
    # ---------------- space and time variables are initialized ---------------- 
    n = length(velocities);
    x = range(0, stop = sideLength, step = Δx); len = length(x);
    # to ensure Δt/Δx is an integer, Δx is adjusted
    Δt_Δx = Δt/Δx |> ceil |> Int64; 
    consistentΔx = Δt/Δt_Δx;
    if consistentΔx != Δx
        @warn "Δx = $(Δx) cannot be used, as Δt/Δx must be an integer; Δx = $(consistentΔx) will be used instead.";
        Δx = consistentΔx;
    end
    spaceTime = (; x, Δx, Δt, Δt_Δx, dims); 
    time = [0.];
    # -------------------- fluid parameters are initialized -------------------- 
    c_s = Δx/Δt / √3;
    c2_s = (Δx/Δt)^2 / 3; c4_s = c2_s^2;
    fluidParamters = (; c_s, c2_s, c4_s, τ);
    ρ = [1. for _ in Array{Int64}(undef, (len for _ in 1:dims)...)];
    u = [[0. for _ in 1:dims] for _ in ρ];
    ρu = u;
    distributions = [[ρ/n for _ in velocities]];
    # ------------------------ the model is initialized ------------------------ 
    model = LBMmodel(spaceTime, time, fluidParamters, ρ, ρu, u, distributions, velocities);
    # to ensure consitensy, ρ, ρu and u are all found using the initial conditions of f_i
    hydroVariablesUpdate!(model);
    # the model is returned
    return model
end


