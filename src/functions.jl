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
    firstStep = vectorFieldDotVector(model.u, ci) |> udotci -> udotci/model.c2_s + udotci.^2 / (2 * model.c4_s)
    secondStep = firstStep - vectorFieldDotVectorField(model.u, model.u)/(2*model.c2_s) .+ 1
    return secondStep .* (wi * model.ρ)
end

"calculates Ω at the last recorded time!"
function collisionOperator(id::Int64, model::LBMmodel)
    checkIdInModel(id, model)
    return -model.distributions[end][id] + equilibrium(id, model) |> f -> model.Δt/model.τ * f
end

function LBMpropagate!(model::LBMmodel)
    # auxilary local function to implement periodic boundary conditions
    pbcShift(X, Δ) = X .+ Δ .+ length(X) .|> x -> (x-1)%length(X) + 1;
    # propagated distributions will be saved in a new vector
    propagatedDistributions = [] |> LBMdistributions ;
    # each propagated distribution is found and saved
    for id in eachindex(model.velocities)
        shiftedCoordinates = [pbcShift(model.spaceCoordinates[i], model.velocities[id].c[i]) for i in eachindex(model.spaceCoordinates)];
        model.distributions[end][id] .+ collisionOperator(id, model) |> fnew -> append!(propagatedDistributions, [fnew[shiftedCoordinates...]]);
    end
    # the new hydrodynamic variables are updated
    hydroVariablesUpdate!(model);
    # Finally, the new distributions and time are appended
    append!(model.distributions, [propagatedDistributions]);
    append!(model.time, [model.time[end]+model.Δt]);
end

function modelInit(velocities::Vector{LBMvelocity}; dim = 2, Δx = 0.01, Δt = 0.01, τ = 1., sideLength = 1)
    n = length(velocities)
    x = range(0, stop = sideLength, step = Δx); len = length(x)
    spaceCoordinates = [x |> eachindex |> collect for _ in 1:dim]
    c_s = Δx/Δt / √3
    c2_s = c_s^2; c4_s = c2_s^2;
    ρ = [1 for _ in Array{Int64}(undef, (len for _ in 1:dim)...)] 
    u = [[0 for _ in 1:dim] for _ in ρ]
    ρu = u
    distributions = [[ρ/n for _ in velocities]]
    time = [0.]
    return LBMmodel(x, spaceCoordinates, Δx, Δt, c_s, c2_s, c4_s, ρ, ρu, u, τ, distributions, velocities, time)
end


