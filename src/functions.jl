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

function LBequation(id::Int64, model::LBMmodel)
    pbcShift(X, Δ) = X .+ Δ .|> x -> (x-1)%length(X) + 1
    hydroVariablesUpdate!(model)
    fnew = model.distributions[end][id] .+ collisionOperator(id, model)
    coordinates = size(fnew) .|> len -> 1:len
    shiftedCoordinates = [pbcShift(coordinates[i], model.velocities[id].c[i]) for i in eachindex(coordinates)]
    append!(model.distributions, [fnew[shiftedCoordinates]])
    append!(model.time, [model.time[end]+model.Δt])
end

#=function model_init()=#
#==#
#=end=#
