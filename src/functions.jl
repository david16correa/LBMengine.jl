#= ==========================================================================================
=============================================================================================
functions
=============================================================================================
========================================================================================== =#

function equilibrium(id::Int64, model::LBMmodel)
    # the id is checked
    checkIdInModel(id, model)
    # the quantities to be used are found
    ci = model.distributions[id].vel.c
    wi = model.distributions[id].vel.weight
    ρ = sum(distribution.f[end] for distribution in model.distributions)
    u = sum(scalarFieldTimesVector(distribution.f[end], distribution.vel.c) for distribution in model.distributions) |> ρu -> ρu ./ ρ
    # the equilibrium distribution is found and returned
    return ((vectorFieldDotVector(u, ci) |> udotci -> udotci./model.c2_s + udotci.^2 ./ (2 * model.c4_s)) - vectorFieldDotVectorField(u, u)/(2*model.c2_s) .+ 1) .* (wi * ρ)
end

"calculates Ω at the last recorded time!"
function collisionOperator(id::Int64, model::LBMmodel)
    checkIdInModel(id, model)
    return model.distributions[id].f - equilibrium(id, model) |> f -> -model.Δt/model.τ * f
end

#=function model_init()=#
#==#
#=end=#
