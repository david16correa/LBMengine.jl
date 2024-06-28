#= ==========================================================================================
=============================================================================================
preamble
=============================================================================================
========================================================================================== =#

# Environment
using Pkg;
Pkg.activate("environment");

#= ==========================================================================================
=============================================================================================
structs
=============================================================================================
========================================================================================== =#

struct LBMvelocity
    c::Vector{Int64}
    weight::Float64
end

mutable struct LBMmodel
    Δx::Float64 # space step
    Δt::Float64 # time step (Δt = N Δx for some natural N)
    c_s::Float64 # speed of sound, usually cₛ² = (1/3) Δx²/Δt²
    c2_s::Float64 # c_s squared
    c4_s::Float64 # c_s to the fourth power
    τ::Float64 # relaxation time
    distributions::Vector{LBMdistribution}
    velocities::Vector{LBMvelocity}
end

mutable struct LBMdistribution
    f::Vector{Matrix{Float64}}
    vel::LBMvelocity
end

#= ==========================================================================================
=============================================================================================
auxilary functions
=============================================================================================
========================================================================================== =#

function checkIdInModel(id::Int64, model::LBMmodel)
    !(0 < id < length(model.distributions)) ? (error("No distribution with id $(id) was found!")) : nothing
end

function scalarFieldTimesVector(a::Matrix, V::Vector)
    return [a * V for a in a]
end
function scalarFieldTimesVector(V::Vector, a::Matrix)
    return [a * V for a in a]
end

function vectorFieldDotVector(F::Matrix, v::Vector)
    dot(v, w) = v .* w |> sum
    return [dot(F, v) for F in F]
end
function vectorFieldDotVector(v::Vector, F::Matrix)
    dot(v, w) = v .* w |> sum
    return [dot(F, v) for F in F]
end

function vectorFieldDotVectorField(V::Matrix, W::Matrix)
    dot(v, w) = v .* w |> sum
    return [dot(V[i,j], W[i,j]) for i in eachrow(V), j in eachcolumn(V)]
end

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
