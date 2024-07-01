#= ==========================================================================================
=============================================================================================
structs
=============================================================================================
========================================================================================== =#

# c and w
struct LBMvelocity
    c::Vector{Int64}
    w::Float64
end

# f_i(x) for all i in the model, independent of time!
LBMdistributions = Vector{Matrix{Float64}}

mutable struct LBMmodel
    spaceTime::NamedTuple # space step (Δx), time step (Δt), space coordinate (x), Δt/Δx, dimensionality (dims)
    time::Vector{Float64} # not in spaceTime bc NamedTuple are immutable!
    fluidParamters::NamedTuple # speec of sound and its powers (c_s, c2_s, c4_s), relaxation time (τ)
    ρ::Matrix{Float64} # mass density
    ρu::Matrix{Vector{Float64}} # momentum density
    u::Matrix{Vector{Float64}} # fluid velocity
    distributions::Vector{LBMdistributions} # f_i(x, t) for all t
    velocities::Vector{LBMvelocity} # c_i for all i
end
