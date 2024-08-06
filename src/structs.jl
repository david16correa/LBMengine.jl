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
LBMdistributions = Vector{Array{Float64}}

mutable struct LBMmodel
    spaceTime::NamedTuple # space step (Δx), time step (Δt), space coordinate (x), Δt/Δx, dimensionality (dims)
    time::Vector{Float64} # not in spaceTime bc NamedTuple are immutable!
    fluidParams::NamedTuple # speed of sound and its powers (c_s, c2_s, c4_s), relaxation time (τ)
    initialConditions::NamedTuple # ρ₀
    ρ::Array{Float64} # mass density
    ρu::Array{Vector{Float64}} # momentum density
    u::Array{Vector{Float64}} # fluid velocity
    distributions::Vector{LBMdistributions} # f_i(x, t) for all t
    velocities::Vector{LBMvelocity} # c_i for all i
    boundaryConditionsParams::NamedTuple # stream invasion regions and index j such that c[i] = -c[j]
end
