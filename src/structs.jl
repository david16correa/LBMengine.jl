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
    x::Vector{Float64}
    spaceCoordinates::Vector{Vector{Int64}}
    Δx::Float64 # space step
    Δt::Float64 # time step (Δt = N Δx for some natural N)
    c_s::Float64 # speed of sound, usually cₛ² = (1/3) Δx²/Δt²
    c2_s::Float64 # c_s^2
    c4_s::Float64 # c_s^4
    ρ::Matrix{Float64} # mass density
    ρu::Matrix{Vector{Float64}} # momentum density
    u::Matrix{Vector{Float64}} # fluid velocity
    τ::Float64 # relaxation time
    distributions::Vector{LBMdistributions} # f_i(x, t) for all t
    velocities::Vector{LBMvelocity} # c_i for all i
    time::Vector{Float64}
end
