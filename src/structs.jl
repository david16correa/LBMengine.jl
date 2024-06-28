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

