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

# f_i(x) for all i in the model
LBMdistributions = AbstractArray{Float64}

# rigid moving particles
mutable struct LBMparticle
    id::Int64
    particleParams::NamedTuple # mass^-1, momentOfInertia^-1, solidRegionGenerator, symmetries
    boundaryConditionsParams::NamedTuple
    position::Vector{Float64}
    velocity::Vector{Float64}
    angularVelocity::Union{Float64, Vector{Float64}}
    nodeVelocity::AbstractArray{Float64}
    momentumInput::Vector
    angularMomentumInput::Union{Float64, Vector}
end

mutable struct LBMmodel
    spaceTime::NamedTuple # space step (Δx), time step (Δt), space coordinate (x), Δt/Δx, dimensionality (dims)
    tick::Int64
    time::Float64 # not in spaceTime bc NamedTuple are immutable!
    fluidParams::NamedTuple # speed of sound and its powers (c_s, c2_s, c4_s), relaxation time (τ)
    initialConditions::NamedTuple # ρ₀
    massDensity::AbstractArray{Float64} # mass density
    momentumDensity::AbstractArray{Float64} # momentum density
    fluidVelocity::AbstractArray{Float64} # fluid velocity
    forceDensity::AbstractArray{Float64}
    distributions::LBMdistributions # f_i(x, t) for one t
    velocities::NamedTuple # c_i for all i
    boundaryConditionsParams::NamedTuple # stream invasion regions and index j such that c[i] = -c[j]
    particles::Vector{LBMparticle}
    schemes::Vector{Symbol}
    #= schemes implemented thus far:
        :bgk (collision model, stable),
        :trt (collision model, stable),
        :bounceBack (boundary conditions, stable),
        :movingWalls (boundary conditions, stable),
        :guo (forcing, stable),
        :ladd (rigid moving particles, stable)
        :saveData (stable)
    =#
end
