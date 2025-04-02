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
    id::UInt8
    particleParams::NamedTuple # mass^-1, momentOfInertia^-1, solidRegionGenerator, symmetries
    boundaryConditionsParams::NamedTuple
    position::AbstractArray{Float64}
    velocity::AbstractArray{Float64}
    angularVelocity::Union{Float64, AbstractArray{Float64}}
    nodeVelocity::AbstractArray{Float64}
    forceInput::AbstractArray{Float64}
    torqueInput::Union{Float64, AbstractArray{Float64}}
end

mutable struct LBMmodel
    spaceTime::NamedTuple # space step (Δx), time step (Δt), space coordinate (x), Δt/Δx, dimensionality (dims)
    tick::Int64
    time::Float64 # not in spaceTime bc NamedTuple are immutable!
    fluidParams::NamedTuple # speed of sound and its powers (c_s, c2_s, c4_s), relaxation time (τ)
    initialConditions::NamedTuple # ρ₀
    massDensity::AbstractArray{Float64} # mass density
    fluidVelocity::AbstractArray{Float64} # fluid velocity
    forceDensity::AbstractArray{Float64}
    distributions::LBMdistributions # f_i(x, t) for one t
    velocities::NamedTuple # c_i for all i
    boundaryConditionsParams::NamedTuple # stream invasion regions and index j such that c[i] = -c[j]
    particles::Vector{LBMparticle}
    particleBonds::AbstractArray
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
