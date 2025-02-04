module LBMengine
    using SparseArrays
    using CairoMakie
    using DataFrames, CSV, Dates
    import LinearAlgebra.dot, LinearAlgebra.norm

    include("structs.jl");
    include("aux.jl");
    include("fluids.jl");
    include("initMethods.jl");
    include("particles.jl")

    # structs
    export LBMvelocity, LBMparticle, LBMdistributions, LBMmodel

    # aux methods - some quick plots
    export plotFluidVelocity, plotMomentumDensity, plotMassDensity

    # main methods - initialization
    export modelInit, addBead!, addSquirmer!
    # main methods - dynamics
    export hydroVariablesUpdate!, LBMpropagate!

    # extra methods - rheology
    export viscousStressTensor

    # extra methods - saving data
    export writeTensor
end # module LBMengine
