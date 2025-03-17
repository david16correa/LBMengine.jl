module LBMengine
    using SparseArrays, CUDA
    using CairoMakie
    using DataFrames, CSV, Dates
    import LinearAlgebra.dot, LinearAlgebra.norm

    include("structs.jl");
    include("aux.jl");

    if CUDA.functional()
        include("fluids_gpu.jl");
        include("aux_gpu.jl");
    else
        include("fluids.jl");
    end

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
