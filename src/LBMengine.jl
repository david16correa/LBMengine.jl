module LBMengine
    using SparseArrays, CUDA # performance dependencies
    using DataFrames, CSV, Dates # data dependencies
    import LinearAlgebra.dot, LinearAlgebra.norm # numerical analysis dependencies
    using CairoMakie # graphics dependencies

    # Dynamically load GPU or CPU modules based on CUDA availability (launch julia with `BYPASS_GPU=true julia` to not use the gpu)
    #= bypassGpu = (get(ENV, "BYPASS_GPU", "false") == "true") # this works only during precompilation; a different approach is necessary =#
    bypassGpu = false
    bypassGpu && (@info "GPU bypass enabled; running on CPU")
    useGpu = (CUDA.functional() && !(bypassGpu))
    hardwareModule = useGpu ? "gpu" : "cpu"

    include("$hardwareModule/structs.jl");
    include("$hardwareModule/aux.jl");
    include("$hardwareModule/fluids.jl");
    include("$hardwareModule/misc.jl");
    include("$hardwareModule/initMethods.jl");
    include("$hardwareModule/particles.jl")

    if useGpu
        include("$hardwareModule/kernels.jl")
    end

    # structs
    export LBMvelocity, LBMparticle, LBMdistributions, LBMmodel

    # aux methods - some quick plots
    export plotFluidVelocity, plotMomentumDensity, plotMassDensity

    # main methods - initialization
    export modelInit, addBead!, addSquirmer!, addLinearBond!, addPolarBond!, addDipoles!
    # main methods - dynamics
    export hydroVariablesUpdate!, LBMpropagate!

    # extra methods - rheology
    export viscousStressTensor

    # extra methods - saving data
    export writeTensor
end
