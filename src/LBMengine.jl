module LBMengine
    using SparseArrays
    using CairoMakie
    using Dates

    include("structs.jl");
    include("aux.jl");
    include("functions.jl");

    # structs
    export LBMvelocity, LBMdistributions, LBMmodel

    # aux methods - graphics stuff
    export plotFluidVelocity, plotMomentumDensity, plotMassDensity
    export anim8fluidVelocity, anim8momentumDensity, anim8massDensity

    # main methods - initialization
    export modelInit
    # main methods - dynamics
    export hydroVariablesUpdate, tick!, LBMpropagate!
end # module LBMengine
