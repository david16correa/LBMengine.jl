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

    # aux methods - graphics stuff
    export plotFluidVelocity, plotMomentumDensity, plotMassDensity
    export anim8fluidVelocity, anim8momentumDensity, anim8massDensity

    # main methods - initialization
    export modelInit, addBead!
    # main methods - dynamics
    export hydroVariablesUpdate!, LBMpropagate!

    # non-exported functions
    #= export scalarFieldTimesVector =#
    #= export vectorFieldDotVector =#
    #= export vectorFieldDotVectorField =#
    #= export cross =#
    #= export vectorCrossVectorField =#
    #= export vectorFieldCrossVector =#
    #= export vectorFieldCrossVectorField =#
    #= export pbcIndexShift =#
    #= export pbcMatrixShift =#
    #= export wallNodes =#
    #= export bounceBackPrep =#
    #= export createFigDirs =#
    #= export createAnimDirs =#
    export save_jpg
    #= export massDensityGet =#
    #= export momentumDensityGet =#
    #= export equilibriumDistribution =#
    #= export collisionOperator =#
    #= export guoForcingTerm =#
    #= export tick! =#
    #= export LBMpropagate! =#
    #= export findInitialConditions =#
    #= export eulerStep! =#
    #= export moveParticles! =#
    #= export writeTrajectories =#
end # module LBMengine
