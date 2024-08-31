module LBMengine
    using SparseArrays
    using CairoMakie
    using Dates

    include("structs.jl");
    include("aux.jl");
    include("fluids.jl");
    include("initMethods.jl");

    # structs
    export LBMvelocity, LBMparticle, LBMdistributions, LBMmodel

    # aux methods - graphics stuff
    export plotFluidVelocity, plotMomentumDensity, plotMassDensity
    export anim8fluidVelocity, anim8momentumDensity, anim8massDensity

    # main methods - initialization
    export modelInit, addBead!
    # main methods - dynamics
    export hydroVariablesUpdate, tick!, LBMpropagate!

    # non-exported functions - aux methods
    #= export dot, scalarFieldTimesVector, vectorFieldDotVector, vectorFieldDotVectorField =#
    #= export pbcIndexShift, pbcMatrixShift, wallNodes, bounceBackPrep, save_jpg =#
    #= export D1Q3, D2Q9, D3Q27 =#

    # non-exported functions - main methods
    #= export massDensityGet, momentumDensityGet =#
    #= export equilibriumDistribution, collisionOperator, guoForcingTerm =#
    #= export findInitialConditions =#

    # non-exported functions - aux methods
    #= export mean, norm =#
end # module LBMengine
