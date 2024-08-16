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

    # non-exported functions - aux methods
    #= export dot, scalarFieldTimesVector, vectorFieldDotVector, vectorFieldDotVectorField =#
    #= export pbcIndexShift, pbcMatrixShift, wallNodes, bounceBackPrep, save_jpg =#
    #= export D1Q3, D2Q9, D3Q27 =#

    # unexported functions - main methods
    #= export massDensityGet, momentumDensityGet =#
    #= export equilibriumDistribution, collisionOperator, guoForcingTerm =#
    #= export findInitialConditions =#

    # unexported functions - aux methods
    #= export mean, norm =#
end # module LBMengine
