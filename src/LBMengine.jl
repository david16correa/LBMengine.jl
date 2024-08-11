module LBMengine
    using SparseArrays
    using CairoMakie
    using Dates

    include("structs.jl");
    include("aux.jl");
    include("functions.jl");

    (!isdir("anims")) ? (mkdir("anims")) : nothing; (!isdir("anims/$(today())")) ? (mkdir("anims/$(today())")) : nothing;
    (!isdir("figs")) ? (mkdir("figs")) : nothing; (!isdir("figs/$(today())")) ? (mkdir("figs/$(today())")) : nothing;

    # structs
    export LBMvelocity, LBMdistributions, LBMmodel

    # aux methods - unsure
    export dot, scalarFieldTimesVector, vectorFieldDotVector, vectorFieldDotVectorField
    export pbcIndexShift, pbcMatrixShift, wallNodes, bounceBackPrep, save_jpg
    export D1Q3, D2Q9, D3Q27

    # main methods - unsure
    export massDensityGet, momentumDensityGet, hydroVariablesUpdate
    export equilibriumDistribution, collisionOperator, guoForcingTerm, tick!
    export findInitialConditions

    # aux methods - misc
    export mean, norm
    # aux methods - graphics stuff
    export plotFluidVelocity, plotMomentumDensity, plotMassDensity
    export anim8fluidVelocity, anim8momentumDensity, anim8massDensity

    # main methods - dynamics
    export LBMpropagate!
    # main methods - initialization
    export modelInit

    greet() = print("Hello World 42!")
end # module LBMengine
