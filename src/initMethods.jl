#= ==========================================================================================
=============================================================================================
init methods
=============================================================================================
========================================================================================== =#

function addBead!(model::LBMmodel;
    massDensity = 0.1,
    radius = 0.1,
    position = :default, # default: origin (actual value is dimensionality dependent)
    velocity = :default, # default: static (actual value is dimensionality dependent)
    angularVelocity = :default, # default: static, (actual value is dimensionality dependent)
    coupleTorques = false,
    coupleForces = true,
    scheme = :default # default: psm (partially saturated method)
)
    # a local function for the general geometry of a centered bead (sphere) is defined
    beadGeometry(x::Vector; radius2 = 0.0625) = sum(x.^2) < radius2

    # the mass is found using the mass density
    mass = massDensity * sum(beadGeometry.(model.spaceTime.X; radius2 = radius^2)) * model.spaceTime.Δx^model.spaceTime.dims

    # position and velocity are defined if necessary
    position == :default && (position = [0. for _ in 1:model.spaceTime.dims])
    velocity == :default && (velocity = [0. for _ in 1:model.spaceTime.dims])
    # the dimensions are checked
    ((length(position) != length(velocity)) || (length(position) != model.spaceTime.dims)) && error("The position and velocity dimensions must match the dimensionality of the model! dims = $(model.spaceTime.dims)")

    # the moment of inertia, initial angular velocity, and angular momentum input are all initialized
    if model.spaceTime.dims == 2
        momentOfInertia = 0.5 * mass * radius^2 # moment of inertia for a disk
        angularVelocity == :default && (angularVelocity = 0.)
        angularMomentumInput = 0.
    elseif model.spaceTime.dims == 3
        momentOfInertia = 0.4 * mass * radius^2 # moment of inertia for a sphere
        angularVelocity == :default && (angularVelocity = [0., 0, 0])
        angularMomentumInput = [0., 0, 0]
    else
        error("For particle simulation dimensionality must be either 2 or 3! dims = $(model.spaceTime.dims)")
    end

    # a new bead is defined and added to the model
    newBead = LBMparticle(
        length(model.particles) + 1,
        (;inverseMass = 1/mass, inverseMomentOfInertia = 1/momentOfInertia, solidRegionGenerator = x -> beadGeometry(x; radius2 = radius^2), symmetries = [:spherical], coupleTorques, coupleForces),
        (; solidRegion = [], streamingInvasionRegions = []),
        position,
        velocity,
        angularVelocity,
        [],
        [0. for _ in 1:model.spaceTime.dims],
        angularMomentumInput,
    )
    append!(model.particles, [newBead]);

    # the schemes of the model are managed
    scheme == :default && (scheme = :psm)

    @assert (newBead.id == 1 || scheme in model.schemes) "$(scheme) cannot be used, as another scheme for particle-fluid collision is being used"

    append!(model.schemes, [scheme])
    model.schemes = model.schemes |> unique
    if !(:bounceBack in model.schemes)
        wallRegion = [false for _ in model.massDensity] |> sparse
        streamingInvasionRegions, oppositeVectorId = bounceBackPrep(wallRegion, model.velocities);
        model.boundaryConditionsParams = merge(model.boundaryConditionsParams, (; wallRegion, streamingInvasionRegions, oppositeVectorId));
        scheme == :ladd && (append!(model.schemes, [:bounceBack]))
    end

    # saving data
    :saveData in model.schemes && writeParticleTrajectory(model.particles[end], model)

    moveParticles!(length(model.particles), model; initialSetup = true)
end

"Initializes f_i to f^eq_i, which is the simplest strategy."
function findInitialConditions(id::Int64, velocities::Vector{LBMvelocity}, fluidParams::NamedTuple, massDensity::Array{Float64}, u::Array{Vector{Float64}}, Δx_Δt::Float64; kwInitialConditions = (; )) 
    # the quantities to be used are saved separately
    ci = velocities[id].c .* Δx_Δt
    wi = velocities[id].w
    if :forceDensity in (kwInitialConditions |> keys)
        consistencyTerm = [[0.; 0] for _ in massDensity]
        fluidIndices = (massDensity .≈ 0) .|> b -> !b;
        consistencyTerm[fluidIndices] = kwInitialConditions.forceDensity[fluidIndices] ./ massDensity[fluidIndices]
        u -= 0.5 * kwInitialConditions.Δt * consistencyTerm
    end
    # the equilibrium distribution is found step by step and returned
    firstStep = vectorFieldDotVector(u, ci) |> udotci -> udotci/fluidParams.c2_s + udotci.^2 / (2 * fluidParams.c4_s)
    secondStep = firstStep - vectorFieldDotVectorField(u, u)/(2*fluidParams.c2_s) .+ 1
    return secondStep .* (wi * massDensity)
end

function modelInit(;
    massDensity = :default, # default: ρ(x) = 1
    fluidVelocity = :default, # default: u(x) = 0
    velocities = :default, # default: chosen by dimensionality (D1Q3, D2Q9, or D3Q27)
    relaxationTimeRatio = :default, # τ/Δt > 1 → under-relaxation, τ/Δt = 1 → full relaxation, 0.5 < τ/Δt < 1 → over-relaxation, τ/Δt < 0.5 → unstable, default: 0.8
    viscosity = :default, # kinematic shear viscosity; its units are length²/time. Default: it's value is taken to be such that the relaxationTime is 0.8Δt
    collisionModel = :default, # {:bgk, :trt}, default: :bgk
    kineticParameter = 0.25, # Λ, for :trt; different choices lead to different stability behaviors! Cf. Krüger p.429
    x = range(0, stop = 1, step = 0.01),
    dims = 2, # default mode must be added!!
    Δt = :default, # default: Δt = Δx
    walledDimensions = :default, # walls around y axis (all non-walled dimensions are periodic!)
    solidNodes = :default, # default: no solid nodes (other than the walls) 
    solidNodeVelocity = :default, # default: static solids - u = [0,0]
    isFluidCompressible = false,
    forceDensity = :default, # default: F(0) = 0
    forcingScheme = :default, # {:guo, :shan}, default: Guo, C. Zheng, B. Shi, Phys. Rev. E 65, 46308 (2002)
    saveData = false, # by default, no data is saved
)
    # the list of schemes is initialized
    schemes = [] |> Vector{Symbol}
    # the keywords for the initial conditions are initialized
    kwInitialConditions = (; )
    boundaryConditionsParams = (; )

    # if default conditions were chosen, ρ is built. Otherwise its dimensions are verified
    if massDensity == :default
        massDensity = [length(x) for _ in 1:dims] |> v -> ones(v...)
    elseif massDensity isa Number
        massDensity = [length(x) for _ in 1:dims] |> v -> massDensity * ones(v...)
    else
        size(massDensity) |> sizeM -> all(x -> x == sizeM[1], sizeM) ? nothing : error("All dimensions must have the same length! size(ρ) = $(sizeM)")
    end

    # the side length is stored
    N = size(massDensity) |> sizeM -> sizeM[1];

    # if default conditions were chosen, u is built. Otherwise its dimensions are verified
    if fluidVelocity == :default
        fluidVelocity = [[0. for _ in 1:dims] for _ in massDensity];
    else
        size(fluidVelocity) |> sizeU -> prod(i == j for i in sizeU for j in sizeU) ? nothing : error("All dimensions must have the same length! size(u) = $(sizeU)")
    end

    #= ------------------------ choosing the velocity set ----------------------- =#
    # if dimensions are too large, and the user did not define a velocity set, then there's an error
    if (dims >= 4) && !(velocities == :default)
        error("for dimensions higher than 3 a velocity set must be defined using a Vector{LBMvelocity}! modelInit(...; velocities = yourInput)")
    # if the user did not define a velocity set, then a preset is chosen
    elseif velocities == :default
        velocities = [[D1Q3]; [D2Q9]; [D3Q27]] |> v -> v[dims]
    # if the user did define a velocity set, its type is verified
    elseif !(velocities isa Vector{LBMvelocity})
        error("please input a velocity set using a Vector{LBMvelocity}!")
    end

    #= ---------------- space and time variables are initialized ---------------- =#
    Δx = step(x);
    # by default Δt = Δx, as this is the most stable
    (Δt == :default) ? (Δt = Δx) : nothing
    # size Δx/Δt is often used, its value is stored to avoid redundant calculations
    Δx_Δt = Δx/Δt |> Float64
    X = [[x[id] for id in Id |> Tuple]  for Id in eachindex(IndexCartesian(), massDensity)]
    spaceTime = (; x, X, Δx, Δt, Δx_Δt, dims); 
    tick, time = 0, 0.;

    #= -------------------- fluid parameters are initialized -------------------- =#
    c_s, c2_s, c4_s = Δx_Δt/√3, Δx_Δt^2 / 3, Δx_Δt^4 / 9;
    collisionModel == :default && (collisionModel = :trt)
    @assert !((viscosity != :default) && (relaxationTimeRatio != :default)) "The viscosity and the relaxationTimeRatio should not be defined simultaneously!"

    relaxationTimeRatio == :default && (relaxationTimeRatio = 0.8)

    if viscosity == :default
        relaxationTime = relaxationTimeRatio * Δt
        viscosity = c2_s * (relaxationTime - Δt/2)
    else
        relaxationTime = viscosity/c2_s + Δt/2
    end

    @assert (collisionModel == :bgk || collisionModel == :trt) "Collision model $collisionModel is not implemented!"

    if collisionModel == :bgk
        fluidParams = (; c_s, c2_s, c4_s, relaxationTime, viscosity, isFluidCompressible);
    elseif collisionModel == :trt
        relaxationTimePlus = relaxationTime
        kineticParameter == :debug && (kineticParameter = viscosity^2/(c4_s * Δt^2)) # here, :trt = :bgk
        relaxationTimeMinus = kineticParameter * c2_s * Δt^2 / viscosity + Δt/2
        fluidParams = (; c_s, c2_s, c4_s, relaxationTime, viscosity, relaxationTimePlus, relaxationTimeMinus, isFluidCompressible);
    end
    append!(schemes, [collisionModel])

    #= -------------------- boundary conditions (bounce back) -------------------- =#
    wallRegion = [false for _ in massDensity]
    if walledDimensions != :default && length(walledDimensions) != 0
        wallRegion = wallNodes(massDensity; walledDimensions = walledDimensions); 

        append!(schemes, [:bounceBack])
    end
    if solidNodes != :default && size(solidNodes) == size(wallRegion)
        wallRegion = wallRegion .|| solidNodes

        append!(schemes, [:bounceBack])
    end
    dims <= 2 && (wallRegion = sparse(wallRegion))

    if :bounceBack in schemes || :trt in schemes
        massDensity[wallRegion] .= 0;
        streamingInvasionRegions, oppositeVectorId = bounceBackPrep(wallRegion, velocities);
        boundaryConditionsParams = merge(boundaryConditionsParams, (; wallRegion, streamingInvasionRegions, oppositeVectorId));
    end

    #= -------------------- boundary conditions (moving walls) ------------------- =#
    if solidNodeVelocity isa Array && solidNodeVelocity[1] isa Vector && size(solidNodeVelocity) == size(massDensity)
        maskedArray = [[0. for _ in 1:dims] for _ in solidNodeVelocity]
        maskedArray[wallRegion] = solidNodeVelocity[wallRegion]
        boundaryConditionsParams = merge(boundaryConditionsParams, (; solidNodeVelocity = maskedArray));

        append!(schemes, [:movingWalls])
    end

    #= --------------------------- forcing scheme prep --------------------------- =#
    # the default forcing scheme is Guo
    forcingScheme == :default && (forcingScheme = :guo)

    # by defualt, there is no force density
    if forceDensity == :default
        forceDensity = [[0., 0]];
    # if a single vector is defined it is assumed the force denisty is constant
    elseif forceDensity isa Vector && size(forceDensity) == size(fluidVelocity[1])
        forceDensity = [forceDensity |> Vector{Float64} for _ in massDensity];
        forceDensity[wallRegion] = [[0.,0] for _ in forceDensity[wallRegion]]
        kwInitialConditions = merge(kwInitialConditions, (; forceDensity, Δt))

        append!(schemes, [forcingScheme])
    # if a force density field is defined its dimensions are verified
    elseif size(forceDensity) == size(massDensity)
        forceDensity[wallRegion] = [[0.,0] for _ in forceDensity[wallRegion]]
        kwInitialConditions = merge(kwInitialConditions, (; forceDensity, Δt))

        append!(schemes, [forcingScheme])
    # if none of the above, the dimensions must be wrong
    else
        error("force density does not have consistent dimensions!")
    end

    #= --------------------------- initial distributions are found --------------------------- =#
    initialDistributions = [
        findInitialConditions(
            id,
            velocities,
            fluidParams,
            massDensity,
            fluidVelocity,
            Δx_Δt;
            kwInitialConditions = kwInitialConditions
        ) 
    for id in eachindex(velocities)]

    #= ---------------------------------- saving data setup ---------------------------------- =#
    if saveData
        append!(schemes, [:saveData])
        mkOutputDirs()
    end

    #= ------------------------- the model is initialized ------------------------ =#
    model = LBMmodel(
        spaceTime, # space step (Δx), time step (Δt), space coordinate (x), Δt/Δx, dimensionality (dims)
        tick,
        time, # not in spaceTime bc NamedTuple are immutable!
        fluidParams, # speed of sound and its powers (c_s, c2_s, c4_s), relaxation time (τ)
        (; massDensity = massDensity), # ρ₀
        massDensity, # mass density
        massDensity.*fluidVelocity, # momentum density
        fluidVelocity, # fluid velocity
        forceDensity,
        initialDistributions, # f_i(x, t) for all t
        velocities, # c_i for all i
        boundaryConditionsParams, # stream invasion regions and index j such that c[i] = -c[j]
        []|>Vector{LBMparticle},
        unique(schemes)
    );

    #= ---------------------------- consistency check ---------------------------- =#
    # to ensure consistency, ρ, ρu and u are all found using the initial conditions of f_i
    hydroVariablesUpdate!(model);
    # if either ρ or u changed, the user is notified
    acceptableError = 0.01;
    fluidRegion = wallRegion .|> b -> !b;
    error_massDensity = (model.massDensity[fluidRegion] - massDensity[fluidRegion] .|> abs)  |> maximum
    error_fluidVelocity = (model.fluidVelocity[fluidRegion] - fluidVelocity[fluidRegion] .|> norm) |> maximum
    if (error_massDensity > acceptableError) || (error_fluidVelocity > acceptableError)
        @warn "the initial conditions for ρ and u could not be met. New ones were defined."
    end

    :saveData in model.schemes && writeTrajectories(model)

    # the model is returned
    return model
end
