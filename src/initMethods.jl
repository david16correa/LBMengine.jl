#= ==========================================================================================
=============================================================================================
init methods
=============================================================================================
========================================================================================== =#

function addBead!(model::LBMmodel;
    massDensity = 1.0,
    radius = 0.1,
    position = :default, # default: origin (actual value is dimensionality dependent)
    velocity = :default, # default: static (actual value is dimensionality dependent)
    angularVelocity = :default, # default: static, (actual value is dimensionality dependent)
    coupleTorques = false,
    coupleForces = false,
    scheme = :default # default: ladd
)
    # a local function for the general geometry of a centered bead (sphere) is defined
    beadGeometry(x::Vector; radius2 = 0.0625) = sum(x.^2) < radius2

    # the mass is found using the mass density
    mass = massDensity * sum(beadGeometry.(model.spaceTime.X; radius2 = radius^2)) * model.spaceTime.latticeParameter^model.spaceTime.dims

    # position and velocity are defined if necessary
    position == :default && (position = fill(0., model.spaceTime.dims))
    velocity == :default && (velocity = fill(0., model.spaceTime.dims))
    # the dimensions are checked
    ((length(position) != length(velocity)) || (length(position) != model.spaceTime.dims)) && error("The position and velocity dimensions must match the dimensionality of the model! dims = $(model.spaceTime.dims)")

    # the moment of inertia, initial angular velocity, and angular momentum input are all initialized
    if model.spaceTime.dims == 2
        momentOfInertia = 0.5 * mass * radius^2 # moment of inertia for a disk
        angularVelocity == :default && (angularVelocity = 0.)
        angularMomentumInput = 0.
        @assert (angularVelocity isa Number) "In two dimensions, angularVelocity must be a number!"
    elseif model.spaceTime.dims == 3
        momentOfInertia = 0.4 * mass * radius^2 # moment of inertia for a sphere
        angularVelocity == :default && (angularVelocity = [0., 0, 0])
        angularMomentumInput = [0., 0, 0]
        @assert (angularVelocity isa Array) "In three dimensions, angularVelocity must be an array!"
    else
        error("For particle simulation dimensionality must be either 2 or 3! dims = $(model.spaceTime.dims)")
    end

    # the momentum input is defined, and the inputs are turned into vectors to allow for multithreading
    momentumInput = fill(0., model.spaceTime.dims);

    momentumInput = fill(momentumInput, length(model.velocities) + 1)
    angularMomentumInput = fill(angularMomentumInput, length(model.velocities) + 1)

    # a new bead is defined and added to the model
    newBead = LBMparticle(
        length(model.particles) + 1,
        (;radius, inverseMass = 1/mass, inverseMomentOfInertia = 1/momentOfInertia, solidRegionGenerator = x -> beadGeometry(x; radius2 = radius^2), properties = [:spherical, :bead], coupleTorques, coupleForces),
        (; solidRegion = [], streamingInvasionRegions = []),
        position,
        velocity,
        angularVelocity,
        [],
        momentumInput,
        angularMomentumInput,
    )
    append!(model.particles, [newBead]);

    # the schemes of the model are managed
    scheme == :default && (scheme = :ladd)
    @assert (scheme == :psm || scheme == :ladd) "$(scheme) cannot be used as a particle-fluid collision scheme!"

    @assert (newBead.id == 1 || scheme in model.schemes) "$(scheme) cannot be used, as another scheme for particle-fluid collision is being used"

    append!(model.schemes, [scheme])
    model.schemes = model.schemes |> unique
    if !(:bounceBack in model.schemes)
        wallRegion = fill(false, size(model.massDensity))
        (model.spaceTime.dims <= 2) ? (wallRegion = sparse(wallRegion)) : (wallRegion = BitArray(wallRegion))
        streamingInvasionRegions, oppositeVectorId = bounceBackPrep(wallRegion, model.velocities);
        model.boundaryConditionsParams = merge(model.boundaryConditionsParams, (; wallRegion, streamingInvasionRegions, oppositeVectorId));
        scheme == :ladd && (append!(model.schemes, [:bounceBack]))
    end

    moveParticles!(length(model.particles), model; initialSetup = true)

    # saving data
    :saveData in model.schemes && writeParticleTrajectory(model.particles[end], model)

    return nothing
end

function addSquirmer!(model::LBMmodel;
    massDensity = 1.0,
    radius = 0.1,
    position = :default, # default: origin (actual value is dimensionality dependent)
    velocity = :default, # default: static (actual value is dimensionality dependent)
    slipSpeed = :default,
    swimmingSpeed = :default,
    swimmingDirection = :default, # default: x-direction
    B1 = :default, # default: 1/normFactor (the normFactor is used to control the maximum slip speed)
    B2 = :default, # default: 0
    beta = :default, # default: B2/B1 = 0
    coupleTorques = false,
    coupleForces = false,
    scheme = :default # default: ladd
)
    # a local function for the general geometry of a centered bead (sphere) is defined
    beadGeometry(x::Vector; radius2 = 0.0625) = sum(x.^2) < radius2

    # the mass is found using the mass density
    mass = massDensity * sum(beadGeometry.(model.spaceTime.X; radius2 = radius^2)) * model.spaceTime.latticeParameter^model.spaceTime.dims

    # position and velocity are defined if necessary
    position == :default && (position = fill(0., model.spaceTime.dims))
    velocity == :default && (velocity = fill(0., model.spaceTime.dims))
    # the dimensions are checked
    ((length(position) != length(velocity)) || (length(position) != model.spaceTime.dims)) && error("The position and velocity dimensions must match the dimensionality of the model! dims = $(model.spaceTime.dims)")

    # the moment of inertia, initial angular velocity, and angular momentum input are all initialized
    if model.spaceTime.dims == 2
        momentOfInertia = 0.5 * mass * radius^2 # moment of inertia for a disk
        angularMomentumInput = 0.
        angularVelocity = 0.
    elseif model.spaceTime.dims == 3
        momentOfInertia = 0.4 * mass * radius^2 # moment of inertia for a sphere
        angularMomentumInput = [0., 0, 0]
        angularVelocity = [0., 0, 0]
    else
        error("For particle simulation dimensionality must be either 2 or 3! dims = $(model.spaceTime.dims)")
    end

    # the momentum input is defined, and the inputs are turned into vectors to allow for multithreading
    momentumInput = fill(0., model.spaceTime.dims);

    momentumInput = fill(momentumInput, length(model.velocities) + 1)
    angularMomentumInput = fill(angularMomentumInput, length(model.velocities) + 1)

    # B1 and B2 are chosen
    @assert any(x -> x == :default, [B1, B2, beta]) "B1, B2, and beta cannot be all simultaneously defined!"
    @assert all(x -> x == :default || x isa Number, [B1, B2, beta]) "B1, B2, and beta can only be numbers!"

    if beta == :default
        B1 == :default && (B1 = 1);
        B2 == :default && (B2 = 0);
        beta = B2/B1; # beta might be useful later
    else
        if B2 == :default
            B1 == :default && (B1 = 1);
            B2 = B1*beta
        else
            @assert beta != 0 "if beta = 0, then B2 can only be zero! In this case, do not declare both."
            B1 = B2/beta
        end
    end

    # the swimming direction is worked out
    if swimmingDirection == :default
        swimmingDirection = fill(0., model.spaceTime.dims);
        swimmingDirection[1] = 1;
    end
    @assert length(swimmingDirection) == model.spaceTime.dims  "the swimming direction must be $(dims)-dimensional!"
    # the swimming direction is normalized
    swimmingDirection = swimmingDirection |> v -> v/norm(v)

    # B1 and B2 are rescaled to ensure either the slip speed or swimming speed is satisfied
    @assert any(x -> x == :default, [slipSpeed, swimmingSpeed]) "slipSpeed and swimmingSpeed cannot be all simultaneously defined!"
    if slipSpeed == :default
        (swimmingSpeed == :default) && (swimmingSpeed = 1e-3);
        @assert swimmingSpeed isa Number "swimmingSpeed must be a number!"
        # the sign in the swimming speed will alter the direction, but not the speed itself
        swimmingDirection *= sign(swimmingSpeed)
        B1 = 3/2 * swimmingSpeed |> abs
        B2 = B1*beta
    else
        @assert slipSpeed isa Number "slip speed must be a number!"
        normFactor = [abs((B1 + B2 * cos(theta)) * sin(theta)) for theta in range(-pi, stop=pi, length=100)] |> maximum
        B1 *= slipSpeed/normFactor
        B2 *= slipSpeed/normFactor
    end

    # a new squirmer is defined and added to the model
    newSquirmer = LBMparticle(
        length(model.particles) + 1,
        (;  radius,
            B1,
            B2,
            swimmingDirection,
            inverseMass = 1/mass,
            inverseMomentOfInertia = 1/momentOfInertia,
            solidRegionGenerator = x -> beadGeometry(x; radius2 = radius^2),
            properties = [:spherical, :squirmer],
            coupleTorques,
            coupleForces), # particleParams
        (; solidRegion = [], streamingInvasionRegions = []),
        position,
        velocity,
        angularVelocity,
        [],
        momentumInput,
        angularMomentumInput,
    )
    append!(model.particles, [newSquirmer]);

    # the schemes of the model are managed
    scheme == :default && (scheme = :ladd)
    @assert (scheme == :psm || scheme == :ladd) "$(scheme) cannot be used as a particle-fluid collision scheme!"
    @assert (newSquirmer.id == 1 || scheme in model.schemes) "$(scheme) cannot be used, as another scheme for particle-fluid collision is being used"
    append!(model.schemes, [scheme])
    model.schemes = model.schemes |> unique
    if !(:bounceBack in model.schemes)
        wallRegion = fill(false, size(model.massDensity))
        (model.spaceTime.dims <= 2) ? (wallRegion = sparse(wallRegion)) : (wallRegion = BitArray(wallRegion))
        streamingInvasionRegions, oppositeVectorId = bounceBackPrep(wallRegion, model.velocities);
        model.boundaryConditionsParams = merge(model.boundaryConditionsParams, (; wallRegion, streamingInvasionRegions, oppositeVectorId));
        scheme == :ladd && (append!(model.schemes, [:bounceBack]))
    end

    moveParticles!(length(model.particles), model; initialSetup = true)

    # saving data
    :saveData in model.schemes && writeParticleTrajectory(model.particles[end], model)

    return nothing
end

"Initializes f_i to f^eq_i, which is the simplest strategy."
function findInitialConditions(id::Int64, velocities::Vector{LBMvelocity}, fluidParams::NamedTuple, massDensity::Array{Float64}, u::Array{Vector{Float64}}, Δx_Δt::Float64; kwInitialConditions = (; )) 
    # the quantities to be used are saved separately
    ci = velocities[id].c .* Δx_Δt
    wi = velocities[id].w
    if :forceDensity in (kwInitialConditions |> keys)
        consistencyTerm = fill(zero(kwInitialConditions.forceDensity[1]), size(massDensity))
        fluidIndices = (massDensity .≈ 0) .|> b -> !b;
        consistencyTerm[fluidIndices] = kwInitialConditions.forceDensity[fluidIndices] ./ massDensity[fluidIndices]
        u -= 0.5 * kwInitialConditions.Δt * consistencyTerm
    end
    # the equilibrium distribution is found step by step and returned
    firstStep = vectorFieldDotVector(u, ci) |> udotci -> udotci*fluidParams.invC2_s + udotci.^2 * (0.5 * fluidParams.invC4_s)
    secondStep = firstStep - vectorFieldDotVectorField(u, u)*(0.5*fluidParams.invC2_s) .+ 1
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
    xlims = :default,
    ylims = :default,
    zlims = :default,
    latticeParameter = :default,
    dims = 2, # default mode must be added!!
    Δt = :default, # default: Δt = Δx
    walledDimensions = :default, # walls around y axis (all non-walled dimensions are periodic!)
    dampenEcho = false, # use characteristic boundary conditions to absorb sound waves
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

    #= - the lattice parameter, relaxation time ratio, and viscosity are sorted out - =#
    @assert any(x -> x == :default, [latticeParameter, relaxationTimeRatio, viscosity]) "latticeParameter, relaxationTimeRatio, and viscosity cannot be all simultaneously defined!"

    if latticeParameter == :default
        (relaxationTimeRatio == :default) && (relaxationTimeRatio = 15);
        (viscosity == :default) && (viscosity = 0.890);
        latticeParameter = 3*viscosity/(relaxationTimeRatio - 0.5)
    else
        if relaxationTimeRatio == :default
            (viscosity == :default) && (viscosity = 0.890);
            relaxationTimeRatio = 3*viscosity/latticeParameter + 0.5
        else
            viscosity = latticeParameter/3 * (relaxationTimeRatio - 0.5)
        end
    end

    #= -------------------------- space time is sorted out -------------------------- =#
    (xlims == :default) && (xlims = (0,1));
    (ylims == :default) && (ylims = xlims);
    (zlims == :default) && (zlims = xlims);

    @assert all(t -> t[2]!=t[1], [xlims, ylims, zlims]) "Coordinate lower and upper limits cannot be equal! Current limits are $([xlims, ylims, zlims][1:dims])"

    x = range(xlims[1], stop = xlims[2], step = latticeParameter)
    y = range(ylims[1], stop = ylims[2], step = latticeParameter)
    z = range(zlims[1], stop = zlims[2], step = latticeParameter)

    coordinates, expectedSizes = [x,y,z][1:dims] |> v -> (v, length.(v))

    #= -------------------------- hydrodynamic variables -------------------------- =#
    # if default conditions were chosen, ρ is built. Otherwise its dimensions are verified
    if massDensity == :default
        massDensity = [length(coordinates[id]) for id in 1:dims] |> v -> ones(v...)
    elseif massDensity isa Number
        massDensity = [length(coordinates[id]) for id in 1:dims] |> v -> massDensity * ones(v...)
    else
        @assert all(id -> size(massDensity)[id]==expectedSizes[id], eachindex(expectedSizes)) "The mass density given has the wrong size! $(expectedSizes) expected."
    end

    # if default conditions were chosen, u is built. Otherwise its dimensions are verified
    if fluidVelocity == :default
        fluidVelocity = fill(fill(0., dims), size(massDensity))
    elseif size(fluidVelocity) |> length == 1
        fluidVelocity = fill(fluidVelocity |> Array{Float64}, size(massDensity))
    else
        @assert all(id -> size(fluidVelocity)[id]==expectedSizes[id], eachindex(expectedSizes)) "The fluid velocity given has the wrong size! $(expectedSizes) expected."
    end

    #= ------------------------ choosing the velocity set ----------------------- =#
    # if the user did not define a velocity set, then a preset is chosen
    if velocities == :default
        # if dimensions are too large, and the user did not define a velocity set, then there's an error
        @assert (dims <= 3) "for dimensions higher than 3 a velocity set must be defined using a Vector{LBMvelocity}! modelInit(...; velocities = yourInput)"
        velocities = [[D1Q3]; [D2Q9]; [D3Q15]] |> v -> v[dims]
    else
        id = findfirst(set -> set == velocities, [:D1Q3, :D2Q9, :D3Q15, :D3Q19, :D3Q27])
        @assert id isa Number "If a default velocity set is chosen, it must be one of the following: [:D1Q3, :D2Q9, :D3Q15, :D3Q19, :D3Q27]"
        velocities = [D1Q3, D2Q9, D3Q15, D3Q19, D3Q27][id]
    end
    @assert (velocities isa Vector{LBMvelocity}) "Please input a velocity set using a Vector{LBMvelocity}!"
    @assert (velocities[1].c |> length == dims) "The chosen velocity must be $dims-dimensional!"

    #= ---------------- space and time variables are initialized ---------------- =#
    # by default Δt = Δx, as this is the most stable
    (Δt == :default) && (Δt = latticeParameter);
    # since Δx/Δt is often used, its value is stored to avoid redundant calculations
    Δx_Δt = latticeParameter/Δt |> Float64
    X = [[coordinates[id][Id[id]] for id in Id |> Tuple |> eachindex]  for Id in eachindex(IndexCartesian(), massDensity)]
    spaceTime = (; coordinates, X, latticeParameter, Δt, Δx_Δt, dims); 
    tick, time = 0, 0.;

    #= -------------------- fluid parameters are initialized -------------------- =#
    c_s, c2_s, c4_s = Δx_Δt/√3, Δx_Δt^2 / 3, Δx_Δt^4 / 9;
    invC_s, invC2_s, invC4_s = √3/Δx_Δt, 3/Δx_Δt^2, 9/Δx_Δt^4;
    collisionModel == :default && (collisionModel = :trt)
    relaxationTime = relaxationTimeRatio * Δt;

    @assert (collisionModel == :bgk || collisionModel == :trt) "Collision model $collisionModel is not implemented!"

    if collisionModel == :bgk
        fluidParams = (; c_s, c2_s, c4_s, invC_s, invC2_s, invC4_s, relaxationTime, viscosity, isFluidCompressible);
    elseif collisionModel == :trt
        relaxationTimePlus = relaxationTime
        kineticParameter == :debug && (kineticParameter = viscosity^2/(c4_s * Δt^2)) # here, :trt = :bgk
        relaxationTimeMinus = kineticParameter * c2_s * Δt^2 / viscosity + Δt/2
        fluidParams = (; c_s, c2_s, c4_s, invC_s, invC2_s, invC4_s, relaxationTime, viscosity, relaxationTimePlus, relaxationTimeMinus, isFluidCompressible);
    end
    append!(schemes, [collisionModel])

    #= -------------------- boundary conditions (bounce back) -------------------- =#
    wallRegion = fill(false, size(massDensity))
    if walledDimensions != :default && length(walledDimensions) != 0
        if dampenEcho
            append!(schemes, [:cbc])
        else
            wallRegion = wallNodes(massDensity; walledDimensions = walledDimensions);
            append!(schemes, [:bounceBack])
        end

        boundaryConditionsParams = merge(boundaryConditionsParams, (; walledDimensions));
    end
    if solidNodes != :default && size(solidNodes) == size(wallRegion)
        wallRegion = wallRegion .|| solidNodes

        append!(schemes, [:bounceBack])
    end
    dims <= 2 ? (wallRegion = sparse(wallRegion)) : (wallRegion = BitArray(wallRegion))

    if :bounceBack in schemes || :trt in schemes
        massDensity[wallRegion] .= 0;
        streamingInvasionRegions, oppositeVectorId = bounceBackPrep(wallRegion, velocities);
        boundaryConditionsParams = merge(boundaryConditionsParams, (; wallRegion, streamingInvasionRegions, oppositeVectorId));
    end

    #= -------------------- boundary conditions (moving walls) ------------------- =#
    if solidNodeVelocity isa Array && solidNodeVelocity[1] isa Vector && size(solidNodeVelocity) == size(massDensity)
        maskedArray = fill(fill(0., dims), size(solidNodeVelocity))
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
        forceDensity = fill(forceDensity |> Vector{Float64}, size(massDensity))
        forceDensity[wallRegion] = [fill(0., dims) for _ in forceDensity[wallRegion]]
        kwInitialConditions = merge(kwInitialConditions, (; forceDensity, Δt))

        append!(schemes, [forcingScheme])
    # if a force density field is defined its dimensions are verified
    elseif size(forceDensity) == size(massDensity)
        forceDensity[wallRegion] = [fill(0., dims) for _ in forceDensity[wallRegion]]
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
        rm("output.lbm"; force=true, recursive=true)
        mkdir("output.lbm")
    end

    #= ---------------------------------- GPU acceleration ---------------------------------- =#
    gpuImmutable = (;)
    gpuTmp = (;)
    if CUDA.functional()
        gpuImmutable = merge(gpuImmutable, (;
            cs = [velocity.c |> CuArray{Int8} for velocity in velocities],
            forceDensity = [f[k] for f in forceDensity, k in 1:dims]|>CuArray{Float64}
        ));
    end

    #= ------------------------------ the model is initialized ------------------------------ =#
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
        []|>Vector{LBMparticle}, # initially there will be no particles
        unique(schemes),
        gpuImmutable,
        gpuTmp,
    );

    #= ---------------------------- consistency check ---------------------------- =#
    # to ensure consistency, ρ, ρu and u are all found using the initial conditions of f_i
    hydroVariablesUpdate!(model);
    # if either ρ or u changed, the user is notified
    acceptableError = 0.01;
    fluidRegion = wallRegion .|> b -> !b;
    error_massDensity = model.massDensity[fluidRegion] - massDensity[fluidRegion] |> M -> abs.(M) |> maximum
    error_fluidVelocity = model.fluidVelocity[fluidRegion] - fluidVelocity[fluidRegion] |> M -> norm.(M) |> maximum
    if (error_massDensity > acceptableError) || (error_fluidVelocity > acceptableError)
        @warn "the initial conditions for ρ and u could not be met. New ones were defined."
    end

    :saveData in model.schemes && writeTrajectories(model)

    # the model is returned
    return model
end
