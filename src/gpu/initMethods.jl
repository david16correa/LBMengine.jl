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
    # the mass is found using the mass density
    mass = massDensity * sum(getSphere(radius, model.spaceTime.X)) * model.spaceTime.latticeParameter^model.spaceTime.dims

    # position and velocity are defined if necessary
    position == :default ? (position = CUDA.fill(0., model.spaceTime.dims)) : (position = position |> CuArray{Float64})
    velocity == :default ? (velocity = CUDA.fill(0., model.spaceTime.dims)) : (velocity = velocity |> CuArray{Float64})
    # the dimensions are checked
    ((length(position) != length(velocity)) || (length(position) != model.spaceTime.dims)) && error("The position and velocity dimensions must match the dimensionality of the model! dims = $(model.spaceTime.dims)")

    # the moment of inertia, initial angular velocity, and angular momentum input are all initialized
    if model.spaceTime.dims == 2
        momentOfInertia = 0.5 * mass * radius^2 # moment of inertia for a disk
        angularVelocity == :default && (angularVelocity = 0.)
        torqueInput = 0.
        @assert (angularVelocity isa Number) "In two dimensions, angularVelocity must be a number!"
    elseif model.spaceTime.dims == 3
        momentOfInertia = 0.4 * mass * radius^2 # moment of inertia for a sphere
        angularVelocity == :default ? (angularVelocity = [0., 0, 0] |> CuArray{Float64}) : (angularVelocity = angularVelocity |> CuArray{Float64})
        torqueInput = [0., 0, 0] |> CuArray{Float64}
        @assert (angularVelocity isa CuArray{Float64}) "In three dimensions, angularVelocity must be an array!"
    else
        error("For particle simulation dimensionality must be either 2 or 3! dims = $(model.spaceTime.dims)")
    end

    # the momentum input is defined
    forceInput = CUDA.fill(0., model.spaceTime.dims);

    solidRegion = fillCuArray(false, size(model.massDensity))

    particleParams = (;
        radius,
        inverseMass = 1/mass,
        inverseMomentOfInertia = 1/momentOfInertia,
        properties = [:spherical, :bead],
        coupleTorques,
        coupleForces
    )

    # a new bead is defined and added to the model
    newBead = LBMparticle(
        length(model.particles) + 1,
        particleParams,
        (; solidRegion, streamingInvasionRegions = []),
        position,
        velocity,
        angularVelocity,
        [],
        forceInput,
        torqueInput,
    )
    append!(model.particles, [newBead]);

    # the schemes of the model are managed
    scheme == :default && (scheme = :ladd)
    @assert (scheme == :psm || scheme == :ladd) "$(scheme) cannot be used as a particle-fluid collision scheme!"
    @assert (scheme == :ladd) "$scheme is not implemented in the gpu modules!"
    @assert (newBead.id == 1 || scheme in model.schemes) "$(scheme) cannot be used, as another scheme for particle-fluid collision is being used"

    append!(model.schemes, [scheme])
    model.schemes = model.schemes |> unique
    if !(:bounceBack in model.schemes)
        streamingInvasionRegions, oppositeVectorId = bounceBackPrep(solidRegion, model.velocities);
        model.boundaryConditionsParams = merge(model.boundaryConditionsParams, (; wallRegion = solidRegion, streamingInvasionRegions, oppositeVectorId));
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
    # the mass is found using the mass density
    mass = massDensity * sum(getSphere(radius, model.spaceTime.X)) * model.spaceTime.latticeParameter^model.spaceTime.dims

    # position and velocity are defined if necessary
    position == :default ? (position = CUDA.fill(0., model.spaceTime.dims)) : (position = position |> CuArray{Float64})
    velocity == :default ? (velocity = CUDA.fill(0., model.spaceTime.dims)) : (velocity = velocity |> CuArray{Float64})
    # the dimensions are checked
    ((length(position) != length(velocity)) || (length(position) != model.spaceTime.dims)) && error("The position and velocity dimensions must match the dimensionality of the model! dims = $(model.spaceTime.dims)")

    # the moment of inertia, initial angular velocity, and angular momentum input are all initialized
    if model.spaceTime.dims == 2
        momentOfInertia = 0.5 * mass * radius^2 # moment of inertia for a disk
        torqueInput = 0.
        angularVelocity = 0.
    else
        momentOfInertia = 0.4 * mass * radius^2 # moment of inertia for a sphere
        torqueInput = [0., 0, 0] |> CuArray{Float64}
        angularVelocity = [0., 0, 0] |> CuArray{Float64}
    end

    # the momentum input is defined
    forceInput = CUDA.fill(0., model.spaceTime.dims);

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
    swimmingDirection = swimmingDirection |> v -> v/norm(v) |> CuArray{Float64}

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

    solidRegion = fillCuArray(false, size(model.massDensity))

    particleParams = (; 
        radius,
        B1,
        B2,
        swimmingDirection,
        inverseMass = 1/mass,
        inverseMomentOfInertia = 1/momentOfInertia,
        properties = [:spherical, :squirmer],
        coupleTorques,
        coupleForces
    )

    # a new squirmer is defined and added to the model
    newSquirmer = LBMparticle(
        length(model.particles) + 1,
        particleParams, # particleParams
        (; solidRegion, streamingInvasionRegions = []),
        position,
        velocity,
        angularVelocity,
        [],
        forceInput,
        torqueInput,
    )
    append!(model.particles, [newSquirmer]);

    # the schemes of the model are managed
    scheme == :default && (scheme = :ladd)
    @assert (scheme == :psm || scheme == :ladd) "$(scheme) cannot be used as a particle-fluid collision scheme!"
    @assert (scheme == :ladd) "$scheme is not implemented in the gpu modules!"
    @assert (newSquirmer.id == 1 || scheme in model.schemes) "$(scheme) cannot be used, as another scheme for particle-fluid collision is being used"
    append!(model.schemes, [scheme])
    model.schemes = model.schemes |> unique
    if !(:bounceBack in model.schemes)
        streamingInvasionRegions, oppositeVectorId = bounceBackPrep(solidRegion, model.velocities);
        model.boundaryConditionsParams = merge(model.boundaryConditionsParams, (; wallRegion = solidRegion, streamingInvasionRegions, oppositeVectorId));
        scheme == :ladd && (append!(model.schemes, [:bounceBack]))
    end

    moveParticles!(length(model.particles), model; initialSetup = true)

    # saving data
    :saveData in model.schemes && writeParticleTrajectory(model.particles[end], model)

    return nothing
end

function addLinearBond!(model::LBMmodel, id1::Int64, id2::Int64; equilibriumDisp = :default, hookConstant = 1e-1)
    particle1 = model.particles[id1]
    particle2 = model.particles[id2]
    equilibriumDisp == :default && (equilibriumDisp = particle2.position - particle1.position |> Array |> norm)
    @assert equilibriumDisp isa Number "equilibriumDisp must be a number!"
    @assert hookConstant isa Number "hookConstant must be a number!"

    append!(model.particleBonds, [(; id1, id2, equilibriumDisp, hookConstant, type=:linear)])
    return nothing
end

function addPolarBond!(model::LBMmodel, id1::Int64, id2::Int64, id3::Int64; equilibriumAngle = :default, hookConstant = 1e-1)
    particle1 = model.particles[id1]
    particle2 = model.particles[id2]
    particle3 = model.particles[id3]

    vecA = particle1.position - particle2.position
    normA = vecA |> Array |> norm
    vecB = particle3.position - particle2.position
    normB = vecB |> Array |> norm

    cosAlpha = sum(vecA .* vecB) / (normA * normB)

    equilibriumAngle == :default && (equilibriumAngle = acos(cosAlpha))
    @assert equilibriumAngle isa Number "equilibriumAngle must be a number!"
    @assert hookConstant isa Number "hookConstant must be a number!"

    append!(model.particleBonds, [(; id1, id2, id3, equilibriumAngle, hookConstant, type=:polar)])
    return nothing
end

"Initializes f_i to f^eq_i, which is the simplest strategy."
function findInitialConditions(
    id::Int64,
    velocities::NamedTuple,
    fluidParams::NamedTuple,
    massDensity::CuArray{Float64},
    u::CuArray{Float64},
    latticeSpeed::Float64;
    kwInitialConditions = (; )
)
    # the quantities to be used are saved separately
    ci = velocities.cs[id] .* latticeSpeed
    wi = velocities.ws[id]
    if :forceDensity in (kwInitialConditions |> keys)
        consistencyTerm = findFluidVelocity(massDensity, kwInitialConditions.forceDensity) # it turns out findFluidVelocity() is useful here; it'll divide F/ρ
        u -= 0.5 * kwInitialConditions.timeStep * consistencyTerm
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
    timeStep = :default, # default: Δt = Δx
    walledDimensions = :default, # walls around y axis (all non-walled dimensions are periodic!)
    dampenEcho = false, # use characteristic boundary conditions to absorb sound waves
    solidNodes = :default, # default: no solid nodes (other than the walls) 
    solidNodeVelocity = :default, # default: static solids - u = [0,0]
    isFluidCompressible = true, # this is more stable for several schemes. E.g. ladd
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

    coordinates, modelSize = (x,y,z)[1:dims] |> v -> (v, length.(v))

    #= -------------------------- hydrodynamic variables -------------------------- =#
    # if default conditions were chosen, ρ is built. Otherwise its dimensions are verified
    if massDensity == :default
        massDensity = fillCuArray(1., modelSize)
    elseif massDensity isa Number
        massDensity = fillCuArray(massDensity|>Float64, modelSize)
    else
        @assert all(id -> size(massDensity)[id]==modelSize[id], eachindex(modelSize)) "The mass density given has the wrong size! $(modelSize) expected."
        massDensity = massDensity |> CuArray{Float64}
    end

    # if default conditions were chosen, u is built. Otherwise its dimensions are verified
    if fluidVelocity == :default
        fluidVelocity = fillCuArray(fill(0., dims), modelSize)
    elseif size(fluidVelocity) |> length == 1
        fluidVelocity = fillCuArray(fluidVelocity|>Array{Float64}, modelSize)
    else
        @assert all(id -> size(fluidVelocity)[id]==modelSize[id], eachindex(modelSize)) "The fluid velocity given has the wrong size! $(modelSize) expected."
        fluidVelocity = fluidVelocity |> CuArray{Float64}
    end

    #= ------------------------ choosing the velocity set ----------------------- =#
    # if the user did not define a velocity set, then a preset is chosen
    if velocities == :default
        # if dimensions are too large, and the user did not define a velocity set, then there's an error
        @assert (dims > 1) "1 dimensional LBM is not implemented in CUDA!"
        @assert (dims <= 3) "for dimensions higher than 3 a velocity set must be defined using a NamedTuple! modelInit(...; velocities = (; cs::Vector{Vector{Int8}}, ws::Vector{Float64})"
        velocities = [:D1Q3, D2Q9, D3Q15, D3Q19, D3Q27][dims]
    else
        id = findfirst(set -> set == velocities, [:D1Q3, :D2Q9, :D3Q15, :D3Q19, :D3Q27])
        @assert id isa Number "If a default velocity set is chosen, it must be one of the following: [:D1Q3, :D2Q9, :D3Q15, :D3Q19, :D3Q27]"
        velocities = [:D1Q3, D2Q9, D3Q15, D3Q19, D3Q27][id]
    end
    @assert (velocities.cs[1] |> length == dims) "The chosen velocity must be $dims-dimensional!"
    velocities = (; cs = velocities.cs.|>CuArray{Int8}, ws = velocities.ws)

    #= ---------------- space and time variables are initialized ---------------- =#
    # by default Δt = Δx, as this is the most stable
    (timeStep == :default) && (timeStep = latticeParameter);
    # latticeSpeed := Δx/Δt
    latticeSpeed = latticeParameter/timeStep |> Float64
    X = coordinatesCuArray(coordinates)
    spaceTime = (; coordinates, X, latticeParameter, timeStep, latticeSpeed, dims);
    tick, time = 0, 0.;

    #= -------------------- fluid parameters are initialized -------------------- =#
    c_s, c2_s, c4_s = latticeSpeed/√3, latticeSpeed^2 / 3, latticeSpeed^4 / 9;
    invC_s, invC2_s, invC4_s = √3/latticeSpeed, 3/latticeSpeed^2, 9/latticeSpeed^4;
    collisionModel == :default && (collisionModel = :trt)
    relaxationTime = relaxationTimeRatio * timeStep;

    @assert (collisionModel == :bgk || collisionModel == :trt) "Collision model $collisionModel is not implemented!"

    if collisionModel == :bgk
        fluidParams = (; c_s, c2_s, c4_s, invC_s, invC2_s, invC4_s, relaxationTime, viscosity, isFluidCompressible);
    elseif collisionModel == :trt
        relaxationTimePlus = relaxationTime
        kineticParameter == :debug && (kineticParameter = viscosity^2/(c4_s * timeStep^2)) # here, :trt = :bgk
        relaxationTimeMinus = kineticParameter * c2_s * timeStep^2 / viscosity + timeStep/2
        fluidParams = (; c_s, c2_s, c4_s, invC_s, invC2_s, invC4_s, relaxationTime, viscosity, relaxationTimePlus, relaxationTimeMinus, isFluidCompressible);
    end
    append!(schemes, [collisionModel])

    #= -------------------- boundary conditions (bounce back) -------------------- =#
    wallRegion = fillCuArray(false, modelSize)
    if walledDimensions != :default && length(walledDimensions) != 0
        if dampenEcho
            @error ":cbc is not currently implemented in the gpu modules! Please use the cpu modules instead."
            append!(schemes, [:cbc])
        else
            wallRegion = wallNodes(modelSize, walledDimensions);
            append!(schemes, [:bounceBack])
        end

        boundaryConditionsParams = (; boundaryConditionsParams..., walledDimensions, wallRegion)
    end
    if solidNodes != :default && size(solidNodes) == size(wallRegion)
        wallRegion = wallRegion .|| solidNodes|>cu

        append!(schemes, [:bounceBack])
    end

    if :bounceBack in schemes || :trt in schemes
        massDensity[wallRegion] .= 0;
        fluidVelocity[vectorBitId(wallRegion)] .= 0;
        streamingInvasionRegions, oppositeVectorId = bounceBackPrep(wallRegion, velocities);
        :bounceBack in schemes && (boundaryConditionsParams = (; boundaryConditionsParams..., streamingInvasionRegions));
        :trt in schemes && (boundaryConditionsParams = (; boundaryConditionsParams..., oppositeVectorId));
    end

    #= -------------------- boundary conditions (moving walls) ------------------- =#
    if solidNodeVelocity isa Array && solidNodeVelocity[1] isa Vector && size(solidNodeVelocity) == size(massDensity)
        maskedArray = fillCuArray(fill(0., dims), modelSize)
        maskedArray[vectorBitId(wallRegion)] = solidNodeVelocity[vectorBitId(wallRegion)]
        boundaryConditionsParams = (; boundaryConditionsParams..., solidNodeVelocity = maskedArray);

        append!(schemes, [:movingWalls])
    end

    #= --------------------------- forcing scheme prep --------------------------- =#
    # the default forcing scheme is Guo
    forcingScheme == :default && (forcingScheme = :guo)

    # by defualt, there is no force density
    if forceDensity == :default
        forceDensity = CuArray{Float64}(undef, 0,0);
    # if a single vector is defined it is assumed the force denisty is constant
    elseif forceDensity isa Vector && length(forceDensity) == dims
        forceDensity = fillCuArray(forceDensity |> Vector{Float64}, modelSize)
        forceDensity[vectorBitId(wallRegion)] .= 0
        kwInitialConditions = (; kwInitialConditions..., forceDensity, timeStep)

        append!(schemes, [forcingScheme])
    # if a force density field is defined its dimensions are verified
    elseif size(forceDensity) == size(massDensity)
        forceDensity[vectorBitId(wallRegion)] .= 0
        kwInitialConditions = (; kwInitialConditions..., forceDensity, timeStep)

        append!(schemes, [forcingScheme])
    # if none of the above, the dimensions must be wrong
    else
        error("force density does not have consistent dimensions!")
    end

    #= --------------------------- initial distributions are found --------------------------- =#
    initialDistributions = CuArray{eltype(massDensity)}(undef, (modelSize..., length(velocities.cs))) # Output matrix
    for id in 1:length(velocities.cs)
        initialDistributions[rang(size(initialDistributions), id)...] = findInitialConditions(
            id,
            velocities,
            fluidParams,
            massDensity,
            fluidVelocity,
            latticeSpeed;
            kwInitialConditions = kwInitialConditions
        )
    end

    #= ---------------------------------- saving data setup ---------------------------------- =#
    if saveData
        append!(schemes, [:saveData])
        rm("output.lbm"; force=true, recursive=true)
        mkdir("output.lbm")
    end

    #= ------------------------------ the model is initialized ------------------------------ =#
    model = LBMmodel(
        spaceTime, # space step (Δx), time step (Δt), space coordinate (x), Δt/Δx, dimensionality (dims)
        tick,
        time, # not in spaceTime bc NamedTuple are immutable!
        fluidParams, # speed of sound and its powers (c_s, c2_s, c4_s), relaxation time (τ)
        (; massDensity = massDensity), # ρ₀
        massDensity, # mass density
        fluidVelocity, # fluid velocity
        forceDensity,
        initialDistributions, # f_i(x, t) for all t
        velocities, # c_i for all i
        boundaryConditionsParams, # stream invasion regions and index j such that c[i] = -c[j]
        []|>Vector{LBMparticle}, # initially there will be no particles
        [],
        unique(schemes),
    );

    # to ensure consistency, ρ, ρu and u are all found using the initial conditions of f_i
    hydroVariablesUpdate!(model);

    :saveData in model.schemes && writeTrajectories(model)

    # the model is returned
    return model
end
