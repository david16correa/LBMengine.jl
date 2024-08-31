#= ==========================================================================================
=============================================================================================
init methods
=============================================================================================
========================================================================================== =#

function addBead!(model::LBMmodel;
    mass = 1.,
    radio = 0.1,
    position = :default, # origin (dimensionality dependent)
    velocity = :default, # static (dimensionality dependent)
    angularVelocity = :default, # static, (dimensionality dependent)
)
    beadGeometry(x::Vector{Float64}; radio = 0.25) = sum(x.^2) <= radio^2

    if model.spaceTime.dims == 2
        momentOfInertia = 0.5 * mass * radio^2 # moment of inertia for a disk
        position == :default && (position = [0., 0])
        solidRegion = [beadGeometry([i,j] - position; radio = radio) for i in model.spaceTime.x, j in model.spaceTime.x] |> sparse
        velocity == :default && (velocity = [0., 0])
        angularVelocity == :default && (angularVelocity = 0.)
        nodeVelocity = [[0.,0]]
        momentumInput = [0., 0]
        angularMomentumInput = 0.
    elseif model.spaceTime.dims == 3
        momentOfInertia = 0.4 * mass * radio^2 # moment of inertia for a sphere
        position == :default && (position = [0., 0, 0])
        solidRegion = [beadGeometry([i,j,k] - position; radio = radio) for i in model.spaceTime.x, j in model.spaceTime.x, k in model.spaceTime.x]
        velocity == :default && (velocity = [0., 0, 0])
        angularVelocity == :default && (angularVelocity = [0., 0, 0])
        nodeVelocity = [[0.,0, 0]]
        momentumInput = [0., 0, 0]
        angularMomentumInput = [0., 0, 0]
    else
        error("dimensionality must be either 2 or 3! dims = $(model.spaceTime.dims)")
    end

    streamingInvasionRegions = bounceBackPrep(solidRegion, model.velocities; returnStreamingInvasionRegions = true)

    newBead = LBMparticle(
        (;inverseMass = 1/mass, inverseMomentOfInertia = 1/momentOfInertia, solidRegionGenerator = x -> beadGeometry(x; radio = radio), symmetries = [:spherical]),
        (; solidRegion, streamingInvasionRegions),
        position,
        velocity,
        angularVelocity,
        nodeVelocity,
        momentumInput,
        angularMomentumInput,
    )
    append!(model.particles, [newBead]);
    append!(model.schemes, [:ladd])
    model.schemes = model.schemes |> unique
    if !(:bounceBack in model.schemes)
        wallRegion = [false for _ in solidRegion] |> sparse
        streamingInvasionRegions, oppositeVectorId = bounceBackPrep(wallRegion, model.velocities);
        model.boundaryConditionsParams = merge(model.boundaryConditionsParams, (; wallRegion, streamingInvasionRegions, oppositeVectorId));
        append!(model.schemes, [:bounceBack])
    end
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
    relaxationTimeRatio = 0.8, # τ/Δt > 1 → under-relaxation, τ/Δt = 1 → full relaxation, 0.5 < τ/Δt < 1 → over-relaxation, τ/Δt < 0.5 → unstable
    x = range(0, stop = 1, step = 0.01),
    dims = 2, # default mode must be added!!
    Δt = :default, # default: Δt = Δx
    walledDimensions = :default, # walls around y axis (all non-walled dimensions are periodic!)
    solidNodes = :default, # default: no solid nodes (other than the walls) 
    solidNodeVelocity = :default, # default: static solids - u = [0,0]
    isFluidCompressible = false,
    forceDensity = :default, # default: F(0) = 0
    forcingScheme = :default # {:guo, :shan}, default: Guo, C. Zheng, B. Shi, Phys. Rev. E 65, 46308 (2002)
)
    # the list of schemes is initialized
    schemes = [] |> Vector{Symbol}
    # the keywords for the initial conditions are initialized
    kwInitialConditions = (; )
    boundaryConditionsParams = (; )

    # if default conditions were chosen, ρ is built. Otherwise its dimensions are verified
    if massDensity == :default
        massDensity = [length(x) for _ in 1:dims] |> v -> ones(v...)
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
    spaceTime = (; x, Δx, Δt, Δx_Δt, dims); 
    time = [0.];

    #= -------------------- fluid parameters are initialized -------------------- =#
    c_s, c2_s, c4_s = Δx_Δt/√3, Δx_Δt^2 / 3, Δx_Δt^4 / 9;
    relaxationTime = Δt * relaxationTimeRatio;
    fluidParams = (; c_s, c2_s, c4_s, relaxationTime, isFluidCompressible);

    #= -------------------- boundary conditions (bounce back) -------------------- =#
    wallRegion = [false for _ in massDensity]
    if walledDimensions != :default
        wallRegion = wallNodes(massDensity; walledDimensions = walledDimensions); 

        append!(schemes, [:bounceBack])
    end
    if solidNodes != :default && size(solidNodes) == size(wallRegion)
        wallRegion = wallRegion .|| solidNodes

        append!(schemes, [:bounceBack])
    end
    dims <= 2 && (wallRegion = sparse(wallRegion))

    if :bounceBack in schemes
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

    #= ------------------------- the model is initialized ------------------------ =#
    model = LBMmodel(
        spaceTime, # space step (Δx), time step (Δt), space coordinate (x), Δt/Δx, dimensionality (dims)
        time, # not in spaceTime bc NamedTuple are immutable!
        fluidParams, # speed of sound and its powers (c_s, c2_s, c4_s), relaxation time (τ)
        (; massDensity = massDensity), # ρ₀
        massDensity, # mass density
        massDensity.*fluidVelocity, # momentum density
        fluidVelocity, # fluid velocity
        forceDensity,
        [initialDistributions], # f_i(x, t) for all t
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

    # the model is returned
    return model
end
