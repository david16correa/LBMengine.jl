#= ==========================================================================================
=============================================================================================
all things fluids
=============================================================================================
========================================================================================== =#

function massDensityGet(model::LBMmodel)
    return sum(distribution for distribution in model.distributions)
end
 
function momentumDensityGet(model::LBMmodel; useEquilibriumScheme = false)
    # en estep punto se dará por hecho que la fuerza es constante!!
    bareMomentum = sum(scalarFieldTimesVector(model.distributions[id], model.velocities[id].c) for id in eachindex(model.velocities))

    if useEquilibriumScheme
        if :shan in model.schemes
            return bareMomentum + model.fluidParams.relaxationTime * model.spaceTime.Δt^2 * model.forceDensity
        end
        if :guo in model.schemes
            return bareMomentum + 0.5 * model.spaceTime.Δt * model.forceDensity
        end
    end

    if :shan in model.schemes || :guo in model.schemes
        return bareMomentum + 0.5 * model.spaceTime.Δt * model.forceDensity
    end

    return bareMomentum
end

function hydroVariablesUpdate!(model::LBMmodel; useEquilibriumScheme = false)
    model.massDensity = massDensityGet(model)
    model.momentumDensity = momentumDensityGet(model; useEquilibriumScheme = useEquilibriumScheme)
    model.fluidVelocity = fill([0.; 0], size(model.massDensity))
    fluidIndices = (model.massDensity .≈ 0) .|> b -> !b;
    model.fluidVelocity[fluidIndices] = model.momentumDensity[fluidIndices] ./ model.massDensity[fluidIndices]
end

#= ==========================================================================================
=============================================================================================
lattice Boltzmann dynamics
=============================================================================================
========================================================================================== =#

function equilibriumDistribution(id::Int64, model::LBMmodel; particleId = :default)
    # the quantities to be used are saved separately
    ci = model.velocities[id].c .* model.spaceTime.Δx_Δt
    wi = model.velocities[id].w

    if particleId isa Int64
        u = model.particles[particleId].nodeVelocity
    else
        u = model.fluidVelocity
    end
    # the equilibrium distribution is found step by step and returned
    firstStep = vectorFieldDotVector(u, ci) |> udotci -> udotci/model.fluidParams.c2_s + udotci.^2 / (2 * model.fluidParams.c4_s)
    secondStep = firstStep - vectorFieldDotVectorField(u, u)/(2*model.fluidParams.c2_s)

    model.fluidParams.isFluidCompressible && return wi * ((secondStep .+ 1) .* model.massDensity)

    return wi * model.massDensity + wi * (model.initialConditions.massDensity .* secondStep)
end

function nonEquilibriumDistribution(id::Int64, model::LBMmodel)
    return model.distributions[id] - equilibriumDistribution(id, model)
end

function collisionStep(model::LBMmodel)
    # the equilibrium distributions are found
    equilibriumDistributions = fill([], size(model.velocities)) |> Vector{Array}
    Threads.@threads for id in eachindex(model.velocities)
        equilibriumDistributions[id] = equilibriumDistribution(id, model)
    end

    Omega = fill([], size(model.velocities)) |> Vector{Array}
    if :bgk in model.schemes
        # the Bhatnagar-Gross-Krook collision opeartor is used
        Threads.@threads for id in eachindex(model.velocities)
            Omega[id] = -model.spaceTime.Δt/model.fluidParams.relaxationTime * (model.distributions[id] - equilibriumDistributions[id])
        end
    elseif :trt in model.schemes
        # the two-relaxation-time collision opeartor is used
        distributionsPlus = fill([], size(model.velocities)) |> Vector{Array{Float64}}
        distributionsMinus = copy(distributionsPlus)
        equilibriumDistributionsPlus = copy(distributionsPlus)
        equilibriumDistributionsMinus = copy(distributionsPlus)
        if Threads.nthreads() > 1
            Threads.@threads for id in eachindex(model.velocities)
                conjugateId = model.boundaryConditionsParams.oppositeVectorId[id]

                distributionsPlus[id] = model.distributions[id] + model.distributions[conjugateId] |> M -> M/2
                distributionsMinus[id] = model.distributions[id] - model.distributions[conjugateId] |> M -> M/2

                equilibriumDistributionsPlus[id] = equilibriumDistributions[id] + equilibriumDistributions[conjugateId] |> M -> M/2
                equilibriumDistributionsMinus[id] = equilibriumDistributions[id] - equilibriumDistributions[conjugateId] |> M -> M/2
            end
        else
            # when run serialized, it is quicker to avoid redundant calculations
            for id in eachindex(model.velocities)
                conjugateId = model.boundaryConditionsParams.oppositeVectorId[id]
                if distributionsPlus[id] == []
                    M = model.distributions[id] + model.distributions[conjugateId] |> M -> M/2
                    distributionsPlus[id] = M
                    distributionsPlus[conjugateId] = M

                    M = model.distributions[id] - model.distributions[conjugateId] |> M -> M/2
                    distributionsMinus[id] = M
                    distributionsMinus[conjugateId] = -M

                    M = equilibriumDistributions[id] + equilibriumDistributions[conjugateId] |> M -> M/2
                    equilibriumDistributionsPlus[id] = M
                    equilibriumDistributionsPlus[conjugateId] = M

                    M = equilibriumDistributions[id] - equilibriumDistributions[conjugateId] |> M -> M/2
                    equilibriumDistributionsMinus[id] = M
                    equilibriumDistributionsMinus[conjugateId] = -M
                end
            end
        end

        OmegaPlus = fill([], size(model.velocities)) |> Vector{Array}
        OmegaMinus = fill([], size(model.velocities)) |> Vector{Array}
        Threads.@threads for id in eachindex(model.velocities)
            OmegaPlus[id] = -model.spaceTime.Δt/model.fluidParams.relaxationTimePlus * (distributionsPlus[id] - equilibriumDistributionsPlus[id])
            OmegaMinus[id] = -model.spaceTime.Δt/model.fluidParams.relaxationTimeMinus * (distributionsMinus[id] - equilibriumDistributionsMinus[id])
            Omega[id] = OmegaPlus[id] + OmegaMinus[id]
        end
    end

    if :psm in model.schemes
        if :bgk in model.schemes
            for particle in model.particles
                equilibriumDistributions_solidNodeVelocity = [equilibriumDistribution(id, model; particleId = particle.id) for id in eachindex(model.velocities)]
                E = particle.boundaryConditionsParams.solidRegion # fuzzy edges are still not implemented!
                B = E |> Array
                #= B = model.fluidParams.relaxationTime/model.spaceTime.Δt - 0.5 |> tau_minus_half -> E * tau_minus_half ./ ((1 .- E) .+ tau_minus_half) =#
                OmegaS = [ (model.boundaryConditionsParams.oppositeVectorId[id] |> conjugateId ->
                    (model.distributions[conjugateId] - equilibriumDistributions[conjugateId] - model.distributions[id] + equilibriumDistributions_solidNodeVelocity[id])
                ) for id in eachindex(model.velocities)]
                Omega = [(1 .- B) .* Omega[id] + B .* OmegaS[id] for id in eachindex(model.velocities)]

                # if the solid is coupled to forces or torques, the fluids momentum is transfered to it
                if particle.particleParams.coupleForces || particle.particleParams.coupleTorques
                    for id in eachindex(model.velocities)[2:end]
                        ci = model.velocities[id].c .* model.spaceTime.Δx_Δt
                        sumTerm = B .* OmegaS[id]
                        particle.particleParams.coupleForces && (particle.momentumInput -= model.spaceTime.latticeParameter^model.spaceTime.dims * sum(sumTerm[E]) * ci)
                        particle.particleParams.coupleTorques && (particle.angularMomentumInput -= model.spaceTime.latticeParameter^model.spaceTime.dims * cross(
                            sum(sumTerm[E] .* [x - particle.position for x in model.spaceTime.X[E]]), ci
                        ))
                    end
                end
            end

        elseif :trt in model.schemes
            for particle in model.particles
                equilibriumDistributions_solidNodeVelocity = [equilibriumDistribution(id, model; particleId = particle.id) for id in eachindex(model.velocities)]
                E = particle.boundaryConditionsParams.solidRegion # fuzzy edges are still not implemented!

                equilibriumDistributions_solidNodeVelocityPlus = [[] for _ in eachindex(model.velocities)] |> Vector{Array{Float64}}
                equilibriumDistributions_solidNodeVelocityMinus = [[] for _ in eachindex(model.velocities)] |> Vector{Array{Float64}}

                for id in eachindex(model.velocities)
                    conjugateId = model.boundaryConditionsParams.oppositeVectorId[id]
                    if equilibriumDistributions_solidNodeVelocityPlus[id] == []
                        M = equilibriumDistributions_solidNodeVelocity[id] + equilibriumDistributions_solidNodeVelocity[conjugateId] |> M -> M/2
                        equilibriumDistributions_solidNodeVelocityPlus[id] = M
                        equilibriumDistributions_solidNodeVelocityPlus[conjugateId] = M

                        M = equilibriumDistributions_solidNodeVelocity[id] - equilibriumDistributions_solidNodeVelocity[conjugateId] |> M -> M/2
                        equilibriumDistributions_solidNodeVelocityMinus[id] = M
                        equilibriumDistributions_solidNodeVelocityMinus[conjugateId] = -M
                    end
                end

                Bplus = E |> Array
                Bminus = E |> Array
                #= B = model.fluidParams.relaxationTime/model.spaceTime.Δt - 0.5 |> tau_minus_half -> E * tau_minus_half ./ ((1 .- E) .+ tau_minus_half) =#
                Omega_fluid = [Omega[id] - Bplus .* OmegaPlus[id] - Bminus .* OmegaMinus[id] for id in eachindex(model.velocities)]

                Omega_solid = [
                    - Bplus .* (equilibriumDistributionsPlus[id] - equilibriumDistributions_solidNodeVelocityPlus[id]) -
                    Bminus .* (2 * distributionsMinus[id] - equilibriumDistributionsMinus[id] - equilibriumDistributions_solidNodeVelocityMinus[id])
                for id in eachindex(model.velocities)]

                Omega = [Omega_fluid[id] + Omega_solid[id] for id in eachindex(model.velocities)]

                # if the solid is coupled to forces or torques, the fluids momentum is transfered to it
                if particle.particleParams.coupleForces || particle.particleParams.coupleTorques
                    for id in eachindex(model.velocities)[2:end]
                        ci = model.velocities[id].c .* model.spaceTime.Δx_Δt
                        sumTerm = Omega_solid[id]
                        particle.particleParams.coupleForces && (particle.momentumInput -= model.spaceTime.latticeParameter^model.spaceTime.dims * sum(sumTerm[E]) * ci)
                        particle.particleParams.coupleTorques && (particle.angularMomentumInput -= model.spaceTime.latticeParameter^model.spaceTime.dims * cross(
                            sum(sumTerm[E] .* [x - particle.position for x in model.spaceTime.X[E]]), ci
                        ))
                    end
                end
            end
        end
    end
    # forcing terms are added
    if :guo in model.schemes
        Threads.@threads for id in eachindex(model.velocities)
            Omega[id] = Omega[id] + guoForcingTerm(id, model)
        end
    end

    output = Omega # I know it's useless
    Threads.@threads for id in eachindex(model.velocities)
        output[id] += model.distributions[id]
    end
    return output
end

function guoForcingTerm(id::Int64, model::LBMmodel)
    # the quantities to be used are saved separately
    ci = model.velocities[id].c .* model.spaceTime.Δx_Δt
    c2_s = model.fluidParams.c2_s
    wi = model.velocities[id].w
    U = model.fluidVelocity
    F = model.forceDensity
    Δt = model.spaceTime.Δt

    # the forcing term is found (terms common for both collision methods are found first)
    secondTerm = vectorFieldDotVector(U, ci) |> udotci -> [ci * v for v in udotci]/model.fluidParams.c4_s
    if :bgk in model.schemes
        firstTerm = [ci - u for u in U] / c2_s
        intermediateStep = vectorFieldDotVectorField(firstTerm + secondTerm, F)
        return wi * Δt * (1 - Δt/(2 * model.fluidParams.relaxationTime)) * intermediateStep
    elseif :trt in model.schemes
        intermediateStep = vectorFieldDotVectorField(-U/c2_s + secondTerm, F)
        plusPart = (1 - Δt/(2 * model.fluidParams.relaxationTimePlus)) * intermediateStep

        intermediateStep = vectorFieldDotVector(F, ci/c2_s)
        minusPart = (1 - Δt/(2 * model.fluidParams.relaxationTimeMinus)) * intermediateStep

        return wi * Δt * (plusPart + minusPart)
    end
end

function cbcBoundaries!(model::LBMmodel)
    # the macroscopic variables are bundled and their derivatives found
    M = [[model.massDensity[id]; model.fluidVelocity[id][1]; model.fluidVelocity[id][2]] for id in eachindex(IndexCartesian(), model.massDensity)];
    dxM = boundaryDerivative(M, model; targetDim = 1, h = model.spaceTime.latticeParameter);
    dyM = boundaryDerivative(M, model; targetDim = 2, h = model.spaceTime.latticeParameter);

    # some auxiliary variables are saved
    sizeM = size(M)

    # Heubes et al. found the choice of γ = 3/4 to be superior
    gamma = 3/4;

    # some fluid variables are declared locally for readibility
    c_s = copy(model.fluidParams.c_s);
    c2_s = copy(model.fluidParams.c2_s);
    rho = copy(model.massDensity);
    u = copy(model.fluidVelocity);

    # all auxiliary variables needed for the method are initialized
    xMat = fill(zeros(1,1), sizeM)
    yMat = copy(xMat)
    pX = copy(xMat)
    pXinv = copy(xMat)
    pY = copy(xMat)
    pYinv = copy(xMat)
    lambdaX = fill(zeros(1), sizeM)
    lambdaY = copy(lambdaX)

    # if absorbent walls are used around the x axis
    if 1 in model.boundaryConditionsParams.walledDimensions
        Lx = fill([], sizeM)
        # auxiliary variables are found
        Threads.@threads for id in eachindex(IndexCartesian(), M)|>collect|>m->m[[1;end],:]
            @inbounds xMat[id] = [u[id][1] rho[id] 0; c2_s/rho[id] u[id][1] 0; 0 0 u[id][1]]
            @inbounds yMat[id] = [u[id][2] 0 rho[id]; 0 u[id][2] 0; c2_s/rho[id] 0 u[id][2]]
            @inbounds pX[id] = [c2_s -c_s*rho[id] 0; 0 0 1; c2_s c_s*rho[id] 0]
            @inbounds pXinv[id] = [0.5/c2_s 0 0.5/c2_s; -0.5/(rho[id]*c_s) 0 0.5/(rho[id]*c_s); 0 1 0]
            @inbounds pY[id] = pX[id][:, [1;3;2]]
            @inbounds pYinv[id] = pXinv[id][[1;3;2], :]
            @inbounds lambdaX[id] = [u[id][1]-c_s, u[id][1], u[id][1]+c_s]
            @inbounds lambdaY[id] = [u[id][2]-c_s, u[id][2], u[id][2]+c_s]

            @inbounds Lx[id] = [(M[id][2] - c_s) * (c2_s*dxM[id][1] - c_s*M[id][1]*dxM[id][2]);
                M[id][2] * dxM[id][3];
                (M[id][2] + c_s) * (c2_s*dxM[id][1] + c_s*M[id][1]*dxM[id][2])]
        end
        # left wall
        for id in eachindex(IndexCartesian(), Lx)|>collect|>M->M[1,:], jd in eachindex(Lx[1])
            lambdaX[id][jd] > 0 && @inbounds Lx[id][jd] = 0.
        end
        # right wall
        for id in eachindex(IndexCartesian(), Lx)|>collect|>M->M[end,:], jd in eachindex(Lx[1])
            lambdaX[id][jd] < 0 && @inbounds Lx[id][jd] = 0.
        end
    end

    # if absorbent walls are used around the y axis
    if 2 in model.boundaryConditionsParams.walledDimensions
        Ly = fill([], sizeM)
        # auxiliary variables are found
        Threads.@threads for id in eachindex(IndexCartesian(), M)|>collect|>m->m[:,[1;end]]
            @inbounds xMat[id] = [u[id][1] rho[id] 0; c2_s/rho[id] u[id][1] 0; 0 0 u[id][1]]
            @inbounds yMat[id] = [u[id][2] 0 rho[id]; 0 u[id][2] 0; c2_s/rho[id] 0 u[id][2]]
            @inbounds pX[id] = [c2_s -c_s*rho[id] 0; 0 0 1; c2_s c_s*rho[id] 0]
            @inbounds pXinv[id] = [0.5/c2_s 0 0.5/c2_s; -0.5/(rho[id]*c_s) 0 0.5/(rho[id]*c_s); 0 1 0]
            @inbounds pY[id] = pX[id][:, [1;3;2]]
            @inbounds pYinv[id] = pXinv[id][[1;3;2], :]
            @inbounds lambdaX[id] = [u[id][1]-c_s, u[id][1], u[id][1]+c_s]
            @inbounds lambdaY[id] = [u[id][2]-c_s, u[id][2], u[id][2]+c_s]

            @inbounds Ly[id] = [(M[id][3] - c_s) * (c2_s*dyM[id][1] - c_s*M[id][1]*dyM[id][3]);
                M[id][3] * dyM[id][2];
                (M[id][3] + c_s) * (c2_s*dyM[id][1] + c_s*M[id][1]*dyM[id][3])]
        end
        # bottom wall
        for id in eachindex(IndexCartesian(), Ly)|>collect|>M->M[:,1], jd in eachindex(Ly[1])
            lambdaY[id][jd] > 0 && @inbounds Ly[id][jd] = 0.
        end
        # top wall
        for id in eachindex(IndexCartesian(), Ly)|>collect|>M->M[:,end], jd in eachindex(Ly[1])
            lambdaY[id][jd] < 0 && @inbounds Ly[id][jd] = 0.
        end
    end

    dtM = fill(zero(M[1]), sizeM);

    # x boundaries
    if 1 in model.boundaryConditionsParams.walledDimensions
        Threads.@threads for id in eachindex(IndexCartesian(), M)|>collect|>m->m[[1;end],:]
            @inbounds dtM[id] = -pXinv[id]*Lx[id] - gamma * yMat[id] * dyM[id]
        end
    end
    # y boundaries
    if 2 in model.boundaryConditionsParams.walledDimensions
        Threads.@threads for id in eachindex(IndexCartesian(), M)|>collect|>m->m[:,[1;end]]
            @inbounds dtM[id] = -pYinv[id]*Ly[id] - gamma * xMat[id] * dxM[id]
        end
    end
    # corners
    corners = fill([], 2);
    corners[1] = [1;sizeM[1]];
    corners[2] = [1;sizeM[2]];
    if all(dim -> dim in model.boundaryConditionsParams.walledDimensions, [1;2])
        Threads.@threads for id in eachindex(IndexCartesian(), M)|>collect|>m->m[corners...]
            @inbounds dtM[id] = -pXinv[id]*Lx[id] - pYinv[id]*Ly[id]
        end
    end

    # macroscopic variables are updated, and used to find new distributions for LBM
    M += model.spaceTime.Δt*dtM
    model.massDensity = [m[1] for m in M]
    model.fluidVelocity = [m[2:3] for m in M]
    equilibriumDistributions = [equilibriumDistribution(id, model) for id in eachindex(model.velocities)]

    for id in eachindex(model.velocities)
        # x boundaries
        if 1 in model.boundaryConditionsParams.walledDimensions
            model.distributions[id][1,:] = equilibriumDistributions[id][1,:]
            model.distributions[id][end,:] = equilibriumDistributions[id][end,:]
        end
        # y boundaries
        if 2 in model.boundaryConditionsParams.walledDimensions
            model.distributions[id][:,1] = equilibriumDistributions[id][:,1]
            model.distributions[id][:,end] = equilibriumDistributions[id][:,end]
        end
    end
end

#= ==========================================================================================
=============================================================================================
time evolution
=============================================================================================
========================================================================================== =#

function tick!(model::LBMmodel)


    # collision (or relaxation)
    collisionedDistributions = collisionStep(model)

    # propagated distributions will be saved in a new vector
    #= propagatedDistributions = [] |> LBMdistributions ; =#
    propagatedDistributions = fill([], length(model.velocities)) |> LBMdistributions

    # streaming (or propagation), with streaming invasion exchange
    Threads.@threads for id in eachindex(model.velocities)
        # distributions are initially streamed
        streamedDistribution = pbcMatrixShift(collisionedDistributions[id], model.velocities[id].c)

        if :bounceBack in model.schemes
            # the wall region, the invasion region of the fluid, and the conjugate of the fluid with opposite momentum are retrieved
            wallRegion = model.boundaryConditionsParams.wallRegion
            conjugateInvasionRegion, conjugateId = model.boundaryConditionsParams |> params -> (params.streamingInvasionRegions[params.oppositeVectorId[id]], params.oppositeVectorId[id])

            # the wall regions are imposed to vanish
            streamedDistribution[wallRegion] .= 0;

            # the boudnary nodes of all rigid moving particles are considered in the streaming invasion exchange step
            if :ladd in model.schemes
                for particle in model.particles
                    wallRegion = wallRegion .|| particle.boundaryConditionsParams.solidRegion
                    conjugateInvasionRegion = conjugateInvasionRegion .|| particle.boundaryConditionsParams.streamingInvasionRegions[model.boundaryConditionsParams.oppositeVectorId[id]]
                end
            end

            # streaming invasion exchange step is performed
            streamedDistribution[conjugateInvasionRegion] = collisionedDistributions[conjugateId][conjugateInvasionRegion]

            # if any wall is moving, its momentum is transfered to the fluid
            if :movingWalls in model.schemes
                ci = model.velocities[id].c .* model.spaceTime.Δx_Δt
                wi = model.velocities[id].w

                uwdotci = pbcMatrixShift(model.boundaryConditionsParams.solidNodeVelocity, model.velocities[id].c) |> uw -> vectorFieldDotVector(uw,ci)
                streamedDistribution[conjugateInvasionRegion] += (2 * wi / model.fluidParams.c2_s) * model.massDensity[conjugateInvasionRegion] .* uwdotci[conjugateInvasionRegion]
            end

            # if any particle is moving, its momentum is exchanged with the fluid
            if :ladd in model.schemes
                ci = model.velocities[id].c .* model.spaceTime.Δx_Δt
                wi = model.velocities[id].w
                for particle in model.particles
                    conjugateBoundaryNodes = particle.boundaryConditionsParams.streamingInvasionRegions[model.boundaryConditionsParams.oppositeVectorId[id]]

                    # if the fluid did not bump into the particle, then the entire scheme can be skipped
                    sum(conjugateBoundaryNodes) == 0 && break

                    # the solids momentum is transfered to the fluid
                    uw = particle.nodeVelocity + pbcMatrixShift(particle.nodeVelocity, model.velocities[id].c)
                    uwdotci = vectorFieldDotVector(uw,ci)
                    streamedDistribution[conjugateBoundaryNodes] += (2 * wi / model.fluidParams.c2_s) * model.massDensity[conjugateBoundaryNodes] .* uwdotci[conjugateBoundaryNodes]

                    # if the solid is coupled to forces or torques, the fluids momentum is transfered to it
                    if particle.particleParams.coupleForces || particle.particleParams.coupleTorques
                        sumTerm = collisionedDistributions[conjugateId][conjugateBoundaryNodes] + streamedDistribution[conjugateBoundaryNodes]
                        particle.particleParams.coupleForces && (particle.momentumInput -= model.spaceTime.latticeParameter^model.spaceTime.dims * sum(sumTerm) * ci)
                        particle.particleParams.coupleTorques && (particle.angularMomentumInput -= model.spaceTime.latticeParameter^model.spaceTime.dims * cross(
                            sum(sumTerm .* [x - particle.position for x in model.spaceTime.X[conjugateBoundaryNodes]]), ci
                        ))
                    end
                end
            end
        end

        # the resulting propagation is appended to the propagated distributions
        #= append!(propagatedDistributions, [streamedDistribution]); =#
        propagatedDistributions[id] = streamedDistribution
    end

    # the distributions and time are updated
    model.distributions = propagatedDistributions
    model.time += model.spaceTime.Δt
    model.tick += 1

    # characteristic boundary conditions
    :cbc in model.schemes && cbcBoundaries!(model)

    # Finally, the hydrodynamic variables are updated
    hydroVariablesUpdate!(model; useEquilibriumScheme = true);
    # and particles are moved, if there are any
    (:ladd in model.schemes || :psm in model.schemes) && (moveParticles!(model));
end

function LBMpropagate!(model::LBMmodel; simulationTime = :default, ticks = :default, verbose = false, ticksBetweenSaves = 10)

    println("Thrads = $(Threads.nthreads())")

    if simulationTime != :default && ticks != :default
        error("simulationTime and ticks cannot be simultaneously chosen, as the time step is defined already in the model!")
    elseif simulationTime == :default && ticks == :default
        time = range(model.spaceTime.Δt, length = 100, step = model.spaceTime.Δt);
    elseif ticks == :default
        time = range(model.spaceTime.Δt, stop = simulationTime::Number, step = model.spaceTime.Δt);
    else
        time = range(model.spaceTime.Δt, length = ticks::Int64, step = model.spaceTime.Δt);
    end

    verbose && (outputTimes = range(1, stop = length(time), length = 50) |> collect .|> round)

    :saveData in model.schemes && mkOutputDirs()

    for t in time |> eachindex
        tick!(model);
        verbose && t in outputTimes && (print("\r t = $(model.time)"); flush(stdout))
        :saveData in model.schemes && (model.tick % ticksBetweenSaves == 0) && writeTrajectories(model)
    end
    print("\r");

    hydroVariablesUpdate!(model);
    :saveData in model.schemes && writeTrajectories(model)

    return nothing
end

#= ==========================================================================================
=============================================================================================
rheology
=============================================================================================
========================================================================================== =#

function viscousStressTensor(model)
    # the non-equilibrium distributions are found
    nonEquilibriumDistributions = [nonEquilibriumDistribution(id, model) for id in eachindex(model.velocities)]

    # the (1 - Δt/2τ) factor is found beforehand
    coeff = 1 - 0.5*model.spaceTime.Δt/model.fluidParams.relaxationTime

    # the viscous stress tensor is returned
    return [
        -coeff * sum(model.velocities[id].c[alpha] * model.velocities[id].c[beta] * nonEquilibriumDistributions[id] for id in eachindex(model.velocities))
    for alpha in 1:model.spaceTime.dims, beta in 1:model.spaceTime.dims]
end
