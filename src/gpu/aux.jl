#= ==========================================================================================
=============================================================================================
GPU accelerated scalar and vector field arithmetic auxiliary functions
=============================================================================================
========================================================================================== =#

# model.distributions will be an array with dims+1 dimensions, where dims is the dimensionality
# of the problem. rang() is written to isolate a single distributions
rang(model::LBMmodel, id) = (((1:s for s in size(model.distributions)[1:end-1]))..., id)
rang(sizeM::Tuple, id) = (((1:s for s in sizeM[1:end-1]))..., id)

function getThreadsAndBlocks(dims, blockSizes)
    if dims == 2
        threads = (16, 16)  # 16x16 thread block
        blocks = (cld(blockSizes[1], threads[1]), cld(blockSizes[2], threads[2]))
    elseif dims == 3
        threads = (8,8,8)  # 8x8x8 thread block
        blocks = (cld(blockSizes[1], threads[1]), cld(blockSizes[2], threads[2]), cld(blockSizes[3], threads[3]))
    end
    return threads, blocks
end

function fillCuArray(val::Number, rowsCols::Tuple; order = 1)
    if order == 1
        dims = length(rowsCols)
    else
        rowsCols = (rowsCols..., order)
        dims = length(rowsCols) - 1
    end
    output = CuArray{typeof(val)}(undef, rowsCols) # Output matrix
    threads, blocks = getThreadsAndBlocks(dims, rowsCols)
    #
    kernel = [fillCuArray2D_kernel, fillCuArray3D_kernel][dims-1]
    @cuda threads=threads blocks=blocks kernel(output, val, rowsCols)
    return output
end
function fillCuArray(val::Array, rowsCols::Tuple)
    rowsCols = (rowsCols..., length(val))
    dims = length(rowsCols) - 1
    output = CuArray{eltype(val)}(undef, rowsCols) # Output matrix
    threads, blocks = getThreadsAndBlocks(dims, rowsCols)
    #
    kernel = [fillCuArray2D_kernel, fillCuArray3D_kernel][dims-1]
    @cuda threads=threads blocks=blocks kernel(output, val|>CuArray{Float64}, rowsCols)
    return output
end

function coordinatesCuArray(coordinates::Tuple)
    rowsCols = length.(coordinates)
    dims = length(coordinates)
    output = CuArray{eltype(coordinates[1])}(undef, (rowsCols..., dims)) # Output matrix
    threads, blocks = getThreadsAndBlocks(dims, rowsCols)

    kernel = [coordinatesCuArray2D_kernel, coordinatesCuArray3D_kernel][dims-1]
    @cuda threads=threads blocks=blocks kernel(output, coordinates, rowsCols, dims)
    return output
end

function vectorBitId(bitId)
    rowsCols = size(bitId)
    dims = length(rowsCols)
    output = CuArray{Bool}(undef, (rowsCols..., dims)) # Output matrix
    threads, blocks = getThreadsAndBlocks(dims, rowsCols)
    #
    kernel = [vectorBitId2D_kernel, vectorBitId3D_kernel][dims-1]
    @cuda threads=threads blocks=blocks kernel(output, bitId, rowsCols, dims)

    return output
end

function scalarFieldTimesVector(scalarField, vector)
    N = length(vector)
    rowsCols = scalarField |> size
    dims = rowsCols |> length
    output = CuArray{eltype(scalarField)}(undef, (rowsCols..., dims)) # Output matrix
    threads, blocks = getThreadsAndBlocks(dims, rowsCols)
    #
    kernel = [scalarFieldTimesVector2D_kernel, scalarFieldTimesVector3D_kernel][dims-1]
    @cuda threads=threads blocks=blocks kernel(output, scalarField, vector, N, rowsCols)
    return output
end

function scalarFieldTimesVectorField(scalarField, vectorField)
    rowsCols = vectorField |> size
    dims = rowsCols |> length |> x -> x - 1 # for a vector field, the array will have one more dimension that the problem!
    output = CuArray{eltype(scalarField)}(undef, rowsCols) # Output matrix
    threads, blocks = getThreadsAndBlocks(dims, rowsCols)
    #
    kernel = [scalarFieldTimesVectorField2D_kernel, scalarFieldTimesVectorField2D_kernel][dims-1]
    @cuda threads=threads blocks=blocks kernel(output, scalarField, vectorField, rowsCols)
    return output
end

function vectorFieldPlusVector(vectorField, vector)
    rowsCols = vectorField |> size
    dims = vector |> length
    output = copy(vectorField)
    threads, blocks = getThreadsAndBlocks(dims, rowsCols)
    #
    kernel = [vectorFieldPlusVector2D_kernel, vectorFieldPlusVector3D_kernel][dims-1]
    @cuda threads=threads blocks=blocks kernel(output, vector, rowsCols)
    return output
end

function vectorFieldDotVector(vectorField, vector)
    rowsCols = size(vectorField)
    dims = length(vector)
    output = CuArray{eltype(vectorField)}(undef, rowsCols[1:end-1])
    threads, blocks = getThreadsAndBlocks(dims, rowsCols)
    #
    kernel = [vectorFieldDotVector2D_kernel, vectorFieldDotVector3D_kernel][dims-1]
    @cuda threads=threads blocks=blocks kernel(output, vectorField, vector, rowsCols)
    return output
end

function vectorFieldDotVectorField(vField, wField)
    rowsCols = vField |> size
    dims = rowsCols |> length |> x -> x - 1 # for a vector field, the array will have one more dimension that the problem!
    output = CuArray{eltype(vField)}(undef, size(vField)[1:end-1])
    threads, blocks = getThreadsAndBlocks(dims, rowsCols)
    #
    kernel = [vectorFieldDotVectorField2D_kernel, vectorFieldDotVectorField3D_kernel][dims-1]
    @cuda threads=threads blocks=blocks kernel(output, vField, wField, rowsCols)
    return output
end

function circshift_gpu(input::CuArray, shift::CuArray)
    rowsCols = input |> size
    dims = shift |> length

    output = similar(input)
    threads, blocks = getThreadsAndBlocks(dims, rowsCols)

    kernel = [circshift2D_kernel, circshift3D_kernel][dims-1]
    @cuda threads=threads blocks=blocks kernel(output, input, shift, rowsCols)
    return output
end

function cross(v_d::AbstractArray, w_d::AbstractArray)
    v = v_d |> Array
    w = w_d |> Array
    dim = length(v) # dimension consistency is not checked bc I trust I won't mess things up
    dim == 2 && (return v[1]*w[2] - v[2]*w[1]) # in two dimensions, vectors are assumed to have zero z component, and only the z component is returned
    dim == 3 && (return [v[2]*w[3] - v[3]*w[2], -v[1]*w[3] + v[3]*w[1], v[1]*w[2] - v[2]*w[1]] |> typeof(v_d))
end

#= I know this method is highly questionable, but it was born out of the need to compute the tangential
velocity using the position vector and the angular velocity in two dimensions. Ω happens to be a scalar
in two dimensions, but momentarily using three dimensions results in a simpler algorithm. =#
function cross(omega::Real, V_d::AbstractArray)
    V = V_d |> Array
    return [-omega * V[2], omega * V[1]] |> typeof(V_d)
end

#= ==========================================================================================
=============================================================================================
LBM@gpu aux
=============================================================================================
========================================================================================== =#

function findFluidVelocity(massDensity, momentumDensity)
    rowsCols = size(momentumDensity)
    dims = rowsCols |> length |> x -> x - 1 # for a vector field, the array will have one more dimension that the problem!
    output = CuArray{eltype(momentumDensity)}(undef, rowsCols) # Output matrix
    #
    threads, blocks = getThreadsAndBlocks(dims, rowsCols)
    #
    kernel = [findFluidVelocity2D_kernel, findFluidVelocity3D_kernel][dims-1]
    @cuda threads=threads blocks=blocks kernel(output, massDensity, momentumDensity, rowsCols)
    return output
end

#= ==========================================================================================
=============================================================================================
bounce-back boundary conditions
=============================================================================================
========================================================================================== =#

function bounceBackPrep(wallRegion::CuArray{Bool}, velocities::NamedTuple; returnStreamingInvasionRegions = false)
    cs = velocities.cs;

    streamingInvasionRegions = [(circshift_gpu(wallRegion, -c) .|| wallRegion) .⊻ wallRegion for c in cs]
    streamingInvasionRegions = CuArray(cat(streamingInvasionRegions...; dims=length(cs[1])+1));

    returnStreamingInvasionRegions && return streamingInvasionRegions

    oppositeVectorId = [findfirst(x -> x == -1 .* c, cs) for c in cs]

    return streamingInvasionRegions, oppositeVectorId
end

#= ==========================================================================================
=============================================================================================
particles@gpu aux
=============================================================================================
========================================================================================== =#

function getSphere(radius::Number, R::CuArray)
    rowsCols, dims = size(R) |> sizeR -> (sizeR, sizeR |> length |> x -> x - 1) # for a vector field, the array will have one more dimension that the problem!
    output = CuArray{Bool}(undef, rowsCols[1:dims]) # Output matrix
    #
    threads, blocks = getThreadsAndBlocks(dims, rowsCols)
    #
    kernel = [getSphere2D_kernel, getSphere3D_kernel][dims-1]
    @cuda threads=threads blocks=blocks kernel(output, radius^2, R, rowsCols)
    return output
end

function getSolidNodes(solidRegionGenerator::Function, R::CuArray)
    rowsCols, dims = size(R) |> sizeR -> (sizeR, sizeR |> length |> x -> x - 1) # for a vector field, the array will have one more dimension that the problem!
    output = CuArray{Bool}(undef, rowsCols[1:dims]) # Output matrix
    #
    threads, blocks = getThreadsAndBlocks(dims, rowsCols)
    #
    kernel = [getSolidNodes2D_kernel, getSolidNodes3D_kernel][dims-1]
    @cuda threads=threads blocks=blocks kernel(output, solidRegionGenerator, R, rowsCols)
    return output
end

# for ladd only the shell around the solid is needed; surely I can find a way to skip the unnecessary nodes and speedup things
function getNodeVelocity(model::LBMmodel; id = 1)
    xMinusR = vectorFieldPlusVector(model.spaceTime.X, -model.particles[id].position)
    xMinusR_norm = vectorFieldDotVectorField(xMinusR, xMinusR) .|> sqrt
    xMinusR_norm .= ifelse.(xMinusR_norm .== 0, 1, xMinusR_norm)


    rowsCols, dims = size(xMinusR_norm) |> sizeM -> (sizeM, length(sizeM)) # for a vector field, the array will have one more dimension that the problem!
    bulkV = CuArray{eltype(model.fluidVelocity)}(undef, (rowsCols..., dims)) # Output matrix
    #
    threads, blocks = getThreadsAndBlocks(dims, rowsCols)
    #
    kernel = [getBulkV2D_kernel, getBulkV3D_kernel][dims-1]
    @cuda threads=threads blocks=blocks kernel(
        bulkV,
        model.particles[id].velocity,
        model.particles[id].angularVelocity,
        xMinusR,
        xMinusR_norm,
        model.particles[id].particleParams.radius,
        model.spaceTime.latticeParameter,
        rowsCols
    )

    (:bead in model.particles[id].particleParams.properties) && (return bulkV)

    fancyR = xMinusR ./ xMinusR_norm
    e_dot_fancyR = vectorFieldDotVector(fancyR, model.particles[id].particleParams.swimmingDirection)

    firstTerm = model.particles[id].particleParams.B1 .+ model.particles[id].particleParams.B2 * e_dot_fancyR
    secondTerm = scalarFieldTimesVectorField(e_dot_fancyR, fancyR) |> vF -> vectorFieldPlusVector(vF, -model.particles[id].particleParams.swimmingDirection)

    return scalarFieldTimesVectorField(firstTerm, secondTerm) + bulkV
end

#= ==========================================================================================
=============================================================================================
saving data
=============================================================================================
========================================================================================== =#

function writeTrajectories(model::LBMmodel)
    if model.spaceTime.dims == 2
        fluidDf = DataFrame(
            tick = model.tick,
            time = model.time,
            id_x = [coordinate[1] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            id_y = [coordinate[2] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            coordinate_x = [coordinate[1] for coordinate in model.spaceTime.X] |> vec,
            coordinate_y = [coordinate[2] for coordinate in model.spaceTime.X] |> vec,
            massDensity = model.massDensity |> vec,
            fluidVelocity_x = [velocity[1] for velocity in model.fluidVelocity] |> vec,
            fluidVelocity_y = [velocity[2] for velocity in model.fluidVelocity] |> vec
        ) # keyword argument constructor
    elseif model.spaceTime.dims == 3
        fluidDf = DataFrame(
            tick = model.tick,
            time = model.time,
            id_x = [coordinate[1] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            id_y = [coordinate[2] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            id_z = [coordinate[3] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            coordinate_x = [coordinate[1] for coordinate in model.spaceTime.X] |> vec,
            coordinate_y = [coordinate[2] for coordinate in model.spaceTime.X] |> vec,
            coordinate_z = [coordinate[3] for coordinate in model.spaceTime.X] |> vec,
            massDensity = model.massDensity |> vec,
            fluidVelocity_x = [velocity[1] for velocity in model.fluidVelocity] |> vec,
            fluidVelocity_y = [velocity[2] for velocity in model.fluidVelocity] |> vec,
            fluidVelocity_z = [velocity[3] for velocity in model.fluidVelocity] |> vec
        ) # keyword argument constructor
    end
    distributionsDf = DataFrame(
        vec.(model.distributions), ["f$(i)" for i in 1:length(model.distributions)]
    ) # vector of vectors constructor

    CSV.write("output.lbm/fluidTrj_$(model.tick).csv", [fluidDf distributionsDf])
end

function writeParticleTrajectory(particle::LBMparticle, model::LBMmodel)
    if model.spaceTime.dims == 2
        particleDf = DataFrame(
            tick = model.tick,
            time = model.time,
            particleId = particle.id,
            position_x = particle.position[1],
            position_y = particle.position[2],
            velocity_x = particle.velocity[1],
            velocity_y = particle.velocity[2],
            angularVelocity = particle.angularVelocity
        )
    elseif model.spaceTime.dims == 3
        particleDf = DataFrame(
            tick = model.tick,
            time = model.time,
            particleId = particle.id,
            position_x = particle.position[1],
            position_y = particle.position[2],
            position_z = particle.position[3],
            velocity_x = particle.velocity[1],
            velocity_y = particle.velocity[2],
            velocity_z = particle.velocity[3],
            angularVelocity_x = particle.angularVelocity[1],
            angularVelocity_y = particle.angularVelocity[2],
            angularVelocity_z = particle.angularVelocity[3],
        )
    end

    if !isfile("output.lbm/particlesTrj.csv")
        CSV.write("output.lbm/particlesTrj.csv", particleDf)
    else
        CSV.write("output.lbm/particlesTrj.csv", particleDf, append = true)
    end
end

function writeParticlesTrajectories(model::LBMmodel)
    for particle in model.particles
        writeParticleTrajectory(particle, model)
    end
end

function writeTensor(model::LBMmodel, T::Array, name::String)
    mkpath("output.lbm")

    if model.spaceTime.dims == 2
        metadataDf = DataFrame(
            tick = model.tick,
            time = model.time,
            id_x = [coordinate[1] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            id_y = [coordinate[2] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            coordinate_x = [coordinate[1] for coordinate in model.spaceTime.X] |> vec,
            coordinate_y = [coordinate[2] for coordinate in model.spaceTime.X] |> vec,
        ) # keyword argument constructor
        componentStrings = ["x", "y"]
    elseif model.spaceTime.dims == 3
        metadataDf = DataFrame(
            tick = model.tick,
            time = model.time,
            id_x = [coordinate[1] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            id_y = [coordinate[2] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            id_z = [coordinate[3] for coordinate in eachindex(IndexCartesian(), model.spaceTime.X)] |> vec,
            coordinate_x = [coordinate[1] for coordinate in model.spaceTime.X] |> vec,
            coordinate_y = [coordinate[2] for coordinate in model.spaceTime.X] |> vec,
            coordinate_z = [coordinate[3] for coordinate in model.spaceTime.X] |> vec,
        ) # keyword argument constructor
        componentStrings = ["x", "y", "z"]
    end

    componentLabels = ["component_"*i*j for i in componentStrings, j in componentStrings] |> vec
    tensorDf = T |> vec |> v -> DataFrame(
        vec.(v), componentLabels
    ) # vector of vectors constructor

    CSV.write("output.lbm/$name.csv", [metadataDf tensorDf])

    return nothing
end
