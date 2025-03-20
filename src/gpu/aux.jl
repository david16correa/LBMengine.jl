#= ==========================================================================================
=============================================================================================
GPU accelerated scalar and vector field arithmetic auxiliary functions
=============================================================================================
========================================================================================== =#

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

function scalarFieldTimesVector2D_kernel(output, scalarField, vector, N, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        for k in 1:N
            output[i,j,k] = scalarField[i,j] * vector[k]
        end
    end
    return nothing
end
function scalarFieldTimesVector3D_kernel(output, scalarField, vector, N, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    #
    if i <= rowsCols[1] && j <= rowsCols[2] && k <= rowsCols[3]
        for l in 1:N
            output[i,j,k,l] = scalarField[i,j,k] * vector[l]
        end
    end
    return nothing
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

function vectorFieldPlusVector2D_kernel(output, vector, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        for k in 1:rowsCols[3]
            output[i,j,k] += vector[k]
        end
    end
    return nothing
end
function vectorFieldPlusVector3D_kernel(output, vector, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    #
    if i <= rowsCols[1] && j <= rowsCols[2] && k <= rowsCols[3]
        for l in 1:rowsCols[4]
            output[i,j,k,l] += vector[l]
        end
    end
    return nothing
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

function vectorFieldDotVector2D_kernel(output, vectorField, vector, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        sumTerm = zero(eltype(output))
        for k in 1:rowsCols[3]
            sumTerm += vectorField[i,j,k] * vector[k]
        end
        output[i,j] = sumTerm
    end
    return nothing
end
function vectorFieldDotVector3D_kernel(output, vectorField, vector, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    #
    if i <= rowsCols[1] && j <= rowsCols[2] && k <= rowsCols[3]
        sumTerm = zero(eltype(output))
        for l in 1:rowsCols[4]
            sumTerm += vectorField[i,j,k,l] * vector[l]
        end
        output[i,j,k] = sumTerm
    end
    return nothing
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

function vectorFieldDotVectorField2D_kernel(output, vField, wField, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        sumTerm = zero(eltype(output))
        for k in 1:rowsCols[3]
            sumTerm += vField[i,j,k] * wField[i,j,k]
        end
        output[i,j] = sumTerm
    end
    return nothing
end
function vectorFieldDotVectorField3D_kernel(output, vField, wField, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    #
    if i <= rowsCols[1] && j <= rowsCols[2] && k <= rowsCols[3]
        sumTerm = zero(eltype(output))
        for l in 1:rowsCols[4]
            sumTerm += vField[i,j,k,l] * wField[i,j,k,l]
        end
        output[i,j,k] = sumTerm
    end
    return nothing
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

function circshift2D_kernel(output, input, shift, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i ≤ rowsCols[1] && j ≤ rowsCols[2]
        new_i = mod1(i - shift[1], rowsCols[1])
        new_j = mod1(j - shift[2], rowsCols[2])
        if length(rowsCols) > 2
            for k in 1:rowsCols[3]
                output[i,j,k] = input[new_i, new_j, k]
            end
        else
            output[i,j] = input[new_i, new_j]
        end
    end
    return nothing
end
function circshift3D_kernel(output, input, shift, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z

    if i <= rowsCols[1] && j <= rowsCols[2] && k <= rowsCols[3]
        new_i = mod1(i - shift[1], rowsCols[1])
        new_j = mod1(j - shift[2], rowsCols[2])
        new_k = mod1(k - shift[3], rowsCols[3])
        if length(rowsCols) > 3
            for l in 1:rowsCols[4]
                output[i,j,k,l] = input[new_i, new_j, new_k, l]
            end
        else
            output[i,j,k] = input[new_i, new_j, new_k]
        end
    end
    return nothing
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

function findFluidVelocity2D_kernel(fluidVelocity, massDensity, momentumDensity, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    rows, cols, depth = size(fluidVelocity)
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        for k in 1:rowsCols[3]  # Iterate over the small depth dimension
            if !(massDensity[i, j] ≈ 0)
                fluidVelocity[i, j, k] = momentumDensity[i, j, k] / massDensity[i, j]
            end
        end
    end
    return nothing
end
function findFluidVelocity3D_kernel(fluidVelocity, massDensity, momentumDensity, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    #
    rows, cols, depth = size(fluidVelocity)
    #
    if i <= rowsCols[1] && j <= rowsCols[2] && k <= rowsCols[3]
        for l in 1:rowsCols[4]  # Iterate over the small depth dimension
            if !(massDensity[i,j,k] ≈ 0)
                fluidVelocity[i,j,k,l] = momentumDensity[i,j,k,l] / massDensity[i,j,k]
            end
        end
    end
    return nothing
end
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

function bounceBackPrep(wallRegion::Union{SparseMatrixCSC, BitArray}, velocities::Vector{LBMvelocity}; returnStreamingInvasionRegions = false)
    cs = [velocity.c |> Tuple for velocity in velocities];

    streamingInvasionRegions = [(circshift(wallRegion, -1 .* c) .|| wallRegion) .⊻ wallRegion for c in cs]
    streamingInvasionRegions = CuArray(cat(streamingInvasionRegions...; dims=length(cs[1])+1));

    returnStreamingInvasionRegions && return streamingInvasionRegions

    oppositeVectorId = [findfirst(x -> x == -1 .* c, cs) for c in cs]


    return streamingInvasionRegions, oppositeVectorId
end
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
