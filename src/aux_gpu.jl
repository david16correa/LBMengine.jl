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
function scalarFieldTimesVector_gpu(scalarField, vector)
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
function vectorFieldPlusVector_gpu(vectorField, vector)
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
function vectorFieldDotVector_gpu(vectorField, vector)
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
function vectorFieldDotVectorField_gpu(vField, wField)
    rowsCols = vField |> size
    dims = rowsCols |> length |> x -> x - 1 # for a vector field, the array will have one more dimension that the problem!
    output = CuArray{eltype(vField)}(undef, size(vField)[1:end-1])
    threads, blocks = getThreadsAndBlocks(dims, rowsCols)
    #
    kernel = [vectorFieldDotVectorField2D_kernel, vectorFieldDotVectorField3D_kernel][dims-1]
    @cuda threads=threads blocks=blocks kernel(output, vField, wField, rowsCols)
    return output
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
function findFluidVelocity_gpu(massDensity, momentumDensity)
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
