#= ==========================================================================================
=============================================================================================
CUDA kernels for gpu acceleration
=============================================================================================
========================================================================================== =#


#= ==========================================================================================
=============================================================================================
scalar and vector field arithmetic auxiliary functions
=============================================================================================
========================================================================================== =#

function fillCuArray3D_kernel(output, val::Number, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    #
    if i <= rowsCols[1] && j <= rowsCols[2] && k <= rowsCols[3]
        if length(rowsCols) > 3
            for l in 1:rowsCols[4]
                @inbounds output[i,j,k,l] = val
            end
        else
            @inbounds output[i,j,k] = val
        end
    end
    return nothing
end
function fillCuArray2D_kernel(output, val::Number, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        if length(rowsCols) > 2
            for k in 1:rowsCols[3]
                @inbounds output[i,j,k] = val
            end
        else
            @inbounds output[i,j] = val
        end
    end
    return nothing
end
function fillCuArray3D_kernel(output, val::AbstractArray, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    #
    if i <= rowsCols[1] && j <= rowsCols[2] && k <= rowsCols[3]
        for l in 1:rowsCols[4]
            @inbounds output[i,j,k,l] = val[k]
        end
    end
    return nothing
end
function fillCuArray2D_kernel(output, val::AbstractArray, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        for k in 1:rowsCols[3]
            @inbounds output[i,j,k] = val[k]
        end
    end
    return nothing
end

function coordinatesCuArray3D_kernel(output, coordinates, rowsCols, dims)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    #
    if i <= rowsCols[1] && j <= rowsCols[2] && k <= rowsCols[3]
        for l in 1:dims
            if l == 1
                @inbounds output[i,j,k,l] = coordinates[l][i]
            elseif l == 2
                @inbounds output[i,j,k,l] = coordinates[l][j]
            else
                @inbounds output[i,j,k,l] = coordinates[l][k]
            end
        end
    end
    return nothing
end
function coordinatesCuArray2D_kernel(output, coordinates, rowsCols, dims)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        for k in 1:dims
            if k == 1
                @inbounds output[i,j,k] = coordinates[k][i]
            else
                @inbounds output[i,j,k] = coordinates[k][j]
            end
        end
    end
    return nothing
end

function vectorBitId3D_kernel(output, bitId, rowsCols, dims)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    #
    if i <= rowsCols[1] && j <= rowsCols[2] && k <= rowsCols[3]
        for l in 1:dims
            @inbounds output[i,j,k,l] = bitId[i,j,l]
        end
    end
    return nothing
end
function vectorBitId2D_kernel(output, bitId, rowsCols, dims)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        for k in 1:dims
            @inbounds output[i,j,k] = bitId[i,j]
        end
    end
    return nothing
end

function scalarFieldTimesVector2D_kernel(output, scalarField, vector, N, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        for k in 1:N
            @inbounds output[i,j,k] = scalarField[i,j] * vector[k]
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
            @inbounds output[i,j,k,l] = scalarField[i,j,k] * vector[l]
        end
    end
    return nothing
end

function scalarFieldTimesVectorField2D_kernel(output, scalarField, vectorField, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        for k in 1:rowsCols[3]
            @inbounds output[i,j,k] = scalarField[i,j] * vectorField[i,j,k]
        end
    end
    return nothing
end
function scalarFieldTimesVectorField3D_kernel(output, scalarField, vectorField, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    #
    if i <= rowsCols[1] && j <= rowsCols[2] && k <= rowsCols[3]
        for l in 1:rowsCols[4]
            @inbounds output[i,j,k,l] = scalarField[i,j,k] * vectorField[i,j,k,l]
        end
    end
    return nothing
end

function vectorFieldPlusVector2D_kernel(output, vector, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        for k in 1:rowsCols[3]
            @inbounds output[i,j,k] += vector[k]
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
            @inbounds output[i,j,k,l] += vector[l]
        end
    end
    return nothing
end

function vectorFieldDotVector2D_kernel(output, vectorField, vector, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        sumTerm = zero(eltype(output))
        for k in 1:rowsCols[3]
            @inbounds sumTerm += vectorField[i,j,k] * vector[k]
        end
        @inbounds output[i,j] = sumTerm
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
            @inbounds sumTerm += vectorField[i,j,k,l] * vector[l]
        end
        @inbounds output[i,j,k] = sumTerm
    end
    return nothing
end

function vectorFieldDotVectorField2D_kernel(output, vField, wField, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        sumTerm = zero(eltype(output))
        for k in 1:rowsCols[3]
            @inbounds sumTerm += vField[i,j,k] * wField[i,j,k]
        end
        @inbounds output[i,j] = sumTerm
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
            @inbounds sumTerm += vField[i,j,k,l] * wField[i,j,k,l]
        end
        @inbounds output[i,j,k] = sumTerm
    end
    return nothing
end

function circshift2D_kernel(output, input, shift, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i ≤ rowsCols[1] && j ≤ rowsCols[2]
        new_i = mod1(i - shift[1], rowsCols[1])
        new_j = mod1(j - shift[2], rowsCols[2])
        if length(rowsCols) > 2
            for k in 1:rowsCols[3]
                @inbounds output[i,j,k] = input[new_i, new_j, k]
            end
        else
            @inbounds output[i,j] = input[new_i, new_j]
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
                @inbounds output[i,j,k,l] = input[new_i, new_j, new_k, l]
            end
        else
            @inbounds output[i,j,k] = input[new_i, new_j, new_k]
        end
    end
    return nothing
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

#= ==========================================================================================
=============================================================================================
particles@gpu aux
=============================================================================================
========================================================================================== =#

function getSphere2D_kernel(output, radius2, R, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        output[i,j] = R[i]^2 + R[j]^2 < radius2
    end
    return nothing
end
function getSphere3D_kernel(output, radius2, R, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    #
    if i <= rowsCols[1] && j <= rowsCols[2] && k <= rowsCols[3]
        output[i,j,k] = R[i]^2 + R[j]^2 + R[k]^2 < radius2
    end
    return nothing
end


function getSolidNodes2D_kernel(output, solidRegionGenerator, R, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        output[i,j] = solidRegionGenerator((R[i], R[j]))
    end
    return nothing
end
function getSolidNodes3D_kernel(output, solidRegionGenerator, R, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    #
    if i <= rowsCols[1] && j <= rowsCols[2] && k <= rowsCols[3]
        output[i,j,k] = solidRegionGenerator((R[i], R[j], R[k]))
    end
    return nothing
end

function getNodeVelocity2D_kernel(output, particleKws, X, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        vel = bulkVelocity(particleKws..., (X[i,j,1], X[i,j,2]));
        return nothing
        output[i,j,1] = vel[1];
        output[i,j,2] = vel[2];
    end
    return nothing
end
function getNodeVelocity3D_kernel(output, particleKws, X, rowsCols)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    #
    if i <= rowsCols[1] && j <= rowsCols[2] && k <= rowsCols[3]
        vel = bulkVelocity(particleKws..., [X[i,j,k,1], X[i,j,k,2], X[i,j,k,3]]);
        output[i,j,1] = vel[1];
        output[i,j,2] = vel[2];
        output[i,j,3] = vel[3];
    end
    return nothing
end


function getBulkV2D_kernel(
    output,
    velocity,
    angularVelocity,
    xMinusR,
    xMinusR_norm,
    radius,
    latticeParameter,
    rowsCols
)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    #
    if i <= rowsCols[1] && j <= rowsCols[2]
        output[i,j,1] = velocity[1] - angularVelocity * xMinusR[i,j,2]
        output[i,j,2] = velocity[2] + angularVelocity * xMinusR[i,j,1]
    end
    return nothing
end
function getBulkV3D_kernel(
    output,
    velocity,
    angularVelocity,
    xMinusR,
    xMinusR_norm,
    radius,
    latticeParameter,
    rowsCols
)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    #
    if i <= rowsCols[1] && j <= rowsCols[2] && k <= rowsCols[3]
        output[i,j,k,1] = velocity[1] + angularVelocity[2]*xMinusR[i,j,k,3] - angularVelocity[3]*xMinusR[i,j,k,2]
        output[i,j,k,2] = velocity[2] - angularVelocity[1]*xMinusR[i,j,k,3] + angularVelocity[3]*xMinusR[i,j,k,1]
        output[i,j,k,3] = velocity[3] + angularVelocity[1]*xMinusR[i,j,k,2] - angularVelocity[2]*xMinusR[i,j,k,1]
    end
    return nothing
end
