#= ==========================================================================================
=============================================================================================
auxilary functions
=============================================================================================
========================================================================================== =#

function checkIdInModel(id::Int64, model::LBMmodel)
    (id in eachindex(model.distributions[end])) ? nothing : (error("No distribution with id $(id) was found!")) 
end

# ---------------- scalar and vector fild arithmetic auxilary functions ---------------- 

dot(v::Vector, w::Vector) = v .* w |> sum

norm(T) = √(sum(el for el ∈ T.*T))

function scalarFieldTimesVector(a::Array, V::Vector)
    return [a * V for a in a]
end

function vectorFieldDotVector(F::Array, v::Vector)
    return [dot(F, v) for F in F]
end

function vectorFieldDotVectorField(V::Array, W::Array)
    return size(V) |> sizeV -> [dot(V[id], W[id]) for id in eachindex(IndexCartesian(), V)]
end

# -------------------------------- shift auxilary functions -------------------------------- 

function pbcIndexShift(indices::UnitRange{Int64}, Δ::Int64)
    if Δ > 0
        return [indices[end-Δ+1:end]; indices[1:end-Δ]]
    elseif Δ < 0
        # originalmente era [indices[(Δ+1):end]; indices[1:Δ]] con un shift positivo, pero Δ < 0
        return [indices[(-Δ+1):end]; indices[1:-Δ]]
    else
        return indices
    end
end

function pbcMatrixShift(M::Union{Array{Float64}, SparseMatrixCSC, BitArray}, Δ::Vector{Int64})
    return size(M) |> sizeM -> [pbcIndexShift(1:sizeM[i], Δ[i]) for i in eachindex(sizeM)] |> shiftedIndices -> M[shiftedIndices...]
end

# ---------------------------- wall and fluid nodes functions ---------------------------- 

function wallNodes(ρ::Array{Float64}, fluidSpeed::Int64)
    # the size, dimensions, and side length of the density field are saved
    sizeM = size(ρ)
    dims, len = length(sizeM), sizeM[1];

    # the wallMap is initialized as an boolean array filled with zeroes,
    # and the indices of the density field are saved.
    wallMap = sizeM |> zeros .|> Bool
    indices = [1:i for i in sizeM];

    # for each dimension, a padding will be addd. To do this, a set of auxilary indices will be needed.
    auxIndices = copy(indices);
    paddingRanges = (1:fluidSpeed, len-fluidSpeed+1:len);

    # the padding is added in every dimension
    for id in eachindex(indices), paddingRange in paddingRanges
        auxIndices[id] = paddingRange;
        wallMap[auxIndices...] .= 1;
        auxIndices = copy(indices)
    end

    #  the final wall map is returned as a sparse matrix
    (dims > 2) ? (return wallMap) : (return wallMap |> sparse)
    #  SparseArrays.jl only works for 1D and 2D. Look into SparseArrayKit.jl for higher dimensional compatibility!!
end

function fluidNodes(ρ::Array{Float64}, fluidSpeed::Int64)
    return wallNodes(ρ, fluidSpeed) .|> boolN -> !boolN
end

# --------------------------- bounce-back boundary conditions --------------------------- 

function bounceBackPrep(wallRegion::Union{SparseMatrixCSC, BitArray}, velocities::Vector{LBMvelocity})
    cs = [velocity.c for velocity in velocities];

    streamingInvasionRegions = [(pbcMatrixShift(wallRegion, -c) .|| wallRegion) .⊻ wallRegion for c in cs]

    oppositeVectorId = [findfirst(x -> x == -c, cs) for c in cs]

    return streamingInvasionRegions, oppositeVectorId
end

# ---------------- some velocity sets ---------------- 

cs = [
      [0],
      #======#
      [1],
      [-1]
];
ws = [
      2/3,
      #======#
      1/6,
      1/6
];
D1Q3 = [LBMvelocity(cs[i], ws[i]) for i in eachindex(cs)];

cs = [
      [0,0],
      #======#
      [1,0],
      [-1,0],
      [0,1],
      [0,-1],
      #======#
      [1,1],
      [-1,1],
      [1,-1],
      [-1,-1]
];
ws = [
      4/9,
      #======#
      1/9,
      1/9,
      1/9,
      1/9,
      #======#
      1/36,
      1/36,
      1/36,
      1/36
];
D2Q9 = [LBMvelocity(cs[i], ws[i]) for i in eachindex(cs)];

cs = [
      [0,0,0], 
      #======#
      [1,0,0],
      [0,1,0],
      [0,0,1],
      [-1,0,0],
      [0,-1,0],
      [0,0,-1],
      #======#
      [1,1,0],
      [1,-1,0],
      [-1,1,0],
      [-1,-1,0],
      [0,1,1],
      [0,1,-1],
      [0,-1,1],
      [0,-1,-1],
      [1,0,1],
      [1,0,-1],
      [-1,0,1],
      [-1,0,-1],
      #======#
      [1,1,1],
      [1,1,-1],
      [1,-1,1],
      [1,-1,-1],
      [-1,1,1],
      [-1,1,-1],
      [-1,-1,1],
      [-1,-1,-1]
];
ws = [
      8/27,
      #======#
      2/27,
      2/27,
      2/27,
      2/27,
      2/27,
      2/27,
      #======#
      1/54,
      1/54,
      1/54,
      1/54,
      1/54,
      1/54,
      1/54,
      1/54,
      1/54,
      1/54,
      1/54,
      1/54,
      #======#
      1/216,
      1/216,
      1/216,
      1/216,
      1/216,
      1/216,
      1/216,
      1/216
];
D3Q27 = [LBMvelocity(cs[i], ws[i]) for i in eachindex(cs)];
