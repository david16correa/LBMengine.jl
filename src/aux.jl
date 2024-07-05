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
        return [indices[(Δ+1):end]; indices[1:Δ]]
    elseif Δ < 0
        # originalmente era [indices[end-Δ+1:end]; indices[1:end-Δ]] con un shift positivo, pero Δ < 0
        return [indices[end+Δ+1:end]; indices[1:end+Δ]]
    else
        return indices
    end
end

function pbcMatrixShift(M::Array{Float64}, Δ::Vector{Int64})
    return size(M) |> sizeM -> [pbcIndexShift(1:sizeM[i], Δ[i]) for i in eachindex(sizeM)] |> shiftedIndices -> M[shiftedIndices...]
end

# ---------------------------- wall and fluid nodes functions ---------------------------- 

function wallNodes(ρ::Array{Float64}, fluidSpeed::Int64)
    sizeM = size(ρ)
    wallMap = sizeM |> zeros .|> Bool
    dims, len = length(sizeM), sizeM[1];

    indices = [1:i for i in sizeM];
    auxIndices = copy(indices);
    paddingRanges = (1:fluidSpeed, len-fluidSpeed+1:len);

    for id in eachindex(indices), paddingRange in paddingRanges
        auxIndices[id] = paddingRange;
        wallMap[auxIndices...] .= 1;
        auxIndices = copy(indices)
    end

    return wallMap
end

function fluidNodes(ρ::Array{Float64}, fluidSpeed::Int64)
    return wallNodes(ρ, fluidSpeed) .|> boolN -> !boolN
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
