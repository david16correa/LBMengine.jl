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

function scalarFieldTimesVector(a::Matrix, V::Vector)
    return [a * V for a in a]
end

function vectorFieldDotVector(F::Matrix, v::Vector)
    return [dot(F, v) for F in F]
end

function vectorFieldDotVectorField(V::Matrix, W::Matrix)
    return size(V) |> sizeV -> [dot(V[i,j], W[i,j]) for i in 1:sizeV[1], j in 1:sizeV[2]]
end

# ---------------- shift auxilary functions ---------------- 

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

