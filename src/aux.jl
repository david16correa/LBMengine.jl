#= ==========================================================================================
=============================================================================================
auxilary functions
=============================================================================================
========================================================================================== =#

function checkIdInModel(id::Int64, model::LBMmodel)
    !(0 < id <= length(model.distributions[end])) ? (error("No distribution with id $(id) was found!")) : nothing
end

function scalarFieldTimesVector(a::Matrix, V::Vector)
    return [a * V for a in a]
end

function vectorFieldDotVector(F::Matrix, v::Vector)
    dot(v, w) = v .* w |> sum
    return [dot(F, v) for F in F]
end

function vectorFieldDotVectorField(V::Matrix, W::Matrix)
    dot(v, w) = v .* w |> sum
    return [dot(V[i,j], W[i,j]) for i in 1:size(V,1), j in 1:size(V,2)]
end

