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

# ------------------------------------ graphics stuff ----------------------------------- 

function save_jpg(name::String, fig::Figure)
    namePNG = name*".png"
    nameJPG = name*".jpg"
    save(namePNG, fig) 
    run(`magick $namePNG $nameJPG`)
    run(`rm $namePNG`)
end

"The animation of the cart moving along the roller coaster is created."
function LBManim8(model::LBMmodel; desired_fps = 30)
    # the step and frame rate of the animation that can be achieved with the provided data are determined from a requested frame rate
    animStep = (model.time[2] - model.time[1]) * desired_fps |> inv |> round |> Int64
    achieved_fps = model.time[1:animStep:end] |> v -> v[2] - v[1] |> inv |> round |> Int64

    animTime = 1:animStep:length(model.time) 

    mkdir("tmp")

    for t in eachindex(model.time)
        ρu = momentumDensity(model; time = t) 
        #----------------------------------heatmap and colorbar---------------------------------
        animationFig, animationAx, hm = heatmap(model.spaceTime.x, model.spaceTime.x, norm.(ρu), alpha = 0.7,
            #=colorrange = (0, 1), =#
            #=highclip = :red, # truncate the colormap =#
            #=lowclip = :blue, # truncate the colormap =#
            axis=(
                title = "momentum density, t = $(model.time[t])",
            )
        );
        animationAx.xlabel = "x"; animationAx.ylabel = "y";
        Colorbar(animationFig[:, end+1], hm,
            #=ticks = (-1:0.5:1, ["$i" for i ∈ -1:0.5:1]),=#
        );
        #--------------------------------------gradient---------------------------------------
        pos = [Point2(i,j) for i ∈ model.spaceTime.x[1:10:end] for j ∈ model.spaceTime.x[1:10:end]];
        vec = [ρu[i,j] for i ∈ eachindex(model.spaceTime.x)[1:10:end] for j ∈ eachindex(model.spaceTime.x)[1:10:end]];
        lengths = norm.(vec) .|> len -> (len == 0) ? (len = 1) : (len = len);
        vec = 0.05 .* vec ./ lengths;
        arrows!(animationFig[1,1], pos, vec, 
            arrowsize = 10, 
            align = :center
        );
        save("tmp/$(t).png", animationFig)
    end

    run(`./createAnim.sh`)
    name = "anims/$(today())/LBM simulation $(Time(now())).mp4"
    run(`mv anims/output.mp4 $(name)`)
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
