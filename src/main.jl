include("preamble.jl")
include("structs.jl");
include("aux.jl");
include("functions.jl");

#= ==========================================================================================
=============================================================================================
main - non-trivial initial conditions
=============================================================================================
========================================================================================== =#

len = 100;
x = range(-1, stop = 1, length = len);
#=ρ = [1. for i in x, j in x];=#
u = [[1; 0.] for i in x, j in x];
#=u = [[i; 0.] for i in x, j in x];=#
#=u = [(i == j == 0) ? ([0.,0]) : [-j; i]./sqrt(i^2 + j^2) for i in x, j in x];=#
model = modelInit(ρ, u);
@time for _ in 1:101
    LBMpropagate!(model);
end
LBManim8(model);

#= ==========================================================================================
=============================================================================================
main - trivial initial conditions
=============================================================================================
========================================================================================== =#

len, dims = 100, 2; 
ρ = [1. for _ in Array{Int64}(undef, (len for _ in 1:dims)...)];
u = [[0. for _ in 1:dims] for _ in ρ];
model = modelInit(ρ, u; τ = 1.);

@time for _ in 1:101
    LBMpropagate!(model);
end

#= ==========================================================================================
=============================================================================================
misc - graphical stuff
=============================================================================================
========================================================================================== =#

t = 101
ρ = massDensity(model; time = t);
fig, ax, hm = heatmap(ρ, 
    axis=(
        title = "mass density, t = $(model.time[t])",
    )
);
Colorbar(fig[1,2], hm);
fig

t = 0

#=t += 10=#
t = 101
ρ = massdensity(model; time = t);
ρu = momentumdensity(model; time = t);
#----------------------------------heatmap and colorbar---------------------------------
z = norm.(ρu);
fig, ax, hm = heatmap(x,x,z, alpha = 0.7,
    axis=(
        title = "momentum density, t = $(model.time[t])",
    )
);
ax.xlabel = "x"; ax.ylabel = "y";
colorbar(fig[:, end+1], hm,
    #=ticks = (-1:0.5:1, ["$i" for i ∈ -1:0.5:1]),=#
);
#--------------------------------------gradient---------------------------------------
pos = [point2(i,j) for i ∈ x[1:10:end] for j ∈ x[1:10:end]];
vec = [ρu[i,j] for i ∈ eachindex(x)[1:10:end] for j ∈ eachindex(x)[1:10:end]];
lengths = norm.(vec) .|> len -> (len == 0) ? (len = 1) : (len = len);
vec = 0.05 .* vec ./ lengths;
arrows!(fig[1,1], pos, vec, 
    arrowsize = 10, 
    align = :center
);
t
fig

