include("preamble.jl")
include("structs.jl");
include("aux.jl");
include("functions.jl");

#= ==========================================================================================
=============================================================================================
main - non-trivial initial conditions
=============================================================================================
========================================================================================== =#

unitφ(x,y) = (x == y == 0) ? ([0.; 0]) : [-y; x] ./ sqrt(x^2 + y^2)
f(x,y) = sqrt(x^2 + y^2) |> ρ -> (ρ < 1) ? (-cos(2π*ρ)/2 + 1/2) : 0
g(x) = -cos(2π*x) + 1
gx(x) = 2π*sin(2π*x)
gaussian(x,y) = exp(-(x^2+y^2)*5) / 2

len = 101;
x = range(-1, stop = 1, length = len); y = copy(x);
ρ = [1. for i in x, j in y];
#=u = [[1; 1.] for i in x, j in x];=#
#=u = [[i*(i-1)*(2j-1); -j*(j-1)*(2i-1)] for i in x, j in y];=#
#=u = [f(i,j) * unitφ(i,j) for i in x, j in x];=#
#=u = [f(i,j) * [1/2; 0]  for i in x, j in x];=#
u = [gaussian(i,j) * [1.; 1]  for i in x, j in x];
#=u = [[-gx(j) * i; g(j)] for i in x, j in x];=#

model = modelInit(ρ, u; walledDimensions = [ ]);

@time for _ in 1:201
    LBMpropagate!(model);
end
@time anim8fluidVelocity(model);
@time anim8massDensity(model);

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

t = 1

t += 100 
ρ = massDensity(model; time = t);
fig, ax, hm = heatmap(ρ, 
    axis=(
      title = "mass density, t = $(model.time[t] |> x -> round(x; digits = 2))",
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

