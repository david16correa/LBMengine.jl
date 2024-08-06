include("preamble.jl")
include("structs.jl");
include("aux.jl");
include("functions.jl");

#= ==========================================================================================
=============================================================================================
main - initial conditions with pressure differences
=============================================================================================
========================================================================================== =#

x = range(-1, stop = 1, step = 0.02); 
ρ = [1. for i in x, j in x];
u = [[0.; 0] for _ in ρ];
#=solidNodes = [((i+0.25)^2 + j^2) < 0.2^2 for i in x, j in x];=#
solidNodes = [
    ((-0.75 < i < -0.25) && j < 0.) || ((0.25 < i < 0.75) && j > -0.)
for i in x, j in x];

model = modelInit(ρ, u;
    densityHL = (1.1, 0.9),
    solidNodes = solidNodes
);

@time for _ in 1:500
    LBMpropagate!(model);
end
@time plotMassDensity(model);
@time plotFluidVelocity(model);
@time plotMomentumDensity(model);

@time anim8massDensity(model);
@time anim8fluidVelocity(model);
@time anim8momentumDensity(model);

#=hydroVariablesUpdate!(model; time = time)=#
fig = Figure();
len = model.spaceTime.x |> length
ax = Axis(fig[1,1], title = "fluid speed, t = $(model.time[end] |> x -> round(x; digits = 2)), x = $(model.spaceTime.x[len/2 |> round |> Int64])");
ax.xlabel = "y"; ax.ylabel = "|u|";
model.u[len/2 |> round |> Int64, :] .|> norm |> v -> lines!(ax, model.spaceTime.x, v);
fig

#= ==========================================================================================
=============================================================================================
main - non-trivial initial conditions (DEPRECATED!)
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
#=u = [[i*(i-1)*(2j-1); -j*(j-1)*(2i-1)] for i in x, j in y];=#
#=u = [f(i,j) * unitφ(i,j) for i in x, j in x];=#
#=u = [f(i,j) * [1/2; 0]  for i in x, j in x];=#
u = [gaussian(i,j) * [1.; 1]  for i in x, j in x];
#=u = [[-gx(j) * i; g(j)] for i in x, j in x];=#


model = modelInit(ρ, u; 
    walledDimensions = [-1],
    pressurizedDimensions = [ ],
);

model = modelInit(ρ, u; 
    walledDimensions = [2],
    pressurizedDimensions = [ ],
);

model = modelInit(ρ, u; 
    walledDimensions = [ ],
    pressurizedDimensions = [ ],
);

@time for _ in 1:101
    LBMpropagate!(model);
end

@time anim8fluidVelocity(model);

@time anim8massDensity(model);
