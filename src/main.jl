include("preamble.jl")
include("structs.jl");
include("aux.jl");
include("functions.jl");

#= ==========================================================================================
=============================================================================================
main
=============================================================================================
========================================================================================== =#

unitφ(x,y) = (x == y == 0) ? ([0.; 0]) : [-y; x] ./ sqrt(x^2 + y^2)
f(x,y) = sqrt(x^2 + y^2) |> ρ -> (ρ < 1) ? (-cos(2π*ρ)/2 + 1/2) : 0
g(x) = -cos(2π*x) + 1
gx(x) = 2π*sin(2π*x)
gaussian(x,y) = exp(-(x^2+y^2)*5) / 2

len = 101;
x = range(-1, stop = 1, length = len);
#=u = [[i*(i-1)*(2j-1); -j*(j-1)*(2i-1)] for i in x, j in y];=#
#=u = [f(i,j) * unitφ(i,j) for i in x, j in x];=#
#=u = [f(i,j) * [1/2; 0]  for i in x, j in x];=#
u = [gaussian(i,j) * [1.; 1]  for i in x, j in x];
#=u = [[-gx(j) * i; g(j)] for i in x, j in x];=#

model = modelInit(; 
    u = u,
    x = x
);

model = modelInit(; 
    u = u,
    x = x,
    walledDimensions = [1, 2],
);

model = modelInit(; 
    u = u,
    x = x,
    walledDimensions = [2],
);

model = modelInit(; 
    u = u,
    x = x,
    walledDimensions = [ ],
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
