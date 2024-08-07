include("preamble.jl")
include("structs.jl");
include("aux.jl");
include("functions.jl");

#= ==========================================================================================
=============================================================================================
main
=============================================================================================
========================================================================================== =#

gaussian(x,y) = exp(-(x^2+y^2)*5) 

len = 201;
x = range(-1, stop = 1, length = len);
#=u = [[i*(i-1)*(2j-1); -j*(j-1)*(2i-1)] for i in x, j in y];=#
#=u = [f(i,j) * unitφ(i,j) for i in x, j in x];=#
#=u = [f(i,j) * [1/2; 0]  for i in x, j in x];=#
u = [gaussian(i,j) * [0.1e-2; 0]  for i in x, j in x]; # |u| must be small for fluid to be incompressible! M ≈ 3u/c_s << 1
#=u = [[-gx(j) * i; g(j)] for i in x, j in x];=#

solidNodes = [
    ((-0.75 < i < -0.25) && j < 0.) || ((0.25 < i < 0.75) && j > -0.)
for i in x, j in x];


model = modelInit(; 
    u = u,
    x = x,
    τ_Δt = 2.0, # τ/Δt > 1 → under-relaxation, τ/Δt = 1 → full relaxation, 0.5 < τ/Δt < 1 → over-relaxation, τ/Δt < 0.5 → unstable
    #=walledDimensions = [ ],=#
    walledDimensions = [2],
    #=walledDimensions = [1, 2],=#
    solidNodes = solidNodes
);

#=u / model.fluidParams.c_s .|> norm |> maximum=# model.fluidParams.c2_s * (model.fluidParams.τ - model.spaceTime.Δt/2) # Viscosity

#=machNumbers = [];=#

@time for _ in 1:500
    LBMpropagate!(model);
    #=u / model.fluidParams.c_s .|> norm |> maximum |> x -> append!(machNumbers, [x])=#
end
#=lines(machNumbers, axis = (xlabel = "t", ylabel = "M",))=#
@time plotMassDensity(model);
@time plotFluidVelocity(model);
#=@time plotMomentumDensity(model);=#
#=@time anim8massDensity(model);=#
@time anim8fluidVelocity(model);
#=@time anim8momentumDensity(model);=#
