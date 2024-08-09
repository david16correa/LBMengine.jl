include("preamble.jl")
include("structs.jl");
include("aux.jl");
include("functions.jl");

#= ==========================================================================================
=============================================================================================
main
=============================================================================================
========================================================================================== =#

len = 100;
x = range(-1, stop = 1, length = len);
solidNodes = [
    ((-0.75 < i < -0.25) && j < 0.75) || ((0.25 < i < 0.75) && j > -0.75)
    #=((-0.75 < i < -0.25) && j < 0.) || ((0.25 < i < 0.75) && j > -0.)=#
    #=((i + 0.3)^2 + j^2) < 0.25^2=#
for i in x, j in x];

model = modelInit(; 
    x = x,
    Δt = :default, # default: Δt = Δx
    τ_Δt = 2.6, # τ/Δt > 1 → under-relaxation, τ/Δt = 1 → full relaxation, 0.5 < τ/Δt < 1 → over-relaxation, τ/Δt < 0.5 → unstable
    walledDimensions = [2],
    solidNodes = solidNodes,
    forceDensity = [0.5e-3, 0.0],
    fluidIsCompressible = false,
    forcingScheme = :guo # {:guo, :shan}, default: Guo, C. Zheng, B. Shi, Phys. Rev. E 65, 46308 (2002)
);

@time LBMpropagate!(model; simulationTime = 5, verbose = true);
#=lines(machNumbers, axis = (xlabel = "t", ylabel = "M",))=#
@time plotMassDensity(model);
@time plotFluidVelocity(model);
#=@time plotMomentumDensity(model);=#
#=model.fluidParams.c2_s * (model.fluidParams.τ - model.spaceTime.Δt/2) # Viscosity=#
model.u / model.fluidParams.c_s .|> norm |> maximum # Mach Number
model.schemes

#=@time anim8massDensity(model);=#
@time anim8fluidVelocity(model);
#=@time anim8momentumDensity(model);=#

fig = Figure();
ax = Axis(fig[1,1], title = "fluid speed, t = $(model.time[end] |> x -> round(x; digits = 2)), y = $(model.spaceTime.x[50])");
ax.xlabel = "y"; ax.ylabel = "|u|";
model.u[:, 50] .|> norm |> v -> lines!(ax, model.spaceTime.x, v);
fig

#= ==========================================================================================
=============================================================================================
poiseuille flow
=============================================================================================
========================================================================================== =#

len = 100;
x = range(-1, stop = 1, length = len);
solidNodes = [
    -0.5 > j || j > 0.5
    #=((i + 0.5)^2 + j^2) < 0.15^2=#
for i in x, j in x];

model = modelInit(; 
    x = x,
    Δt = :default, # default: Δt = Δx
    τ_Δt = 2.6, # τ/Δt > 1 → under-relaxation, τ/Δt = 1 → full relaxation, 0.5 < τ/Δt < 1 → over-relaxation, τ/Δt < 0.5 → unstable
    walledDimensions = [2],
    solidNodes = solidNodes,
    forceDensity = [0.5e-3, 0.0],
    fluidIsCompressible = false,
    forcingScheme = :guo # {:guo, :shan}, default: Guo, C. Zheng, B. Shi, Phys. Rev. E 65, 46308 (2002)
);

@time LBMpropagate!(model; simulationTime = 5, verbose = true);
#=model.u / model.fluidParams.c_s .|> norm |> maximum # Mach Number=#
#=lines(machNumbers, axis = (xlabel = "t", ylabel = "M",))=#
@time plotMassDensity(model);
@time plotFluidVelocity(model);
fig = Figure();
ax = Axis(fig[1,1], title = "fluid speed, t = $(model.time[end] |> x -> round(x; digits = 2)), x = $(model.spaceTime.x[50])");
ax.xlabel = "y"; ax.ylabel = "|u|";
model.u[50, :] .|> norm |> v -> lines!(ax, model.spaceTime.x, v);
fig

#= ==========================================================================================
=============================================================================================
medir el promedio del valor absoluto de la velocidad. Y luego hacer una gráfica vs tau
=============================================================================================
========================================================================================== =#

gaussian(x,y) = exp(-(x^2+y^2)*5) 
len = 201;
x = range(-1, stop = 1, length = len);
u = [gaussian(i,j) * [0.1e-2; 0]  for i in x, j in x]; # |u| must be small for fluid to be incompressible! M ≈ 3u/c_s << 1
solidNodes = [
    ((-0.75 < i < -0.25) && j < 0.) || ((0.25 < i < 0.75) && j > -0.)
for i in x, j in x];

relaxationTimes = []
fluidSpeeds = []

model = modelInit(; u = u, x = x, walledDimensions = [2], solidNodes = solidNodes,
    τ_Δt = 0.8, # τ/Δt > 1 → under-relaxation, τ/Δt = 1 → full relaxation, 0.5 < τ/Δt < 1 → over-relaxation, τ/Δt < 0.5 → unstable
);

@time for τ_Δt in range(0.5, stop = 2, length = 2)
    @show τ_Δt
    model = modelInit(; u = u, x = x, walledDimensions = [2], solidNodes = solidNodes,
        τ_Δt = τ_Δt, # τ/Δt > 1 → under-relaxation, τ/Δt = 1 → full relaxation, 0.5 < τ/Δt < 1 → over-relaxation, τ/Δt < 0.5 → unstable
    );
    LBMpropagate!(model, 5);
    model.u .|> norm |> mean |> normU -> append!(fluidSpeeds, [normU]);
    append!(relaxationTimes, [τ_Δt]);
end

fluidSpeeds
relaxationTimes

lines(relaxationTimes, fluidSpeeds, axis = (xlabel = "τ/Δt", ylabel = "average fluid speed",))
lines(relaxationTimes .|> τ -> model.fluidParams.c2_s * model.spaceTime.Δt * (τ - 1/2), fluidSpeeds, axis = (xlabel = "viscosity", ylabel = "average fluid speed",))
lines(relaxationTimes .|> τ -> model.fluidParams.c2_s * model.spaceTime.Δt * (τ - 1/2), fluidSpeeds .|> u -> u/model.fluidParams.c_s, axis = (xlabel = "viscosity", ylabel = "average Mach number",))


