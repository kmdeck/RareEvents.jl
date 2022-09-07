using Revise
using RareEvents
using Statistics
using DelimitedFiles
using Plots

examples_dir = joinpath(pkgdir(RareEvents), "examples")
include(joinpath(examples_dir, "standard_map.jl"))

dt = 1.0
alg_kwargs = ();
FT = Float64
K_sm =2.0
model = StandardMap{Float64}(K_sm)
evolve_wrapper(tspan) = (u0) -> evolve_mapping_system(model, u0, tspan)


# Set up rare event algorithm
τ = 10.0
nensemble = 150
u0 = [[rand()*2π, rand()*2π, 1e-5, 1e-5] for _ in 1:nensemble]
kvals = [0.0,-0.05,-0.1]
ϵ = 0.0001
metric(y; K= K_sm) = (y[4]*y[3]*2.0*(1.0+K*cos(y[1])))/(y[3]^2.0+y[4]^2.0)
t_end = 150.0
t_start = 0.0
tspan = (t_start, t_end)
plot1 = plot()
extract_I(u) = u[2]
extract_θ(u) = u[1]
colors = [:cyan, :purple, :red]
ms = [1.0, 1.0, 1.0]
labels = ["k=0", "k = -0.05", "k = -0.1"]
# Run GKLT
for j in 1:2
    k = kvals[j]

    sim = RareEventSampler{Float64}(dt, τ, tspan,u0, evolve_wrapper, nensemble, metric, k, ϵ, iter)
    run!(sim);
    flatten_θ = extract_θ.(sim.ensemble[1])
    flatten_I = extract_I.(sim.ensemble[1])
    for i in 2:nensemble
        flatten_θ = vcat(flatten_θ, extract_θ.(sim.ensemble[i]))
        flatten_I = vcat(flatten_I, extract_I.(sim.ensemble[i]))
    end
    scatter!(plot1, flatten_θ, flatten_I, color = colors[j], ms = ms[j], label = labels[j], markerstrokewidth=0)
    
end
plot!(plot1, xlabel = "θ", ylabel = "I", xticks = ([0,π,2π],["0","π","2π"]), yticks = ([0, π, 2π],["0","π","2π"]))
savefig(plot1, "standard_map.png")
