using Revise
using RareEvents
using Statistics
using DelimitedFiles
using LinearAlgebra
using Plots
examples_dir = joinpath(pkgdir(RareEvents), "examples")
include(joinpath(examples_dir, "ornstein_uhlenbeck.jl"))
n_processes = Sys.CPU_THREADS
addprocs(n_processes)

dt = 0.1
alg_kwargs = ();
FT = Float64
d = 1
θ = Diagonal(ones(d)) .+ zeros(Float64,d,d)
σ = Diagonal(ones(d)) .+ zeros(Float64,d,d)

model = OrnsteinUhlenbeck{FT}(θ, σ, d)

# pmap wants a function with a single argument. We call pmap over ensemble
# members (initial conditions) with fixed tspan for each
@everywhere evolve_wrapper(tspan) = (u0) -> evolve_stochastic_system(model, u0, tspan, dt, alg_kwargs)

τ = 0.5
nensemble = 10
u0 = [copy(zeros(d)) for i in 1:nensemble]
k = 0.3
ϵ = 0.001
metric(y) = sum(map(abs, y)) # vs y[1]
T_a = 100.0
tspan = (0.0, T_a)
T = 50
NT = Int64.(round(T/dt))
sim = RareEventSampler{Float64}(dt, τ, tspan,u0, evolve_wrapper, nensemble, metric, k, ϵ)
run!(sim);

rmprocs([p for p in procs() if p != myid()])
