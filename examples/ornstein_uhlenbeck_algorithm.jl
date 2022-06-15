using Revise
using RareEvents
using Statistics
using DelimitedFiles

include("examples/ornstein_uhlenbeck.jl")
n_processes = Sys.CPU_THREADS
addprocs(n_processes)

dt = 0.1
alg_kwargs = ();
FT = Float64
θ = [1.0 0.0; 0.0 1.0]
σ = [1.0 0.0; 0.0 1.0]
d = 2
model = OrnsteinUhlenbeck{FT}(θ, σ, d)
@everywhere evolve_wrapper(tspan) = (u0) -> evolve_stochastic_system(model, u0, tspan, dt, alg_kwargs)

τ = 0.5
nensemble = 600
u0 = [copy(zeros(d)) for i in 1:nensemble]
k = 0.3
ϵ = 0.001
metric(y) = y[1]#mean(map(abs, y))
T_a = 100.0
tspan = (0.0, T_a)

iters = 10
# Vector of length iters. Each element is an nensemble x d matrix
a_m = [zeros(FT, nensemble, d) for k in 1:iters]
# vector of length iters. Each element is a vector of length nensemble
lr_matrix = [zeros(nensemble) for k in 1:iters]
a_range = Array(0.0:0.03:1.0)
# Vector of length iters. Each element is a length(a_range)-1 x d matrix
Na = [zeros(FT,length(a_range)-1,d) for k in 1:iters]
for iter in 1:iters
    println(iter)
    sim = RareEventSampler{Float64}(dt, τ, tspan,u0, evolve_wrapper, nensemble, metric, k, ϵ)
    
    run!(sim);
    T = 50
    NT = Int64.(round(T/dt))
    moving_average_row = Matrix{FT}(undef, length(sim.ensemble[1])-NT,d)
    lr_vector = zeros(nensemble)
    λ = sum(log.(sim.weight_norm))/T_a    
    
    for i in 1:nensemble
        trajectory = [sim.ensemble[i][j] for j in 1:length(sim.ensemble[i])]
        moving_average!(moving_average_row, trajectory, NT)
        lr = likelihood_ratio(sim, trajectory, λ, T_a)
        lr_vector[i] = lr
        Na[iter] .+= N_event(moving_average_row, lr, a_range)
        a_m[iter][i,:] .= maximum(moving_average_row,dims = 1)[:]
    end

    lr_matrix[iter] .= lr_vector
end

rmprocs([p for p in procs() if p != myid()])

writedlm("k_03_bootstrap_Na_refactored.csv", Na)
writedlm("k_03_bootstrap_lr_refactored.csv", lr_matrix)
writedlm("k_03_bootstrap_refactored.csv", a_m)
