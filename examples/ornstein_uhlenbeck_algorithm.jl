using Revise
using RareEvents
using Statistics
using DelimitedFiles


include("ornstein_uhlenbeck.jl")
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

iters = 30
a_m = zeros(nensemble*iters)
a_m = reshape(a_m, (nensemble,iters))
lr_matrix = similar(a_m)
a_range = Array(0.0:0.03:1.0)
Na = zeros(length(a_range)*iters)
Na = reshape(Na, (length(a_range), iters))
state_index = 1 # index to look at extremes of...needs to generalize
for iter in 1:iters
    println(iter)
    sim = RareEventSampler{Float64}(dt, τ, tspan,u0, evolve_wrapper, nensemble, metric, k, ϵ)
    
    run!(sim);
    T = 50
    NT = Int64.(round(T/dt))
    moving_average_matrix = zeros((nensemble, length(sim.ensemble[1])-NT))
    moving_average_row = zeros(length(sim.ensemble[1])-NT)
    lr_vector = zeros(nensemble)
    λ = sum(log.(sim.weight_norm))/T_a    
    
    for i in 1:nensemble
        trajectory = [sim.ensemble[i][j][state_index] for j in 1:length(sim.ensemble[i])]
        moving_average_row .= moving_average_matrix[i,:]
        moving_average!(moving_average_row, trajectory, NT)
        moving_average_matrix[i,:] .= moving_average_row
        lr_vector[i] = likelihood_ratio(sim, trajectory, λ, T_a)
    end
    Na[:,iter] .+= N_event(moving_average_matrix, lr_vector, a_range)

    a_m[:,iter] = maximum(moving_average_matrix, dims = 2)[:]
    lr_matrix[:,iter] = lr_vector
end
writedlm("k_03_bootstrap_Na_refactored.csv", Na)
writedlm("k_03_bootstrap_lr_refactored.csv", lr_matrix)
writedlm("k_03_bootstrap_refactored.csv", a_m)
