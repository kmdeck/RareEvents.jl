using Revise
using Plots
using RareEvents
using Statistics
using SpecialFunctions
using Distributed
using DelimitedFiles
n_processes = Sys.CPU_THREADS
addprocs(n_processes)

include("ornstein_uhlenbeck.jl")

dt = 0.1
# pmap wants a function with a single argument. We iterate over ensemble
# members (initial conditions) with fixed tspan for each
alg_kwargs = ();
@everywhere evolve_wrapper(tspan) = (u0) -> evolve_ornstein_uhlenbeck1D(u0, tspan, dt, alg_kwargs)



Nτ = 5 # Nτ*dt = τ, the resample time
T_a = 100.0 # total integration time
N = 600
d = 1
u0 = zeros(d) # No harm in starting at the same IC here for all ensemble members
k = 0.3
ϵ = 0.05
score_function(x; k = k, dt =dt) = exp.(k .*sum(x[2:end]+x[1:end-1])/2.0*dt)
a_m = zeros(N*30)
a_m = reshape(a_m, (N,30))
lr_matrix = similar(a_m)
for iter in 1:30
    sim = RareEventSampler{Float64}(dt, u0, (0.0, T_a), N, Nτ,evolve_wrapper, score_function, ϵ);
    
    run!(sim);
    T = 50
    NT = Int64.(round(T/dt))
    moving_average_matrix = zeros((N, length(sim.ensemble[1])-NT))
    moving_average_row = zeros(length(sim.ensemble[1])-NT)
    likelihood_ratio = zeros(N)
    λ = sum(log.(sim.R))/T_a
    lr(trajectory, dt, T_a, k, λ) = exp(T_a*λ)/score_function(trajectory)
    
    
    for i in 1:N
        sol = [sim.ensemble[i][j][1] for j in 1:length(sim.ensemble[i])]
        moving_average_row .= moving_average_matrix[i,:]
        moving_average!(moving_average_row, sol, NT)
        moving_average_matrix[i,:] .= moving_average_row
        likelihood_ratio[i] = lr(sol, dt, T_a, k, λ)
    end
    a_m[:,iter] = maximum(moving_average_matrix, dims = 2)[:]
    lr_matrix[:,iter] = likelihood_ratio
end
writedlm("k_05_bootstrap_lr.csv", lr_matrix)
writedlm("k_05_bootstrap.csv", a_m)

#μ, σ = ensemble_statistics(sim.ensemble, 1)
#plot2 = plot(0.0:dt:T_a,μ,grid=false,ribbon=σ,fillalpha=.5)
