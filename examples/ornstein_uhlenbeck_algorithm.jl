using Revise
using Plots
using RareEvents
using Statistics
using SpecialFunctions
using Distributed
using DelimitedFiles
n_processes = Sys.CPU_THREADS
addprocs(n_processes)

include("examples/ornstein_uhlenbeck.jl")

dt = 0.1
# pmap wants a function with a single argument. We iterate over ensemble
# members (initial conditions) with fixed tspan for each
alg_kwargs = ();
@everywhere evolve_wrapper(tspan) = (u0) -> evolve_ornstein_uhlenbeck1D(u0, tspan, dt, alg_kwargs)



Nτ = 5 # Nτ*dt = τ, the resample time
T_a = 100.0 # total integration time
N = 3000
d = 1
u0 = zeros(d) # No harm in starting at the same IC here for all ensemble members
k = 0.5
ϵ = 0.05
score_function(x; k = k, dt =dt) = exp.(k .*sum(x[2:end]+x[1:end-1])/2.0*dt)

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
a_m = maximum(moving_average_matrix, dims = 2)[:]


event_magnitude, rtn, σ_rtn= return_curve(a_m, T_a-T, likelihood_ratio);
plot1 = plot()

plot!(event_magnitude, rtn, ribbon = σ_rtn, yaxis = :log, ylim = [1, 1e10], yticks = [1,1e5,1e10], xticks = [0,0.4,0.8], xlim = [0, 1], label = "")

direct_a_m = readdlm("direct_1e8_a_m.csv")
direct_event_magnitude, direct_rtn, direct_σ_rtn= return_curve(direct_a_m[:], T_a-T, ones(length(direct_a_m[:])));

plot!(direct_event_magnitude, direct_rtn, ribbon = direct_σ_rtn, yaxis = :log, ylim = [1, 1e10], yticks = [1,1e5,1e10], xticks = [0,0.4,0.8], xlim = [0, 1], label = "Direct")
plot!(legend = :bottomright)

μ, σ = ensemble_statistics(sim.ensemble, 1)
plot2 = plot(0.0:dt:T_a,μ,grid=false,ribbon=σ,fillalpha=.5)
plot(plot1, plot2)
#=
### Analytic estimate does not work yet
σ = 0.142 # Computed standard deviation of moving average of x(t), from a direct integration,
# over time T=50. It has a mean of 0. This is independent of dt if dt is small enough.
# If Y = moving average, Y~ N(0, σ^2) = P_0
μ_k = k*σ^2.0*T # Mean of moving average distribution resulting from algorithm. 
# The standard dev is the same. so Y_k ∼P_k = N(μ_k, σ^2).
# So, we can check that the moving_average_matrix has a mean and standard dev of μ_k and σ:
mean(moving_average_matrix[:])
std(moving_average_matrix[:])
# We want return periods for moving average but under P_0
# erfc(x) = 2/√π*∫_x^∞ e^(-t^2) dt

analytic(a, σ, ΔT, μ) = 2.0*ΔT/SpecialFunctions.erfc((a-μ)/√2/σ)
analytic_rtn = analytic.(event_magnitude,σ, T_a-T, 0.0)
plot!(log10.(analytic_rtn), event_magnitude, color = "blue", label = "analytic estimate")

# Think more about:  does this hold even though we are looking
# event mangitude = the max of the moving average over q samples?
# The analytic expectation is FOR the moving average, not the max?
# P_k(max(Y) = η) ∝ (q choose 1 ) F_k(η)^(q-1) P_k(η) ?
# where F_k(η) is the cumulative distribution ∫_-∞^η P_k(s) ds
=#
