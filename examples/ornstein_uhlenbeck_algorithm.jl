using Revise
using Plots
using RareEvents
using Statistics
using SpecialFunctions
using Distributed
include("./examples/ornstein_uhlenbeck.jl")

dt = 0.1
# pmap wants a function with a single argument. We iterate over ensemble
# members (initial conditions) with fixed tspan for each
alg_kwargs = ();
@everywhere evolve_wrapper(tspan) = (u0) -> evolve_ornstein_uhlenbeck1D(u0, tspan, dt; alg_kwargs...)

n_processes = Sys.CPU_THREADS
addprocs(n_processes)


Nτ = 5 # Nτ*dt = τ, the resample time
T_a = 100.0 # total integration time
N = 1000
d = 1
u0 = zeros(d) # No harm in starting at the same IC here for all ensemble members
k = 0.3
ϵ = 0.0
score_function(x; k = k, dt =dt) = exp.(k .*sum(x[2:end]+x[1:end-1])/2.0*dt)

sim = RareEventSampler{Float64}(dt, u0, (0.0, T_a), N, Nτ,evolve_wrapper, score_function, ϵ);

run!(sim);
T = 50
NT = Int64.(round(T/dt))
moving_average_matrix = zeros((N, length(sim.ensemble[1])-NT))
moving_average_row = zeros(length(sim.ensemble[1])-NT)
p_n1 = zeros(N)
λ = sum(log.(sim.R))/T_a
po_over_pk(trajectory, dt, T_a, k, λ) = 1.0/N*exp(T_a*λ)/score_function(trajectory)


for i in 1:N
    sol = [sim.ensemble[i][j][1] for j in 1:length(sim.ensemble[i])]
    moving_average_row .= moving_average_matrix[i,:]
    moving_average!(moving_average_row, sol, NT)
    moving_average_matrix[i,:] .= moving_average_row
    p_n1[i] = po_over_pk(sol, dt, T_a, k, λ)
end

event_magnitude, rtn = return_curve(moving_average_matrix, T_a-T, p_n1);
plot()
plot!(log10.(rtn), event_magnitude, ylim = [0,1.0], xlim = [0,15], color = "red", label = "k = 0.3, N=3e3", yticks = [0,0.4,0.8])

### Analytic estimate does not work yet
σ = 0.142 # Computed standard deviation of moving average of x(t), from a direct integration,
# over time T=50. It has a mean of 0. This is independent of dt if dt is small enough.
# If Y = moving average, Y~ N(0, σ^2) = P_0
μ_k = k*σ^2.0*T # Mean of moving average distribution resulting from algorithm. 
# The standard dev is the same. so Y_k ∼P_k = N(μ_k, σ^2).

# So, we can check that the moving_average_matrix has a mean and standard dev of μ_k and σ:
mean(moving_average_matrix[:])
std(moving_average_matrix[:])

# So far so good...but then:
# What we actually are computing return curves for is the maximum of q samples
# of Y, where q = (T_a-T) /dt


# 2x Number exceeding x of a Gaussian with zero mean and unit standard dev:
# erfc(x) = 2/√π*∫_x^∞ e^(-t^2) dt
# 2x what we want, and √2 * σ = 1
 # and nonzero mean
# Expected for Y = moving average
analytic(a, σ, ΔT, μ) = ΔT/(SpecialFunctions.erfc((a-μ)/√2/σ)/(2))
# Should this also apply for event_magnitude, the max of the moving average?
# E(max over q samples) = σ √(2log(q))+μ_k
analytic_rtn = analytic.(event_magnitude, std(event_magnitude), T_a-T, mean(event_magnitude))

plot!(log10.(analytic_rtn), event_magnitude, color = "blue", label = "analytic estimate")
# Think more about:  does this hold even though we are looking
# event mangitude = the max of the moving average over q samples?
# The analytic expectation is FOR the moving average, not the max?
# P_k(max(Y) = η) ∝ (q choose 1 ) F_k(η)^(q-1) P_k(η) ?
# where F_k(η) is the cumulative distribution ∫_-∞^η P_k(s) ds
