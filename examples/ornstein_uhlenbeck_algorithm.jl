using Revise
using DifferentialEquations
using Plots
using RareEvents
using Statistics
using SpecialFunctions
function deterministic_tendency!(dudt,u,p,t)
    dudt .= -p.θ *u 
end

function stochastic_tendency!(dudW,u,p,t)
    dudW .= p.σ
end
t0 = 0.0
W0 = 0.0
p = (θ = 1.0, σ = 1.0)
dW = WienerProcess(t0, W0,0.0)#, reseed = false)
dt = 0.1
τ = 0.5
Nτ = Int(round(τ/dt))
T_a = 100.0
N = 3000
d = 1
u0 = zeros(d)
k = 0.5
ϵ = 0.0
# Note - your score function has to handle the dimensionality of your state vector
score_function(x; k = k, dt = dt) = exp.(k .*sum(x[2:end]+x[1:end-1])/2.0*dt)
# Note that the tspan and u0 will be replaced after each interval τ according to algorithm; 
sde_ensemble_prob = SDEProblem(deterministic_tendency!, stochastic_tendency!, u0, (0,1),p, noise= dW)
sim = RareEventSampler{Float64}(dt, d, (0.0, T_a), N, Nτ, sde_ensemble_prob, score_function, ϵ);

run!(sim, u0);
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

event_magnitude, rtn, σ_rtn = return_curve(a_m, T_a-T, likelihood_ratio);

plot!(rtn, event_magnitude, xerr = σ_rtn, xaxis = :log, ylim = [0,1.0], xlim = [1,1e15], label = "k = 0.5, N=3000", yticks = [0,0.4,0.8],msc = :auto)


#=
σ = 0.142
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

=#
