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
ks = [0.2,0.3,0.4,0.5,0.6, 0.7]
ϵ = 0.0
# Note - your score function has to handle the dimensionality of your state vector
a_m = zeros(length(ks), N)
Nsteps = (Int(round((T_a-t0)/dt))+1)
μ = zeros(length(ks), Nsteps)
σ = zeros(length(ks), Nsteps)
lrs = zeros(length(ks), N)
# What happens if the likelihood ratio is >1? More common with original than with algorithm. Is this ok?
for i in 1:length(ks)
    k = ks[i]
    score_function(x; k = k, dt = dt) = exp.(k .*sum(x[2:end]+x[1:end-1])/2.0*dt)
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
    a_m[i,:] .= maximum(moving_average_matrix, dims = 2)[:]
    lrs[i,:] .= likelihood_ratio
    m, s = ensemble_statistics(sim.ensemble,1)
    μ[i,:] .= m
    σ[i,:] .= s
end
t = Array(1:1:Nsteps)./Nsteps .*T_a
plot1 = plot(t,μ[1,:]; ribbon = σ[1,:], label = string("k = ", string(ks[1])))
plot!(t,μ[end,:]; ribbon = σ[end,:],label = string("k = ", string(ks[end])))
plot!(xlabel = "time", ylabel = "x")


plot2 = plot(xaxis = :log, ylim = [0,1.0], xlim = [1,1e12], yticks = [0,0.4,0.8],msc = :auto, xticks = [1,1e5,1e10], legend = :bottomright,
             xlabel = "return time", ylabel = "magnitude") 
for i in 1:(length(ks)-1)
    k = ks[i]
    event_magnitude, rtn, σ_rtn = return_curve(a_m[i,:], T_a-T, lrs[i,:]);
    plot!(rtn, event_magnitude, label = string("k = ", string(ks[i])))
end
event_magnitude, rtn, σ_rtn = return_curve(reduce(vcat,a_m[1:5,:]), T_a-T, reduce(vcat,lrs[1:5,:]));
plot!(rtn, event_magnitude, label = "all", color = "black")
σ0 = 0.142
# If Y = moving average, Y~ N(0, σ^2) = P_0
μ_k = k*σ0^2.0*T # Mean of moving average distribution resulting from algorithm. 
analytic(a, σ, ΔT, μ) = 2.0*ΔT/SpecialFunctions.erfc((a-μ)/√2/σ)
analytic_rtn = analytic.(event_magnitude,σ0, T_a-T, 0.0)
plot!(analytic_rtn, event_magnitude, color = "blue", label = "analytic estimate")
plot(plot1, plot2)
savefig("./return_time_with_k.png")


