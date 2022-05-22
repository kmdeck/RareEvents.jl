using DifferentialEquations
using DifferentialEquations.EnsembleAnalysis
using Plots
using RareEvents
using Statistics

function deterministic_tendency!(dudt,u,p,t)
    dudt .= -p.θ *u 
end

function stochastic_tendency!(dudW,u,p,t)
    # adds additive noise p.σ dW_t
    # Note that for a Weiner process, this is draws from
    # p.σ N(0,dt)
    dudW .= p.σ
end
#=
for i in 1:8
    t0 =0.0
    W0 = 0.0
    u0 = [0.0,]
    p = (θ = 1.0, σ = 1.0)
    dW = WienerProcess(t0, W0,0.0, reseed = false)
    
    tf = 1e4
    tspan = (t0,tf)
    dt = timestep[i]
    T = 50
    Δt_save = dt
    NT = Int64.(round(T/Δt_save))
    
    prob = SDEProblem(deterministic_tendency!, stochastic_tendency!, u0, tspan,p, noise= dW)
    sol= solve(prob, dt = dt, saveat = 0:Δt_save:tf, maxiters = 1e7);# what's the deal with max_iter needing to be > nsteps?
    u = [sol.u[k][1] for k in 1:length(sol.t)-1]
    # u has the same statistics regardless of dt. that's expected!
    
    A = zeros(length(u)-NT)
    moving_average!(A,u,NT)
    σ[i] = std(A)
end
σ = zeros(8)
timestep = [0.003, 0.01,0.03,0.1,0.3,1.0,3.0,10.0]
# even dt = 1 is fine!
 0.13980975581577174
 0.14255180746457424
 0.14119076130224256
 0.14202238113746662
 0.14218398288278097
 0.13794906633002438
 0.19384396356049075
 0.2834163945433053
=#

t0 =0.0
W0 = 0.0
u0 = [0.0,]
p = (θ = 1.0, σ = 1.0)
dW = WienerProcess(t0, W0,0.0, reseed = false)

tf = 1e6
tspan = (t0,tf)
dt = 1.0
T = 50
Δt_save = dt
NT = Int64.(round(T/Δt_save))

prob = SDEProblem(deterministic_tendency!, stochastic_tendency!, u0, tspan,p, noise= dW)
sol= solve(prob, dt = dt, saveat = 0:Δt_save:tf, maxiters = 1e7);# what's the deal with max_iter needing to be > nsteps?
u = [sol.u[k][1] for k in 1:length(sol.t)-1]

A = zeros(length(u)-NT)
moving_average!(A,u,NT)

Ta = 100
ΔT = Ta-T
m = Int64.(round(ΔT/Δt_save))
# split into non-overlapping segments of length ΔT
M = Int(length(A)/m)
segment_matrix  = reshape(A, (M,m))
FT = Float64
likelihood_ratio = FT.(ones(M))
a_m = maximum(segment_matrix, dims = 2)[:]
event_magnitude, rtn, σ_rtn = return_curve(a_m, FT(ΔT), likelihood_ratio)

plot(rtn, event_magnitude, xerr = σ_rtn, xaxis = :log, ylim = [0,1.0], xlim = [1,1e15], label = "Direct Sampling", yticks = [0,0.4,0.8], xticks = [1,1e5,1e10,1e15])
plot!(legend = :bottomright)

