using Plots
using RareEvents
using Statistics
include("./examples/ornstein_uhlenbeck.jl")

# Carry out integration
t0 =0.0
tf = 1e4
tspan = (t0,tf)
dt = 1.0
T = 50
Δt_save = dt
NT = Int64.(round(T/Δt_save))
alg_kwargs = (; maxiters = 1e7,)
direct_solution = evolve_ornstein_uhlenbeck1D(u0, tspan, dt, alg_kwargs)
u = [direct_solution[k][1] for k in 1:length(direct_solution)]
t = Array(t0:dt:tf)

# Obtain moving average
A = zeros(length(u)-NT-1)
moving_average!(A,u,NT)

# Compute return curve
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
