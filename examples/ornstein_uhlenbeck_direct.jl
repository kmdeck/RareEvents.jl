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
p_m = FT.(ones(M)./M)
event_magnitude, rtn = return_curve(segment_matrix, FT(ΔT), p_m)

plot(log10.(rtn), event_magnitude, ylim = [0,0.8], xlim = [0,7], color = "blue", label = "dt =1.0, N_MA = 50, Direct")
plot!(legend = :bottomright)
