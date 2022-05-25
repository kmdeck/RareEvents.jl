using Plots
using RareEvents
using Statistics
include("./examples/ornstein_uhlenbeck.jl")

# Carry out integration
t0 =0.0
tf = 1e6
tspan = (t0,tf)
dt = 1.0
T = 50
Δt_save = dt
NT = Int64.(round(T/Δt_save))
alg_kwargs = (; maxiters = 1e7,)
Nsteps = Int((tf-t0)/dt +1)
max_length = (Nsteps-NT-1)/m

Ta = 100
ΔT = Ta-T
m = Int64.(round(ΔT/Δt_save))
a_m = zeros(Int(max_length*100))
a_m = reshape(a_m, (Int(max_length), 100))
for i in 1:100
    direct_solution = evolve_ornstein_uhlenbeck1D(u0, tspan, dt, alg_kwargs);
    u = [direct_solution[k][1] for k in 1:length(direct_solution)]
    t = Array(t0:dt:tf)
    # Obtain moving average
    A = zeros(length(u)-NT-1)
    moving_average!(A,u,NT)
    # split into non-overlapping segments of length ΔT
    M = Int(length(A)/m)
    segment_matrix  = reshape(A, (M,m))
    FT = Float64
    a_m[:,i] = maximum(segment_matrix, dims = 2)[:]
end


event_magnitude, rtn, σ_rtn = return_curve(a_m[:], FT(ΔT),  FT.(ones(length(a_m[:]))))
)
plot(event_magnitude, rtn, ribbon = σ_rtn, yaxis = :log, ylim = [1, 1e10], yticks = [1,1e5,1e10], xticks = [0,0.4,0.8], xlim = [0, 1], label = "Direct")
plot!(legend = :bottomright)
