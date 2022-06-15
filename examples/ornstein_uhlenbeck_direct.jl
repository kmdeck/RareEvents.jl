using Plots
using RareEvents
using Statistics
using DelimitedFiles
include("ornstein_uhlenbeck.jl")
θ = 1.0
σ = 1.0
d = 1
model = OrnsteinUhlenbeck{Float64}(θ, σ, d)

# Carry out integration
t0 =0.0
tf = 1e6
tspan = (t0,tf)
dt = 0.1
T = 50
Δt_save = dt
NT = Int64.(round(T/Δt_save))
alg_kwargs = (; maxiters = 1e7,)
Nsteps = Int((tf-t0)/dt +1)
Ta = 100
ΔT = Ta-T
m = Int64.(round(ΔT/Δt_save))
max_length = (Nsteps-NT-1)/m

u0=[0.0,]
a_m = zeros(Int(max_length*30))
a_m = reshape(a_m, (Int(max_length), 30))
a_range = Array(0.0:0.03:1.0)
Na = zeros(length(a_range))
for i in 1:30
    direct_solution = evolve_stochastic_system(model, u0, tspan, dt, alg_kwargs);
    u = [direct_solution[k][1] for k in 1:length(direct_solution)]
    t = Array(t0:dt:tf)
    # Obtain moving average
    A = zeros(length(u)-NT-1)
    moving_average!(A,u,NT)
    # split into non-overlapping segments of length ΔT
    M = Int(length(A)/m)
    segment_matrix  = reshape(A, (M,m))
    Na .+= N_event(segment_matrix, ones(size(segment_matrix)[1]), a_range)
    a_m[:,i] = maximum(segment_matrix, dims = 2)[:]
end


writedlm("direct_1e7_a_m_dt_01_refactored.csv", a_m)
writedlm("direct_1e7_Na_dt_01_refactored.csv", Na)
