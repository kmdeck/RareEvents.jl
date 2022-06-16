using Plots
using RareEvents
using Statistics
using DelimitedFiles
include("examples/ornstein_uhlenbeck.jl")
θ = 1.0
σ = 1.0
d = 1
model = OrnsteinUhlenbeck{Float64}(θ, σ, d)

# Carry out integration
t0 =0.0
tf = 1.8e6
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
max_length = Int((Nsteps-NT-1)/m)
iters = 30

u0=[0.0,]
# Vector of length iters. Each element is an nensemble x d matrix
a_m = [zeros(Float64, max_length, d) for k in 1:iters]
a_range = Array(0.0:0.03:1.0)
Na = zeros(length(a_range))
for i in 1:iters
    println(iters)
    direct_solution = evolve_stochastic_system(model, u0, tspan, dt, alg_kwargs);
    t = Array(t0:dt:tf)
    # Obtain moving average
    A = zeros(Float64,length(u)-NT-1,1)
    moving_average!(A,direct_solution,NT)
    # split into non-overlapping segments of length ΔT
    Na .+= N_event(A, 1.0, a_range)
    a_m[i] = maximum(reshape(A, (max_length, m)), dims = 2)
end

path = string("examples/ornstein_uhlenbeck_output/direct/"
writedlm(string(path, "N.csv"), Na)
writedlm(string(path, "maxes.csv"), a_m)
