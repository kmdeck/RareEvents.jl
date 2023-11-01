#=
This solves the equation
du = (∇^2 - β) tanh(γu) - α + σ dW
for a 2D image u, of size NxN, assuming
- Dirichlet periodic boundary conditions
half indices indicate faces
- Δx = Δy = 1
- <dW(x1,y1,t) dW(x2,y2,s)> = δ(t-s)[(x1-x2)^2 + (y1-y2)^2 + 1]^(-1/2)
- σ, α, β, γ are scalars
=#
using LinearAlgebra
using ProgressBars
using Random
using StatsBase
include("./utils.jl")
include("./rhs.jl")
# Set up model
FT = Float32
# Image is NxN
N = 8
α = FT(3)
β = FT(0.1)
γ = FT(20)
σ = FT(2.0)

# Boundary condition - periodic
Γ = FT.(reshape(zeros(N^4), (N^2,N^2)))
for i1 in 1:N
    for j1 in 1:N
        k1 = (j1-1)*N+i1
        for i2 in 1:N
            for j2 in 1:N
                k2 = (j2-1)*N+i2
                Γ[k1,k2] = FT(1.0/sqrt((i1-i2)^2 + (j1-j2)^2+1))
            end
        end
    end
end
# We create correlated noise from uncorrelated noise with the Cholesky
# decomposition of Γ
# Note: for nonsquare matrices, SVD can be used. DiffEq uses SVD.
#γ = svd(Γ)
#ΓL = γ.U * Diagonal(sqrt.(γ.S))
ΓL = cholesky(Γ).L
# Allocate noise vectors up front.
W = zeros(FT, N*N)
W_corr = similar(W)
model = LudoDiffusionSDE(σ, α, β, γ, N, ΓL, W, W_corr)

deterministic_tendency! = make_deterministic_tendency(model)
stochastic_increment! = make_stochastic_increment(model)

# Initial condition
u = FT.(zeros(N^2));
# Preallocate
du = similar(u);
# Integration time, timestep, nsteps
tspan = FT.((0.0,2.5e6))
dt = FT(0.02)
nsteps = Int((tspan[2]-tspan[1])/dt)
# Saving interval, steps per interval, total # of solutions saved
dt_save = FT(1.0)
n_steps_per_save = Int(round(dt_save/dt))
savesteps = 0:n_steps_per_save:nsteps

# Preallocate the solution array
solution = zeros(FT, (N,N, Int(nsteps/n_steps_per_save)));

# Simulate
@time for i in ProgressBar(1:nsteps)
    t = tspan[1]+dt*(i-1)
    Euler_Maruyama_step!(du, u, t, deterministic_tendency!, stochastic_increment!, dt)
    if i ∈ savesteps
        save_index = Int(i/n_steps_per_save)
        solution[:,:, save_index] .= reshape(u, (N,N))
    end
end

# regularize
solution = regularization(solution)

#spinup should be zero
spinup = 15000
#n_savesteps_in_spinup = Int(spinup / dt_save)
#solution = solution[:,:,n_savesteps_in_spinup:end]
clims= (percentile(solution[:],0.1), percentile(solution[:], 99.9))
anim = convert_to_animation(solution[:,:,1:200], 1, clims)
gif(anim, string("ludo_anim","_$N", "x$N", "_$σ","_$α","_$β","_$γ",".gif"), fps = 10)

# autocorrelation time analysis
# We'll look at the center row
values = zeros(size(solution)[3],N)
for i in 1:size(solution)[3]
    values[i,:] .= solution[N ÷ 2,:,i]
end
lags = Array(1:1:(size(solution)[3]-1)) # not in units of time
ac = StatsBase.autocor(values, lags; demean = true)
mean_ac = mean(ac, dims = 2)[:]
Plots.plot(lags *dt_save, mean(ac, dims = 2)[:], label = "", ylabel = "Autocorrelation Coeff", xlabel = "Lag (time)", xlim = [0,2.5e3])
Plots.savefig(string("ludo_ac","_$N", "x$N", "_$σ","_$α","_$β","_$γ",".png"))

τ = maximum(lags[mean_ac .> 0.25])*dt_save
@show τ

#Single pixel
p528 = [solution[N ÷ 2,N ÷ 2,k] for k in 1:size(solution)[3]]
Plots.histogram(p528, label = "", ylabel = "Frequency", xlabel = "Value")
Plots.savefig(string("ludo_single_pixel","_$N", "x$N", "_$σ","_$α","_$β","_$γ",".png"))
