#=
This solves the equation
du = θ ∇^2 u dt + σ dW
for a 1D variable u, of size N, assuming
- Dirichlet boundary conditions:
u(1/2) = u(N+1/2), where
half indices indicate faces
- Δx = 1
- <dW(x1,t) dW(x2,s)> = δ(t-s)[(x1-x2)^2 + 1]^(-1/2)
- σ, θ are scalars
=#
using LinearAlgebra
using Random
using HDF5
include("./utils.jl")
include("./rhs.jl")
# Set up model
FT = Float32
# size of grid is N
N = 64
θ = FT(30.0)

# Boundary condition - Dirichlet, constant on each boundary
u_bc = FT(0)
σ = FT(1.0)
Γ = FT.(reshape(zeros(N^2), (N,N)))
for k1 in 1:N
    for k2 in 1:N
        Γ[k1,k2] = FT(1.0/sqrt((k1-k2)^2 + 1))
    end
end

# We create correlated noise from uncorrelated noise with the Cholesky
# decomposition of Γ
# Note: for nonsquare matrices, SVD can be used. DiffEq uses SVD.
#γ = svd(Γ)
#ΓL = γ.U * Diagonal(sqrt.(γ.S))
ΓL = cholesky(Γ).L
# Allocate noise vectors up front.
W = zeros(FT, N)
W_corr = similar(W)
model = LinearDiffusion1dSDE(σ,θ, N, Dirichlet(u_bc), ΓL, W, W_corr)

deterministic_tendency! = make_deterministic_tendency(model)
stochastic_increment! = make_stochastic_increment(model)

# Initial condition
u = FT.(zeros(N));
# Preallocate
du = similar(u);
# Integration time, timestep, nsteps
tspan = FT.((0.0,1.92e6))
dt = FT(0.01)
nsteps = Int((tspan[2]-tspan[1])/dt)
# Saving interval, steps per interval, total # of solutions saved
dt_save = FT(1.0)
n_steps_per_save = Int(round(dt_save/dt))
savesteps = 0:n_steps_per_save:nsteps

# Preallocate the solution array
solution = zeros(FT, (N, Int(nsteps/n_steps_per_save)));

# Simulate
@time for i in 1:nsteps
    t = tspan[1]+dt*(i-1)
    Euler_Maruyama_step!(du, u, t, deterministic_tendency!, stochastic_increment!, dt)
    if i ∈ savesteps
        save_index = Int(i/n_steps_per_save)
        solution[:, save_index] .= u
    end
end
#Save
fname = "./grf_1d.hdf5"
fid = h5open(fname, "w")
fid[string("N_64_$σ","_$θ","_$dt","_$dt_save")] = solution
close(fid)
