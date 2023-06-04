"""
This solves the equation
du = 1/2  ∇^2 u + σ dW
for a 2D image u, of size NxN assuming
- Dirichlet boundary conditions:
u(1/2,:) = u(N+1/2,:) = u(:, 1/2) = u(:, N+1/2), where
half indices indicate faces
- Δx = Δy = 1
- <dW(x1,y1,t) dW(x2,y2,s)> = δ(t-s)[(x1-x2)^2 + (y1-y2)^2 + 1]^(-1/2)
- σ is a scalar
"""
#TODO: figure out what value of σ we want;
#Understand correlation time better
function convert_to_animation(x, time_stride, clims)
           init_frames = size(x)[3]
           x = x[:,:,1:time_stride:init_frames]
           frames = size(x)[3]
           animation = @animate for i = 1:frames
                   heatmap(
                       x[:,:,i],
                       xaxis = false, yaxis = false, xticks = false, yticks = false,
                       clims = clims
                   )
           end
           return animation
       end

using HDF5
using LinearAlgebra
using Random
FT = Float32
# Image is NxN
N = 32
# Boundary condition - Dirichlet, constant on each boundary
u_bc = FT(0)
# Correlation matrix
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

# Heat equation deterministic tendency: du/dt = 1/2∇^2 u
function f!(du, u, t::FT; N=N, u_bc = u_bc) where {FT}
    du .= FT(0)
    for i in 1:N
        for j in 1:N
            k = (j-1)*N+i
            k_ip1 = k+1
            k_im1 = k-1
            k_jp1 = k + N
            k_jm1 = k - N
            # At the boundary, we have u_bc on the face
            # while u[k] is at the cell center. -> Δx -> Δx/2
            if i == N          
                du[k] += FT(1/2)*((u[k_im1] - u[k]) - (u[k] - u_bc)/2)
            elseif i ==1
                du[k] += FT(1/2)*((u_bc - u[k])/2 - (u[k] - u[k_ip1]))
            else
                du[k] += FT(1/2)*((u[k_im1] - u[k]) - (u[k] - u[k_ip1]))
            end
            if j == N
                du[k] += FT(1/2)*((u_bc - u[k])/2 - (u[k] - u[k_jm1]))
            elseif j ==1
                du[k] += FT(1/2)*((u[k_jp1] - u[k]) - (u[k] - u_bc)/2)
            else
                du[k] += FT(1/2)*((u[k_jp1] - u[k]) - (u[k] - u[k_jm1]))
            end
        end
    end
end

# Stochastic increment du = σ dW
σ = FT(0.1)
# We create correlated noise from uncorrelated noise with the Cholesky
# decomposition of Γ
ΓL = cholesky(Γ).L
# Allocate noise vectors up front.
W = zeros(FT, N*N)
W_corr = similar(W)
function g!(du, u, t; ΓL = ΓL, N = N, W = W, σ = σ, W_corr = W_corr)
    randn!(W)
    mul!(W_corr,ΓL, W)
    du .= σ .* W_corr
end

# Time stepping scheme for a single step of length Δt
function Euler_Maruyama_step!(du,u,t,f!,g!, dt)
    # Deterministic step
    du .= FT(0)
    f!(du,u,t)
    u .+=  du .* dt
    # Stochastic step
    du .= FT(0)
    g!(du,u,t)
    u .+=  sqrt(dt) .* du
    return u
end

# Initial condition
u = FT.(zeros(N^2));
# Preallocate
du = similar(u);
# Integration time, timestep, nsteps
# Based on an estimate of the autocorrelation time of ∼150
# And wanting 1e4 autocorrelation times, with ~5 samples each.
tspan = FT.((0.0,1.5e3))
dt = FT(0.1)
nsteps = Int((tspan[2]-tspan[1])/dt)
# Saving interval, steps per interval, total # of solutions saved
dt_save = FT(15)
n_steps_per_save = Int(round(dt_save/dt))
savesteps = 0:n_steps_per_save:nsteps

# Preallocate the solution array
solution = zeros(FT, (N,N, Int(nsteps/n_steps_per_save)));

# Simulate
@time for i in 1:nsteps
    t = tspan[1]+dt*(i-1)
    Euler_Maruyama_step!(du, u, t, f!, g!, dt)
    if i ∈ savesteps
        save_index = Int(i/n_steps_per_save)
        solution[:,:, save_index] .= reshape(u, (N,N))
    end
end


clims= (minimum(minimum(solution[:])),maximum(maximum(solution[:])))
anim = convert_to_animation(solution, 1, clims)
gif(anim, "anim.gif", fps = 1)


#Save
fname = "./grf.hdf5"
fid = h5open(fname, "w")
fid["res_32x32"] = solution
close(fid)


#=
# autocorrelation time analysis

values = zeros(size(solution)[3],N)
for i in 1:size(solution)[3]
    values[i,:] .= solution[16,:,i]
end
p528 = [solution[16,16,k] for k in 1:size(solution)[3]]
Plots.plot(1:size(solution)[3], p528)
lags = Array(1:1:(size(solution)[3]-1)) # not in units of time
ac = StatsBase.autocor(values, lags; demean = true)
mean_ac = mean(ac, dims = 2)[:]

Plots.plot(lags *dt_save, mean(ac, dims = 2)[:])
τ = maximum(lags[mean_ac .> 0.4])*dt_save
n_τ = (tspan[2]-tspan[1])/τ
=#
