using HDF5
using LinearAlgebra
using Random
FT = Float32
N = 32
u_bc = FT(0)
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
function f!(du, u, t::FT; N=N, u_bc = u_bc) where{FT}
    du .= FT(0)
    for i in 1:N
        for j in 1:N
            k = (j-1)*N+i
            k_ip1 = k+1
            k_im1 = k-1
            k_jp1 = k + N
            k_jm1 = k - N  
            if i == N          
                du[k] += FT(1/2)*((u[k_im1] - u[k]) - (u[k] - u_bc))
            elseif i ==1
                du[k] += FT(1/2)*((u_bc - u[k]) - (u[k] - u[k_ip1]))
            else
                du[k] += FT(1/2)*((u[k_im1] - u[k]) - (u[k] - u[k_ip1]))
            end
            if j == N
                du[k] += FT(1/2)*((u_bc - u[k]) - (u[k] - u[k_jm1]))
            elseif j ==1
                du[k] += FT(1/2)*((u[k_jp1] - u[k]) - (u[k] - u_bc))
            else
                du[k] += FT(1/2)*((u[k_jp1] - u[k]) - (u[k] - u[k_jm1]))
            end
        end
    end
end
ΓL = cholesky(Γ).L
W = zeros(FT, N*N)
σ = FT(1.0)
W_corr = similar(W)
function g!(du, u, t; ΓL = ΓL, N = N, W = W, σ = σ, W_corr = W_corr)
    randn!(W)
    mul!(W_corr,ΓL, W)
    du .= σ .* W_corr
end

function Euler_Maruyama_step!(du,u,t,f!,g!, dt)
    du .= FT(0)
    f!(du,u,t)
    u .+=  du .* dt
    du .= FT(0)
    g!(du,u,t)
    u .+=  sqrt(dt) .* du
    return u
end


u = FT.(zeros(N^2));
du = similar(u);
    
# TODO: Run for approximately 125*1e4 = 1.25e6, saving output every 25-40
tspan = FT.((0.0,3e6))
dt = FT(0.1)
dt_save = FT(30)
n_steps_per_save = Int(round(dt_save/dt))
nsteps = Int((tspan[2]-tspan[1])/dt)
savesteps = 0:n_steps_per_save:nsteps

solution = zeros(FT, (N,N, Int(nsteps/n_steps_per_save)));

@time for i in 1:nsteps
    t = tspan[1]+dt*(i-1)
    Euler_Maruyama_step!(du, u, t, f!, g!, dt)
    if i ∈ savesteps
        save_index = Int(i/n_steps_per_save)
        solution[:,:, save_index] .= reshape(u, (N,N))
    end
end
fname = "./grf.hdf5"
fid = h5open(fname, "w")
fid["res_32x32"] = solution
close(fid)
