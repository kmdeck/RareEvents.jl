# This code computes the response functions numerically and using the linear approximation

using LinearAlgebra
using ProgressBars
using Random
using HDF5
using StatsBase
using Statistics
using Distributions

include("./utils.jl")
include("./rhs.jl")

FT = Float64

N = 8
α = FT(0.2)
β = FT(0.5)
γ = FT(0.1)
σ = FT(1.0)
T = 10000
tau = 20
dt = FT(0.1)
dt_save = dt
n_tau = Int(tau/dt_save)

function get_neighbors_avg(A, ind, D)
    N = size(A, 1)
    idx = CartesianIndices(A)[(ind-1)*N+ind]
    i, j = idx[1], idx[2]
    indices = [(i+D, j), (i-D, j), (i, j+D), (i, j-D)]
    periodic_indices = [(mod(x[1]-1, N)+1, mod(x[2]-1, N)+1) for x in indices]
    avg = mean([A[idx...] for idx in periodic_indices])
    return avg
end

function overall_avg(A, D)
    N = size(A, 1)
    avg_values = [get_neighbors_avg(A, ind, D) for ind in 1:N]
    return mean(avg_values)
end

function mean_response(response; n_dist=4, tau=n_tau)
    mean_res = zeros(n_dist,tau)
    for i in 1:n_dist
        for j in 1:tau
            mean_res[i,j] = overall_avg(response[:,:,j],i-1)
        end
    end
    return mean_res
end

Γ = FT.(reshape(zeros(N^4), (N^2,N^2)))
for i1 in 1:N
    for j1 in 1:N
        k1 = (j1-1)*N+i1
        for i2 in 1:N
            for j2 in 1:N
                k2 = (j2-1)*N+i2
                Γ[k1,k2] = FT(1/sqrt(min(abs(i1-i2),N-abs(i1-i2))^2 + min(abs(j1-j2),N-abs(j1-j2))^2+1))
            end
        end
    end
end
ΓL = cholesky(Γ).L
W = zeros(FT, N*N)
W_corr = similar(W)
Model = LudoDiffusionSDE(σ, α, β, γ, N, Periodic(), ΓL, W, W_corr)

function trajectory(u, tspan, dt, dt_save, seed; model = Model)
    Random.seed!(abs(seed))
    deterministic_tendency! = make_deterministic_tendency(model)
    stochastic_increment! = make_stochastic_increment(model)
    du = similar(u)
    nsteps = Int((tspan[2]-tspan[1])/dt)
    n_steps_per_save = Int(round(dt_save/dt))
    savesteps = 0:n_steps_per_save:nsteps - n_steps_per_save
    solution = zeros(FT, (N^2, Int(nsteps/n_steps_per_save)))
    solution[:, 1] .= reshape(u, (N^2,))
    for i in 1:nsteps
        t = tspan[1]+dt*(i-1)
        Euler_Maruyama_step!(du, u, t, deterministic_tendency!, stochastic_increment!, dt)
        if i ∈ savesteps
            save_index = Int(i/n_steps_per_save)
            solution[:, save_index+1] .= reshape(u, (N^2,))
        end
    end
    return solution
end

tspan = FT.((0.0,T))
u0 = 2*rand(FT, N^2).-1

trj = trajectory(u0, tspan, dt, dt_save, rand(Int))

pl1 = plot(trj[1,1:10:10000],title="time series",xlabel="time",linewidth=3,label="")

lags = [0:1000...]
acf = autocor(trj[1,:],lags)
t_decorr = 0.
for i in eachindex(acf)
    if acf[i] < 0.1
        t_decorr = i*dt
        break
    end
end
pl2 = plot(acf[1:Int(floor(t_decorr/dt_save))],title="autocorrelation",xlabel="time",linewidth=3)

mu = mean(trj)
sig = std(trj)
dist = Normal(mu, sig)
xax = collect(-6sig:0.01:6sig)  
yax = map(v -> pdf(dist, v), xax)

pl3 = plot(xax, yax, label="normal distribution", linewidth=3, title="distribution of the time series", legend=:topleft)
pl3 = stephist!(reshape(trj,(N^2*size(trj)[2])),normalize=:pdf,label="histogram",linewidth=3)

cum = zeros(4)
for i in 1:4
    cum[i] = cumulant(reshape(trj,(N^2*size(trj)[2])),i)
end
pl4 = scatter(cum, label="cumulants", title="cumulants", xlabel="order")

pl = plot(pl1,pl2,pl3,pl4,layout=(2,2),size=(1000,700))
display(pl)
savefig(pl, "trj_gamma_$γ.png")

trj_t = transpose(trj) .- mean(trj) 
invC0 = inv(cov(trj_t))
responseL = zeros(N^2,N^2,n_tau)
for i in ProgressBar(1:n_tau)
    responseL[:,:,i] = cov(trj_t[i:end,:],trj_t[1:end-i+1,:]) * invC0
end
responseL_mean = mean_response(responseL)

n_ens = 1000
eps = 0.01
responseN_ens = zeros(N^2,n_tau,n_ens)

for i in ProgressBar(1:n_ens)
    R1 = rand(Int)
    u0 = 2*rand(FT, N^2).-1
    tspan = FT.((0.0,2*t_decorr))
    X0 = trajectory(u0, tspan, dt, dt_save, R1)[:,end]
    X0eps = copy(X0)
    X0eps[1] += eps
    tspan = FT.((0.0,tau))
    R2 = rand(Int)
    t1 = trajectory(X0, tspan, dt, dt_save, R2)
    t2 = trajectory(X0eps, tspan, dt, dt_save, R2)
    responseN_ens[:,:,i] = (t2 .- t1)./eps
end
responseN = reshape(mean(responseN_ens, dims=3),(N^2,n_tau))
err = zeros(N^2,n_tau)
for i in 1:n_tau
    err[:,i] = std(responseN_ens[:,i,:], dims=2)
end

pl = plot(
    plot([responseN[1,:].+err[1,:]./sqrt(n_ens) responseN[1,:].-err[1,:]./sqrt(n_ens) responseL_mean[1,:]] ,label=["numerics" "" "linear app"],color=[:red :red :blue],title="1 -> 1",xlabel="time",ylabel="response", linewidth=3),
    plot([responseN[2,:].+err[2,:]./sqrt(n_ens) responseN[2,:].-err[2,:]./sqrt(n_ens) responseL_mean[2,:]],label=["" "" ""],color=[:red :red :blue :black],title="1 -> 2",xlabel="time",ylabel="response", linewidth=3),
    # plot([.-responseN[3,:].+error[3,:]./sqrt(n_thr*n_ens) .-responseN[3,:].-error[3,:]./sqrt(n_thr*n_ens) responseL_mean[3,:] responseS_norm_mean[3,:]],label=["" "" "" ""],color=[:red :red :blue :black],title="1 -> 3",xlabel="time",ylabel="response", linewidth=3),
    # plot([.-responseN[4,:].+error[4,:]./sqrt(n_thr*n_ens) .-responseN[4,:].-error[4,:]./sqrt(n_thr*n_ens) responseL_mean[4,:] responseS_norm_mean[4,:]],label=["" "" "" ""],color=[:red :red :blue :black],title="1 -> 4",xlabel="time",ylabel="response", linewidth=3),
    layout=(1,2),
    size=(1000,500),
    plot_title="alpha = $α, beta = $β, gamma = $γ, sigma = $σ"
)
display(pl)
savefig(pl, "response_gamma_$γ.png")