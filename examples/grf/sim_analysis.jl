using DifferentialEquations: CorrelatedWienerProcess!, SDEProblem, solve, init, step!
using Plots
using StatsBase
using HDF5

include("./rhs.jl")
include("./utils.jl")
FT = Float32
N = 32
Γ = FT.(reshape(zeros(N^4), (N^2,N^2)))
u0 = FT.(zeros(N^2))

for i1 in 1:N
    for j1 in 1:N
        k1 = (j1-1)*N+i1
        for i2 in 1:N
            for j2 in 1:N
                k2 = (j2-1)*N+i2
                if k1 == k2
                    Γ[k1,k2] = FT(1.0)
                else
                    Γ[k1,k2] = FT(1.0/sqrt((i1-i2)^2 + (j1-j2)^2))
                end
                
            end
        end
    end
end
u_bc = FT(0)
model = LinearDiffusionSDE(FT(1.0), Γ, N, Dirichlet(u_bc))
# [X] Test that a smaller timestep gives the same results. (0.1 and 0.02)
# TODO: Run for approximately 300*1e4 = 3e6, saving output every 75
tspan = FT.((0.0,1e4))
dt = FT(0.1)
T = 2.5
n = Int(round(T/dt))

saveat = tspan[1]:dt*n:tspan[2];
alg_kwargs = (adaptive=false,);
@time sol = evolve_stochastic_system(model, u0, tspan, saveat, dt, alg_kwargs);
u = sol.u
t = sol.t
# autocorrelation time
center_row = Int64.((Array(1:1:N) .-1) .*N .+N/2)
values = zeros(length(u), N)
for i in 1:length(u)
    values[i,:] .= u[i][center_row]
end
p528 = [u[k][528] for k in 1:length(t)]
Plots.plot(t, p528)
lags = Array(1:1:(length(t)-1)) # not in units of time
ac = StatsBase.autocor(values, lags; demean = true)
mean_ac = mean(ac, dims = 2)[:]

Plots.plot(lags *T, mean(ac, dims = 2)[:])
τ = maximum(lags[mean_ac .> 0.1])*T
n_τ = (tspan[2]-tspan[1])/τ

clims= (minimum(minimum(u[:])),maximum(maximum(u[:])))
anim = convert_to_animation(u[1:100], 1, N, clims)
gif(anim, "anim.gif", fps = 5)

