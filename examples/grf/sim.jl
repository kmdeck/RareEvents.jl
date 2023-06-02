using DifferentialEquations: CorrelatedWienerProcess!, SDEProblem, solve
using Plots
using StatsBase


function convert_to_animation(x, time_stride, N, clims)
    init_frames = length(x)
    x = x[1:time_stride:init_frames]
    frames = length(x)
    animation = @animate for i = 1:frames
            heatmap(
                reshape(x[i],(N,N)),
                xaxis = false, yaxis = false, xticks = false, yticks = false,
                clims = clims
            )
    end
    return animation
end
abstract type AbstractBoundaryConditions end

struct Periodic <: AbstractBoundaryConditions end
struct Dirichlet{FT} <: AbstractBoundaryConditions
    bc::FT
end



struct LinearDiffusionSDE{FT <:AbstractFloat, BC}
    σ::FT
    Γ::Matrix{FT}
    N::Int
    bc::BC
end

function stochastic_system(model::LinearDiffusionSDE{FT, Periodic}) where {FT}
    function deterministic_tendency!(du,u,p,t)
        N = p.N
        du .= 0

        for i in 1:N
            for j in 1:N
                k = (j-1)*N+i
                k_ip1 = k+1
                k_im1 = k-1
                k_jp1 = k + N
                k_jm1 = k - N  
                if i == N          
                    du[k] += 1/2*((u[k_im1] - u[k]) - (u[k] - u[(j-1)*N+1]))
                elseif i ==1
                    du[k] += 1/2*((u[(j-1)*N+N] - u[k]) - (u[k] - u[k_ip1]))
                else
                    du[k] += 1/2*((u[k_im1] - u[k]) - (u[k] - u[k_ip1]))
                end
                if j == N
                    du[k] += 1/2*((u[i] - u[k]) - (u[k] - u[k_jm1]))
                elseif j ==1
                    du[k] += 1/2*((u[k_jp1] - u[k]) - (u[k] - u[(N-1)*N+i]))
                else
                    du[k] += 1/2*((u[k_jp1] - u[k]) - (u[k] - u[k_jm1]))
                end
            end
        end
    end
    
    function stochastic_tendency!(du,u,p,t)
        du .= p.σ
    end
    p = (σ = model.σ, model.N)
    return deterministic_tendency!, stochastic_tendency!, p
end


function stochastic_system(model::LinearDiffusionSDE{FT, Dirichlet{FT}}) where {FT}
    function deterministic_tendency!(du,u,p,t)
        N = p.N
        du .= 0
        u_bc = p.u_bc
        for i in 1:N
            for j in 1:N
                k = (j-1)*N+i
                k_ip1 = k+1
                k_im1 = k-1
                k_jp1 = k + N
                k_jm1 = k - N  
                if i == N          
                    du[k] += 1/2*((u[k_im1] - u[k]) - (u[k] - u_bc))
                elseif i ==1
                    du[k] += 1/2*((u_bc - u[k]) - (u[k] - u[k_ip1]))
                else
                    du[k] += 1/2*((u[k_im1] - u[k]) - (u[k] - u[k_ip1]))
                end
                if j == N
                    du[k] += 1/2*((u_bc - u[k]) - (u[k] - u[k_jm1]))
                elseif j ==1
                    du[k] += 1/2*((u[k_jp1] - u[k]) - (u[k] - u_bc))
                else
                    du[k] += 1/2*((u[k_jp1] - u[k]) - (u[k] - u[k_jm1]))
                end
            end
        end
    end
    
    function stochastic_tendency!(du,u,p,t)
        du .= p.σ
    end
    p = (u_bc = model.bc.bc, σ = model.σ, model.N)
    return deterministic_tendency!, stochastic_tendency!, p
end


function evolve_stochastic_system(model::LinearDiffusionSDE, u0, tspan, saveat, dt, alg_kwargs)
    deterministic_tendency!, stochastic_tendency!, p = stochastic_system(model)
    Γ = model.Γ
    dW = CorrelatedWienerProcess!(Γ, tspan[1], zeros(model.N^2), zeros(model.N^2))
    prob = SDEProblem(deterministic_tendency!, stochastic_tendency!, u0, tspan, p; noise = dW)
    sol = solve(prob, dt = dt, saveat = saveat; alg_kwargs...);
    return sol.t, sol.u
end
N = 32
Γ = reshape(zeros(N^4), (N^2,N^2))
u0 = zeros(N^2)

for i1 in 1:N
    for j1 in 1:N
        k1 = (j1-1)*N+i1
        for i2 in 1:N
            for j2 in 1:N
                k2 = (j2-1)*N+i2
                if k1 == k2
                    Γ[k1,k2] = 1.0
                else
                    Γ[k1,k2] = 1.0/sqrt((i1-i2)^2 + (j1-j2)^2)
                end
                
            end
        end
    end
end

           
#model = LinearDiffusionSDE(1.0, Γ, N, Periodic())
model = LinearDiffusionSDE(1.0, Γ, N, Dirichlet(0.0))
# TODO: Test that a smaller timestep gives the same results.
# Tested at 3e4 originally
# TODO: Run for approximately 
tspan = (0.0,2e6)
dt = 0.1
T = 50
n = Int(T/dt)
saveat = tspan[1]:dt*n:tspan[2]
alg_kwargs = (adaptive=false,)
t, u = evolve_stochastic_system(model, u0, tspan, saveat, dt, alg_kwargs);
# autocorrelation time
center_row = Int64.((Array(1:1:N) .-1) .*N .+N/2)
values = zeros(length(u), N)
for i in 1:length(u)
    values[i,:] .= u[i][center_row]
end
p528 = [u[k][528] for k in 1:length(t)]
Plots.plot(t, p528)
lags = Array(1:1:(length(t)-1)) # in units of time
ac = StatsBase.autocor(values, lags; demean = true)
mean_ac = mean(ac, dims = 2)[:]

Plots.plot(lags, mean(ac, dims = 2)[:])
# by eye, a lag of lags[3000]*dt = 3e2 is the autocorrelation time
τ = maximum(lags[abs.(mean_ac) .> 0.1])*T
n_τ = (tspan[2]-tspan[1])/τ

#clims= (minimum(minimum(sol.u[:])),maximum(maximum(sol.u[:])))
#anim = convert_to_animation(sol, 300, N, clims)
#gif(anim, "anim.gif", fps = 10)
