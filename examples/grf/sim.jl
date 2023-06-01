using DifferentialEquations: CorrelatedWienerProcess!, SDEProblem, solve
using Plots

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
                    du[k] += ((u[k_im1] - u[k]) - (u[k] - u[(j-1)*N+1]))
                elseif i ==1
                    du[k] += ((u[(j-1)*N+N] - u[k]) - (u[k] - u[k_ip1]))
                else
                    du[k] += ((u[k_im1] - u[k]) - (u[k] - u[k_ip1]))
                end
                if j == N
                    du[k] += ((u[i] - u[k]) - (u[k] - u[k_jm1]))
                elseif j ==1
                    du[k] += ((u[k_jp1] - u[k]) - (u[k] - u[(N-1)*N+i]))
                else
                    du[k] += ((u[k_jp1] - u[k]) - (u[k] - u[k_jm1]))
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


function evolve_stochastic_system(model::LinearDiffusionSDE, u0, tspan, dt, alg_kwargs)
    deterministic_tendency!, stochastic_tendency!, p = stochastic_system(model)
    Γ = model.Γ
#    W0 = reshape(randn(model.N^4)*dt, (model.N^2, model.N^2))
    dW = CorrelatedWienerProcess!(Γ, tspan[1], zeros(model.N^2), zeros(model.N^2))
    prob = SDEProblem(deterministic_tendency!, stochastic_tendency!, u0, tspan, p; noise = dW)
    sol = solve(prob, dt = dt, saveat = tspan[1]:dt:tspan[2]; alg_kwargs...);
    return sol.u
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
                    Γ[k1,k2] = 1.0/sqrt((i1-i2)^2 + (j1-j2)^2)#cos((i1-i2)*2*pi/4)*cos((j1-j2)*2*pi/4)
                end
                
            end
        end
    end
end

           
model = LinearDiffusionSDE(0.1, Γ, N, Periodic())

tspan = (0.0,40.0)
dt = 0.1
alg_kwargs = (adaptive=false,)
sol = evolve_stochastic_system(model, u0, tspan, dt, alg_kwargs)
clims= (minimum(minimum(sol[:])),maximum(maximum(sol[:])))
anim = convert_to_animation(sol, 1, N, clims)
gif(anim, "anim.gif", fps = 20)
