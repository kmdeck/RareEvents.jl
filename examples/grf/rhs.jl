"""
 Model
"""
struct LinearDiffusionSDE{FT <:AbstractFloat, BC}
    σ::FT
    Γ::Matrix{FT}
    N::Int
    bc::BC
end

abstract type AbstractBoundaryConditions end

struct Periodic <: AbstractBoundaryConditions end

struct Dirichlet{FT} <: AbstractBoundaryConditions
    bc::FT
end

function stochastic_system(model::LinearDiffusionSDE{FT, Periodic}) where {FT}
    function deterministic_tendency!(du,u,p,t)
        N = p.N
        du .= FT(0)

        for i in 1:N
            for j in 1:N
                k = (j-1)*N+i
                k_ip1 = k+1
                k_im1 = k-1
                k_jp1 = k + N
                k_jm1 = k - N  
                if i == N          
                    du[k] += FT(1/2)*((u[k_im1] - u[k]) - (u[k] - u[(j-1)*N+1]))
                elseif i ==1
                    du[k] += FT(1/2)*((u[(j-1)*N+N] - u[k]) - (u[k] - u[k_ip1]))
                else
                    du[k] += FT(1/2)*((u[k_im1] - u[k]) - (u[k] - u[k_ip1]))
                end
                if j == N
                    du[k] += FT(1/2)*((u[i] - u[k]) - (u[k] - u[k_jm1]))
                elseif j ==1
                    du[k] += FT(1/2)*((u[k_jp1] - u[k]) - (u[k] - u[(N-1)*N+i]))
                else
                    du[k] += FT(1/2)*((u[k_jp1] - u[k]) - (u[k] - u[k_jm1]))
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
        du .= FT(0)
        u_bc = p.u_bc
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
    
    function stochastic_tendency!(du,u,p,t)
        du .= p.σ
    end
    p = (u_bc = model.bc.bc, σ = model.σ, model.N)
    return deterministic_tendency!, stochastic_tendency!, p
end

function evolve_stochastic_system(model::LinearDiffusionSDE{FT}, u0, tspan, saveat, dt, alg_kwargs) where {FT}
    deterministic_tendency!, stochastic_tendency!, p = stochastic_system(model)
    Γ = model.Γ
    dW = CorrelatedWienerProcess!(Γ, tspan[1], FT.(zeros(model.N^2)), FT.(zeros(model.N^2)))
    prob = SDEProblem(deterministic_tendency!, stochastic_tendency!, u0, tspan, p; noise = dW)
    sol = solve(prob, dt = dt, saveat = saveat; alg_kwargs...);
    return sol
end


function create_integrator(model::LinearDiffusionSDE{FT}, u0, tspan, dt) where {FT}
    deterministic_tendency!, stochastic_tendency!, p = stochastic_system(model)
    Γ = model.Γ
    dW = CorrelatedWienerProcess!(Γ, tspan[1], FT.(zeros(model.N^2)), FT.(zeros(model.N^2)))
    prob = SDEProblem(deterministic_tendency!, stochastic_tendency!, u0, tspan, p; noise = dW, save_noise=false)
    integrator = init(
        prob,
        dt = dt,
        saveat = tspan[1]:(tspan[2]-tspan[1]):tspan[2],
        adaptive=false)
    return integrator
end
