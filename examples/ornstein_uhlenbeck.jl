using Distributed
@everywhere using DifferentialEquations: WienerProcess, SDEProblem, solve
# Stand-in for a more complex Model - returns necessary RHS functions, parameters

struct OrnsteinUhlenbeck{FT <:AbstractFloat}
    θ::Union{FT, Matrix{FT}}
    σ::Union{FT, Matrix{FT}}
    d::Int
end


@everywhere function stochastic_system(model::OrnsteinUhlenbeck)
    function deterministic_tendency!(dudt,u,p,t)
        dudt .= -p.θ *u 
    end
    function stochastic_tendency!(dudW,u,p,t)
        dudW .= p.σ
    end
    p = (θ = model.θ, σ = model.σ)
    return deterministic_tendency!, stochastic_tendency!, p
end
# Sets up the simulation - whatever your problem needs to be integrated
# forwards in time with output saved every dt, given IC, tspan,
# and other sim_kwargs

## Noise process should be part of OU?

@everywhere function evolve_stochastic_system(model::OrnsteinUhlenbeck, u0, tspan, dt, alg_kwargs)
    deterministic_tendency!, stochastic_tendency!, p = stochastic_system(model)
    W0 = randn(model.d)*dt
    dW = WienerProcess(tspan[1], W0,0.0)
    prob = SDEProblem(deterministic_tendency!, stochastic_tendency!, u0, tspan, p; noise_rate_prototype=zeros(model.d,model.d), noise = dW)
    sol = solve(prob, dt = dt, saveat = tspan[1]:dt:tspan[2]; alg_kwargs...);
    return sol.u
end
