@everywhere using DifferentialEquations: WienerProcess, SDEProblem, solve
# Stand-in for a more complex Model - returns necessary RHS functions, parameters
@everywhere function ornstein_uhlenbeck1D_model(θ, σ, t0)
    function deterministic_tendency!(dudt,u,p,t)
        dudt .= -p.θ *u 
    end
    function stochastic_tendency!(dudW,u,p,t)
        dudW .= p.σ
    end
    p = (θ = 1.0, σ = 1.0)
    return deterministic_tendency!, stochastic_tendency!, p
end
# Sets up the simulation - whatever your problem needs to be integrated
# forwards in time with output saved every dt, given IC, tspan,
# and other sim_kwargs

@everywhere function evolve_ornstein_uhlenbeck1D(u0, tspan, dt, alg_kwargs)
    deterministic_tendency!, stochastic_tendency!, p = ornstein_uhlenbeck1D_model(1.0,1.0, tspan[1])
    W0 = randn()*dt
    dW = WienerProcess(tspan[1], W0,0.0)
    prob = SDEProblem(deterministic_tendency!, stochastic_tendency!, u0, tspan, p, noise = dW)
    sol = solve(prob, dt = dt, saveat = tspan[1]:dt:tspan[2]; alg_kwargs...);
    return sol.u
end
