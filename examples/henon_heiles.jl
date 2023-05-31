using Distributed
@everywhere using DifferentialEquations: ODEProblem, solve, ImplicitMidpoint

struct HenonHeiles end

@everywhere function deterministic_system(model::HenonHeiles)
    function tendency!(dudt,u,p,t)
        dudt[1] = u[3]
        dudt[2] = u[4]
        dudt[3] = -u[1]-2.0*u[1]*u[2]
        dudt[4] = -u[2]-(u[1]^2-u[2]^2)
    end
    return tendency!
end

@everywhere function evolve_deterministic_system(model::HenonHeiles, u0, tspan, dt, alg_kwargs)
    tendency! = deterministic_system(model)
    prob = ODEProblem(tendency!, u0, tspan, nothing)
    sol = solve(prob, ImplicitMidpoint(), dt = dt, saveat = tspan[1]:dt:tspan[2]; alg_kwargs...);
    return sol.u
end
