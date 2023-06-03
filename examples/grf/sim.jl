using DifferentialEquations: CorrelatedWienerProcess!, SDEProblem, solve, init, step!, EM
using Plots
using StatsBase

include("./rhs.jl")
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
# TODO: Run for approximately 125*1e4 = 1.25e6, saving output every 25-40
tspan = FT.((0.0,1e2))
dt = FT(0.1)
dt_save = FT(25)
n_steps_per_save = Int(round(dt_save/dt))
integrator = create_integrator(model, u0, tspan, dt)
nsteps = Int((tspan[2]-tspan[1])/dt)
savesteps = 0:n_steps_per_save:nsteps

solution = zeros(FT, (N, N, Int(nsteps/n)));
@time for i in 1:nsteps
    step!(integrator)
    if i ∈ savesteps
        save_index = Int(i/n)
        solution[:,:,save_index] .= reshape(integrator.u, (N,N))
    end
end
