using Revise
using RareEvents
using Statistics
using DelimitedFiles
using LinearAlgebra
using Plots

examples_dir = joinpath(pkgdir(RareEvents), "examples")
include(joinpath(examples_dir, "ornstein_uhlenbeck.jl"))
n_processes = Sys.CPU_THREADS
addprocs(n_processes)

# Set up model and wrapper function that evoves a single trajectory
dt = 0.1
alg_kwargs = ();
FT = Float64
d=1
θ = Diagonal(ones(d)) .+ zeros(Float64,d,d)
σ = Diagonal(ones(d)) .+ zeros(Float64,d,d)
model = OrnsteinUhlenbeck{FT}(θ, σ, d)
@everywhere evolve_wrapper(tspan) = (u0) -> evolve_stochastic_system(model, u0, tspan, dt, alg_kwargs)


# Set up rare event algorithm
τ = 0.5
nensemble = 500
u0 = [copy(zeros(d)) for i in 1:nensemble]
k = 0.4
ϵ = 0.002
metric(y) = y[1]
T_a = 100.0
tspan = (0.0, T_a)
T = 50
NT = Int64.(round(T/dt))


# Set up bootstrap iterations
iters = 10
em = zeros(nensemble*iters)
em = reshape(em, (nensemble,iters))
r = similar(em)
r_paper = similar(em)
σr = similar(em)

a_range = Array(0.0:0.03:1.0)
Na = zeros(length(a_range)*iters)
Na = reshape(Na, (length(a_range), iters))

algorithm_cost = nensemble*(tspan[2]-tspan[1])*iters*dt
for iter in 1:iters
    println(iter)
    sim = RareEventSampler{Float64}(dt, τ, tspan,u0, evolve_wrapper, nensemble, metric, k, ϵ)
    
    run!(sim);
    moving_average_matrix = zeros((nensemble, length(sim.ensemble[1])-NT))
    moving_average_row = zeros(length(sim.ensemble[1])-NT)
    lr_vector = zeros(nensemble)
    λ = sum(log.(sim.weight_norm))/T_a    
    
    for i in 1:nensemble
        trajectory = [metric(sim.ensemble[i][j]) for j in 1:length(sim.ensemble[i])]
        moving_average_row .= moving_average_matrix[i,:]
        moving_average!(moving_average_row, trajectory, NT)
        moving_average_matrix[i,:] .= moving_average_row
        lr_vector[i] = likelihood_ratio(sim, trajectory, λ, T_a)
    end
    Na[:,iter] .+= N_event(moving_average_matrix, lr_vector, a_range)

    a_m = maximum(moving_average_matrix, dims = 2)[:]
    em[:,iter], r[:,iter], r_paper[:,iter], σr[:, iter]= return_curve(a_m, T_a-T, lr_vector);

end
rmprocs([p for p in procs() if p != myid()])


# Evolve the solution directly; compute statistics
tspan_direct = (0.0, 5e5)
direct_solution = evolve_stochastic_system(model, zeros(d), tspan_direct, dt, (maxiters = 1e8,));
direct_cost = (tspan_direct[2]-tspan_direct[1])*dt
u = [metric(direct_solution[k]) for k in 1:length(direct_solution)]
A = zeros(length(u)-NT-1)
moving_average!(A,u,NT)
M = Int(length(A)/NT)
segment_matrix  = reshape(A, (NT,M))
em_direct, r_direct, r_paper_direct, σr_direct = return_curve(maximum(segment_matrix, dims = 1)[:], FT(T_a-T),  FT.(ones(M)))




# Make plots
mean_a = zeros(length(a_range)-1)
mean_rtn = zeros(length(a_range)-1)
std_rtn = zeros(length(a_range)-1)
for i in 1:(length(a_range)-1)
    mask = (em[:] .< a_range[i+1]) .& ( em[:] .>= a_range[i])
    if sum(mask) >0
        mean_a[i] = mean(em[:][mask])
        mean_rtn[i] = log10.(mean(r_paper[:][mask]))
        std_rtn[i] = std(r_paper[:][mask])/mean(r_paper[:][mask])./log(10.0)
    end
    
end
nonzero = (mean_a .!= 0.0) .& (isnan.(std_rtn) .== 0)
final_a = mean_a[nonzero]
final_r = mean_rtn[nonzero]
final_σr = std_rtn[nonzero]
plot(final_a, final_r, ribbon = final_σr, label = "GKLT Algorithm")
plot!(em_direct, log10.(r_paper_direct), ribbon = σr_direct ./ r_paper_direct ./ log(10.0), label = "Direct Integration")
plot!(xlabel = "Event magnitude")
plot!(ylabel = "Log10(Return Time)")
plot!(legend = :topleft)
plot!(xrange = [0.2,0.75])
savefig("return_curve_ou.png")



# We can also count the number of events seen directly.
Na_direct = N_event(segment_matrix, ones(NT), a_range)
p_direct = Na_direct[:]./sum(Na_direct)

N03 = mean(Na, dims =2 )
σN03 = std(Na, dims =2 )
p03 = N03[:]./sum(N03)
σp03 = σN03[:] / sum(N03)
plot(a_range[:], log10.(p03), ribbon = σp03[:] ./ p03[:] ./log(10.0), label = "GKLT Algorithm")
plot!(a_range[:], log10.(p_direct), label = "Direct Integration")
plot!(ylabel = "Log10(Probability of Event)", xlabel = "Event magnitude")
savefig("probabilities_ou.png")
