using Revise
using RareEvents
using Statistics
using DelimitedFiles
using LinearAlgebra
using SpecialFunctions
using Plots

examples_dir = joinpath(pkgdir(RareEvents), "examples")
include(joinpath(examples_dir, "ornstein_uhlenbeck.jl"))
n_processes = Sys.CPU_THREADS
addprocs(n_processes)

# Set up model and wrapper function that evolves a single trajectory
dt = 0.1
alg_kwargs = ();
FT = Float64
dimensions =1
θ = Diagonal(ones(dimensions)) .+ zeros(Float64,dimensions,dimensions)
σ = Diagonal(ones(dimensions)) .+ zeros(Float64,dimensions,dimensions)
model = OrnsteinUhlenbeck{FT}(θ, σ, dimensions)
@everywhere evolve_wrapper(tspan) = (u0) -> evolve_stochastic_system(model, u0, tspan, dt, alg_kwargs)


# Set up rare event algorithm
τ = 0.5
nensemble = 1000
u0 = [copy(zeros(dimensions)) .+ randn(dimensions)*0.71 for i in 1:nensemble]
k = 0.3
ϵ = 0.002
metric(y) = y[1]
t_end = 50.0
t_start = 0.0
tspan = (t_start, t_end)
# averaging window
T = 50
steps_per_window = Int64.(round(T/dt))


# Run GKLT
iters = 5
em = zeros(nensemble*iters)
em = reshape(em, (nensemble,iters))
r = similar(em)
r_paper = similar(em)
σr = similar(em)
bin = 0.03
a_range = Array(0.0:bin:1.0)
counts = zeros(length(a_range))
algorithm_cost = nensemble*(t_end-t_start)*iters*dt
for iter in 1:iters
    println(iter)
    sim = RareEventSampler{Float64}(dt, τ, tspan,u0, evolve_wrapper, nensemble, metric, k, ϵ, iter)
    
    run!(sim);
    a_m = zeros(nensemble)
    lr_vector = zeros(nensemble)
    λ = sum(log.(sim.weight_norm))/(t_end-t_start)
    
    for i in 1:nensemble
        trajectory = [metric(sim.ensemble[i][j]) for j in 1:length(sim.ensemble[i])]
        lr_vector[i] = likelihood_ratio(sim, trajectory, λ, t_end-t_start)
        # This just counts the number of (treated as independent) block means per bin, over all trajectories, over all iterations. Not doing anything with it, yet
        counts .+=  count_events_per_bin(block_mean(trajectory, steps_per_window), lr_vector[i],a_range)
        # This is for estimating return times via the Ragone paper method; via `return_curve`.
        a_m[i] = maximum(block_mean(trajectory, steps_per_window))

    end
    em[:,iter], r[:,iter], r_paper[:,iter], σr[:, iter]= return_curve(a_m, t_end-t_start, lr_vector);

end
rmprocs([p for p in procs() if p != myid()])


# Evolve the solution directly; compute statistics
tspan_direct = (0.0, 2.5e5)
direct_solution = evolve_stochastic_system(model, zeros(dimensions), tspan_direct, dt, (maxiters = 1e8,));
direct_cost = (tspan_direct[2]-tspan_direct[1])*dt
u = [metric(direct_solution[k]) for k in 1:length(direct_solution)]
A = block_mean(u, steps_per_window)
em_direct, r_direct, r_paper_direct, σr_direct = return_curve(A, FT(T),  FT.(ones(length(A))))

# Fit GEV
m = 100 
blocks = block_maxima(A, m)
params, nll = fit_gev(blocks, [std(blocks), mean(blocks), 0.1])
chain = evolve_mcmc(blocks, params, [0.01,0.01,0.05], log_likelihood_gev,1000000; save_at = 100)
n_samples = 100
chain_samples = chain[Int.(round.(rand(n_samples)*length(chain)))]
gev_rtn = zeros(n_samples, length(a_range))
for (i, sample) in enumerate(chain_samples)
    cdf = min.(gev_cdf.(a_range, sample[1], sample[2], sample[3]).^(1.0/m), 1.0-eps(FT))
    gev_rtn[i,:] .= 50.0 ./ (1.0 .- cdf)
end
median_curve = [median(gev_rtn[:,k]) for k in 1:length(a_range)]
upper_curve = [sort(gev_rtn[:,k])[83] for k in 1:length(a_range)]
lower_curve = [sort(gev_rtn[:,k])[17] for k in 1:length(a_range)]


# GKLT curve - we need to combine iterations into single result
mean_a = zeros(length(a_range)-1)
mean_rtn = zeros(length(a_range)-1)
std_rtn = zeros(length(a_range)-1)
for i in 1:(length(a_range)-1)
    mask = (em[:] .< a_range[i+1]) .& ( em[:] .>= a_range[i])
    if sum(mask) >0
        mean_a[i] = mean(em[:][mask])
        mean_rtn[i] = mean(r_paper[:][mask])
        std_rtn[i] = std(r_paper[:][mask])
    end
end
nonzero = (mean_a .!= 0.0) .& (isnan.(std_rtn) .== 0)
final_a = mean_a[nonzero]
final_r = mean_rtn[nonzero]
final_σr = std_rtn[nonzero]

plot1 = plot()
plot!(plot1,final_a, final_r, ribbon = final_σr, label = "GKLT Algorithm")
plot!(plot1,em_direct[2:end], r_paper_direct[2:end], ribbon = σr_direct[2:end], label = "Direct Integration")
plot!(plot1, a_range, median_curve, ribbon = (median_curve .- lower_curve, upper_curve .- median_curve), label = "GEV", color = :purple)
plot!(plot1,xlabel = "Event magnitude")
plot!(plot1,ylabel = "Return Time")
plot!(plot1, legend = :topleft)
plot!(plot1, xrange = [0.0,0.8])
plot!(plot1, yaxis = :log)
plot!(plot1, yrange = (1,1e10), yticks = [1, 1e2, 1e4, 1e6, 1e8, 1e10])
savefig(plot1, "return_curve_ou.png")
@assert direct_cost ≈ algorithm_cost
