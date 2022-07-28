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
nensemble = 600
u0 = [copy(zeros(d)) .+ randn(d)*0.71 for i in 1:nensemble]
k = 0.3
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
algorithm_cost = nensemble*(tspan[2]-tspan[1])*iters*dt
for iter in 1:iters
    println(iter)
    sim = RareEventSampler{Float64}(dt, τ, tspan,u0, evolve_wrapper, nensemble, metric, k, ϵ, iter)
    
    run!(sim);
    a_m = zeros(nensemble)
    lr_vector = zeros(nensemble)
    λ = sum(log.(sim.weight_norm))/T_a    
    
    for i in 1:nensemble
        trajectory = [metric(sim.ensemble[i][j]) for j in 1:length(sim.ensemble[i])]
        lr_vector[i] = likelihood_ratio(sim, trajectory, λ, T_a)
        a_m[i] =  maximum(block_mean(trajectory, NT))

    end
    em[:,iter], r[:,iter], r_paper[:,iter], σr[:, iter]= return_curve(a_m, T_a-T, lr_vector);

end
rmprocs([p for p in procs() if p != myid()])


# Evolve the solution directly; compute statistics
tspan_direct = (0.0, 5e5)
direct_solution = evolve_stochastic_system(model, zeros(d), tspan_direct, dt, (maxiters = 1e8,));
direct_cost = (tspan_direct[2]-tspan_direct[1])*dt
u = [metric(direct_solution[k]) for k in 1:length(direct_solution)]
A = block_mean(u, NT)# same stats as segment matrix
m = 100
em_direct, r_direct, r_paper_direct, σr_direct = return_curve(A, FT(T),  FT.(ones(length(A))))

# Fit GEV
m = 100 # I see sensitivity here!
blocks = block_maxima(A, m)
params, nll = fit_gev(blocks, [std(blocks), mean(blocks), 0.1])

histogram(blocks, norm = true, label = "Data")
bin = 0.01
magnitude = Array(-0.06:bin:1.0)
cdf_A= (gev_cdf.(magnitude, params[1], params[2], params[3])).^(1.0/m)
pdf_A= (cdf_A[2:end] .- cdf_A[1:end-1])/bin

# Lots of plots
#=
plot!(magnitude, gev_pdf.(magnitude, params[1], params[2], params[3]), label = "Fit")
# looks reasonable. Now to look at CDF.
plot(sort(blocks), Array(1:1:length(blocks))./length(blocks), label  ="Data")
plot!(magnitude, gev_cdf.(magnitude, params[1], params[2], params[3]), label = "Fit")


# Yes, the CDF of A ^m looks like CDF of maxima over m
plot(sort(blocks), Array(1:1:length(blocks))./length(blocks))
plot!(sort(A), (Array(1:1:length(A))./length(A)).^m)
plot!(magnitude, gev_cdf.(magnitude, params[1], params[2], params[3]), label = "Fit")

# What about the CDF of A
plot(sort(blocks), (Array(1:1:length(blocks))./length(blocks)).^(1/m))
plot!(sort(A), Array(1:1:length(A))./length(A))

# Maybe this is only good in the tail? I think so?
plot(magnitude, cdf_A)
plot!(sort(A), Array(1:1:length(A))./length(A))
=#



# Make plots
plot1 = plot()
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
plot!(plot1,final_a, final_r, ribbon = final_σr, label = "GKLT Algorithm")
plot!(plot1,em_direct, log10.(r_paper_direct), ribbon = σr_direct ./ r_paper_direct ./ log(10.0), label = "Direct Integration")
plot!(plot1, magnitude, log10.(50.0 ./ (1.0 .- cdf_A)), label = "GEV")
plot!(plot1, magnitude, log10.(50.0 ./ (1.0 .- 0.5 .*(1 .+erf.(magnitude ./2 .^0.5 ./std(A))))), label = "Gaussian")
plot!(plot1,xlabel = "Event magnitude")
plot!(plot1,ylabel = "Log10(Return Time)")
plot!(plot1, legend = :topleft)
plot!(plot1, xrange = [0.0,0.1.0])
savefig(plot1, "return_curve_ou.png")
