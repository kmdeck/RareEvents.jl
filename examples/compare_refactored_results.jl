using DelimitedFiles
using Plots
using RareEvents
using Statistics

files = ["k_03_bootstrap_Na.csv", "refactored_results/k_03_bootstrap_Na_refactored.csv","k_05_bootstrap_Na.csv", "refactored_results/k_05_bootstrap_Na_refactored.csv","direct_1e7_Na_dt_01.csv","refactored_results/direct_1e7_Na_dt_01_refactored.csv" ]
titles = ["original, k=0.3", "refactored, k = 0.3","original, k=0.5", "refactored, k = 0.5","original, direct", "refactored, direct"]
plot1 = plot()
plot2 = plot()
a_range = Array(0.0:0.03:1.0)
for i in [1,2,5,6]
    Na = readdlm(files[i])
    N = mean(Na, dims =2 )
    σN = std(Na, dims =2 )
    p = N[:]./sum(N)
    σp = σN[:] / sum(N)
    plot!(plot1,a_range[:], log10.(p), ribbon = σp[:] ./ p[:] ./log(10.0), label = titles[i])

    nonzero = p .!= 0.0
    RT = 50.0./p[nonzero]
    plot!(plot2, a_range[nonzero], log10.(RT), ribbon = σp[nonzero]./p[nonzero]./log(10.0), label = titles[i])
end
#σ = 0.14
#plot!(a_range[:], log10.(0.03/sqrt(2.0*π*σ^2.0) .* exp.(-1.0 .*(a_range[:] .^2.0 ./(2.0*σ^2.0)))), label= "Gaussian")
plot!(plot1, ylabel = "P(a)Δa", xlabel = "a")
plot!(plot1, legend = :bottomleft)
plot!(plot2,ylim = [0,13], xlim = [0,1.0])
plot!(plot2,legend = :bottomright)
plot!(plot2,ylabel = "Log10(Return Time of Event of Magnitude a)")
plot!(plot2,xlabel = "magnitude of event a")


# Using the max method
a_k05 = readdlm("k_05_bootstrap.csv")
lr_k05 = readdlm("k_05_bootstrap_lr.csv")
a_k03 = readdlm("k_03_bootstrap.csv")
lr_k03 = readdlm("k_03_bootstrap_lr.csv")



a = a_k05
lr = lr_k05
N = 600
T_a = 100
T = 50.0

iters = 100
em = zeros((N-1)*iters)
em = reshape(em, ((N-1),iters))
r = similar(em)
σr = similar(em)
for i in 1:iters
    event_magnitude, rtn, _= return_curve(a[:,i], T_a-T, lr[:,i]);
    em[:,i] = event_magnitude
    r[:,i] = rtn
end
mean_a = zeros(length(a_range)-1)
mean_rtn = zeros(length(a_range)-1)
std_rtn = zeros(length(a_range)-1)
for i in 1:(length(a_range)-1)
    mask = (em[:] .< a_range[i+1]) .& ( em[:] .>= a_range[i])
    if sum(mask) >0
        mean_a[i] = mean(em[:][mask])
        mean_rtn[i] = log10.(mean(r[:][mask]))
        std_rtn[i] = std(r[:][mask])/mean(r[:][mask])./log(10.0)
    end
    
end
nonzero = (mean_a .!= 0.0) .& (isnan.(std_rtn) .== 0)
final_a = mean_a[nonzero]
final_r = mean_rtn[nonzero]
final_σr = std_rtn[nonzero]
plot!(final_a, final_r, label = "k=0.5 via max")

savefig("./rt_event_3x.png")


### Fit p = e^-(TI(a)) -> -log(p)/T = I(a)
poly = fit(a_range[nonzero03], -log.(p03[nonzero03]) ./50.0, 4)
plot(a_range, poly.(a_range), label = "k = 0.3")
plot!(xlabel = "a")
plot!(ylabel = "I(a)")
poly = fit(a_range[nonzero05], -log.(p05[nonzero05]) ./50.0, 4)
plot!(a_range, poly.(a_range), label = "k = 0.5")
poly = fit(a_range[nonzero_direct], -log.(p_direct[nonzero_direct]) ./50.0, 4)
plot!(a_range, poly.(a_range), label = "direct")
savefig("./rate_function.png")
