using DelimitedFiles
using Plots
using RareEvents
using Statistics
using Polynomials

path = string("ornstein_uhlenbeck_output/algorithm/k_")
files = [string(path, "0.3_N.csv"),string(path, "0.5_N.csv"),string(path, "0.3_N_mixed_score.csv"),string(path, "0.5_N_mixed_score.csv"),string(path, "0.7_N_mixed_score.csv"),string(path, "0.5_N_mixed_score10.csv")]
titles = ["k = 0.3", "k = 0.5", "k = 0.3 mixed", "k = 0.5 mixed", "k = 0.7 mixed", "k=0.5, d= 10, mixed"]
plot1 = plot()
plot2 = plot()
plot3 = plot()

d = 2
a_range = Array(0.0:0.03:1.0)
a = 0.5*(a_range[1:end-1] .+ a_range[2:end])
bins = length(a_range)-1
for i in [1,2,3,4]
    Na = readdlm(files[i])
    for j in 2:2
        N = mean(Na[:,(j-1)*bins+1:j*bins],dims = 1)
        σN =  std(Na[:,(j-1)*bins+1:j*bins],dims = 1)
        p =  N[:]./sum(N[:])
        σp = σN[:]./sum(N[:])
        nonzero = p .!= 0.0
        plot!(plot1,a[nonzero], log10.(p[nonzero]), ribbon = σp[nonzero] ./ p[nonzero] ./log(10.0), label = string(titles[i], ", u",string(j)))
        
        RT = 50.0./p[nonzero]
        plot!(plot2, a[nonzero], log10.(RT), ribbon = σp[nonzero]./p[nonzero]./log(10.0), label = string(titles[i], ", u",string(j)))
        
        #poly = fit(a[nonzero], -log.(p[nonzero]) ./50.0, 2)
        plot!(plot3, a[nonzero], -log.(p[nonzero]) ./50.0,label = string(titles[i], ", u",string(j)))
       # plot!(plot3, a, poly.(a))
    end
    
end
σ = 0.141
plot!(plot1,a_range[:], log10.(0.03/sqrt(2.0*π*σ^2.0) .* exp.(-1.0 .*(a_range[:] .^2.0 ./(2.0*σ^2.0)))), label= "Gaussian")
plot!(plot1, ylabel = "P(a)Δa", xlabel = "a")
plot!(plot1, legend = :bottomleft)
plot!(plot2,ylim = [0,13], xlim = [0,1.0])
plot!(plot2,legend = :bottomright)
plot!(plot2,ylabel = "Log10(Return Time of Event of Magnitude a)")
plot!(plot2,xlabel = "magnitude of event a")
plot!(plot3,xlabel = "a")
plot!(plot3,ylabel = "I(a)")
######

#=

# Using the max method
a_k05 = readdlm("k_05_bootstrap_refactored.csv")
lr_k05 = readdlm("k_05_bootstrap_lr_refactored.csv")
a_k03 = readdlm("k_03_bootstrap_refactored.csv")
lr_k03 = readdlm("k_03_bootstrap_lr_refactored.csv")



a = a_k03
lr = lr_k03
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
plot!(final_a, final_r, label = "k=0.3 via max")

savefig("./rt_event_3x.png")
=#
