using DelimitedFiles
using Plots
using RareEvents
using Statistics

Na_k03 = readdlm("k_03_bootstrap_Na.csv")
Na_k05 = readdlm("k_05_bootstrap_Na.csv")
Na_direct = readdlm("direct_1e7_Na_dt_01.csv")
N03 = mean(Na_k03, dims =2 )
σN03 = std(Na_k03, dims =2 )
p03 = N03[:]./sum(N03)
σp03 = σN03[:] / sum(N03)
N05 = mean(Na_k05, dims =2 )
σN05 = std(Na_k05, dims =2 )
p05 = N05[:]./sum(N05)
σp05 = σN05[:] / sum(N05)
a_range = Array(0.0:0.03:1.0)
p_direct = Na_direct[:]./sum(Na_direct)
plot1 = plot()
# if y = log10(p), σy = σp/p/ln(10)
plot!(a_range[:], log10.(p03), ribbon = σp03[:] ./ p03[:] ./log(10.0), label = "k =0.3")
plot!(a_range[:], log10.(p05), ribbon = σp05[:] ./ p05[:] ./log(10.0), label = "k =0.5")
plot!(a_range[:], log10.(p_direct), label = "Direct")
plot!(legend = :bottomleft)

σ = 0.142
plot!(a_range[:], log10.(0.03/sqrt(2.0*π*σ^2.0) .* exp.(-1.0 .*(a_range[:] .^2.0 ./(2.0*σ^2.0)))), label= "Gaussian")
plot!(ylabel = "P(a)Δa", xlabel = "a")
savefig("./p_event.png")

nonzero03 = p03 .!= 0.0
RT03 = 50.0./p03[nonzero03]
nonzero05 = p05 .!=0.0
RT05 = 50.0./p05[nonzero05]
RT_direct = 50.0 ./ p_direct[nonzero_direct]
nonzero_direct = p_direct .!=0.0
plot(a_range[nonzero03], log10.(RT03), ribbon = σp03[nonzero03]./p03[nonzero03]./log(10.0), label = "k=0.3")
plot!(a_range[nonzero05], log10.(RT05), ribbon = σp05[nonzero05]./p05[nonzero05]./log(10.0), label = "k=0.5")

plot!(a_range[nonzero_direct], log10.(RT_direct), label = "Direct")
plot!(ylim = [0,10], xlim = [0,1.0])
plot!(legend = :bottomright)
plot!(ylabel = "Log10(Return Time of Event of Magnitude a)")
plot!(xlabel = "magnitude of event a")
savefig("./rt_event.png"
