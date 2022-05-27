using DelimitedFiles
using Plots
using RareEvents
using Statistics

a_k05 = readdlm("k_05_bootstrap.csv")
lr_k05 = readdlm("k_05_bootstrap_lr.csv")
a_k03 = readdlm("k_03_bootstrap.csv")
lr_k03 = readdlm("k_03_bootstrap_lr.csv")

plot1 = plot()


a = a_k05
lr = lr_k05
N = 600
T_a = 100
T = 50.0


em = zeros((N-1)*30)
em = reshape(em, ((N-1),30))
r = similar(em)
σr = similar(em)
for i in 1:30
    event_magnitude, rtn, σ_rtn= return_curve(a[:,i], T_a-T, lr[:,i]);
    em[:,i] = event_magnitude
    r[:,i] = rtn
    σr[:,i] = σ_rtn
    #plot!(plot1, em[:,i], log10.(r[:,i]), label = "")
end
a_range = Array(minimum(em):0.03:maximum(em))
mean_a = zeros(length(a_range)-1)
mean_rtn = zeros(length(a_range)-1)
std_rtn = zeros(length(a_range)-1)
for i in 1:(length(a_range)-1)
    mask = (em[:] .< a_range[i+1]) .& ( em[:] .>= a_range[i])
    if sum(mask) >0
        mean_a[i] = mean(em[:][mask])
        std_rtn[i] = std(log10.(r[:][mask]))
        mean_rtn[i] = mean(log10.(r[:][mask]))
    end
    
end
nonzero = (mean_a .!= 0.0) .& (isnan.(std_rtn) .== 0)
final_a = mean_a[nonzero]
final_r = mean_rtn[nonzero]
final_σr = std_rtn[nonzero]
plot!(final_a, final_r, ribbon = final_σr, label = "k=0.5")    
#plot!(em[:,1], log10.(r[:,1]), yerror = σr[:,1]./ r[:,1], label = "k = 0.5 instance")


plot!(ylim = [1, 12], yticks = [1,2,4,6,8,10,12], xticks = [0,0.4,0.8], xlim = [0, 1])


direct_a_m = readdlm("direct_1e7_a_m.csv")
direct_event_magnitude, direct_rtn, direct_σ_rtn= return_curve(direct_a_m[:], T_a-T, ones(length(direct_a_m[:])));

plot!(direct_event_magnitude, log10.(direct_rtn), ribbon = direct_σ_rtn./ direct_rtn, label = "Direct")
plot!(legend = :bottomright)

#=
### Analytic estimate does not work yet
σ = 0.142 # Computed standard deviation of moving average of x(t), from a direct integration,
# over time T=50. It has a mean of 0. This is independent of dt if dt is small enough.
# If Y = moving average, Y~ N(0, σ^2) = P_0
μ_k = k*σ^2.0*T # Mean of moving average distribution resulting from algorithm. 
# The standard dev is the same. so Y_k ∼P_k = N(μ_k, σ^2).
# So, we can check that the moving_average_matrix has a mean and standard dev of μ_k and σ:
mean(moving_average_matrix[:])
std(moving_average_matrix[:])
# We want return periods for moving average but under P_0
# erfc(x) = 2/√π*∫_x^∞ e^(-t^2) dt

analytic(a, σ, ΔT, μ) = 2.0*ΔT/SpecialFunctions.erfc((a-μ)/√2/σ)
analytic_rtn = analytic.(event_magnitude,σ, T_a-T, 0.0)
plot!(log10.(analytic_rtn), event_magnitude, color = "blue", label = "analytic estimate")

# Think more about:  does this hold even though we are looking
# event mangitude = the max of the moving average over q samples?
# The analytic expectation is FOR the moving average, not the max?
# P_k(max(Y) = η) ∝ (q choose 1 ) F_k(η)^(q-1) P_k(η) ?
# where F_k(η) is the cumulative distribution ∫_-∞^η P_k(s) ds
=#

