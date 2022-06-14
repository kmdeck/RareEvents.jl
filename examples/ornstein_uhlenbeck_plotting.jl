using DelimitedFiles
using Plots
using RareEvents
using Statistics

a_k05 = readdlm("k_05_bootstrap.csv")
lr_k05 = readdlm("k_05_bootstrap_lr.csv")
a_k03 = readdlm("k_03_bootstrap.csv")
lr_k03 = readdlm("k_03_bootstrap_lr.csv")

plot1 = plot()


a = a_k03
lr = lr_k03
N = 600
T_a = 100
T = 50.0

iters = 100
em = zeros((N-1)*iters)
em = reshape(em, ((N-1),iters))
r = similar(em)
for i in 1:iters
    event_magnitude, rtn, _= return_curve(a[:,i], T_a-T, lr[:,i]);
    em[:,i] = event_magnitude
    r[:,i] = rtn
end
a_range = Array(minimum(em):0.03:maximum(em))
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
plot!(final_a, final_r, ribbon = final_σr, label = "k=0.3")    

event_magnitude, rtn, σ_rtn= return_curve(a[:], T_a-T, lr[:]);
# x = rtn
# y = log10(x)
# dy = dx/x/log(10)
plot!(event_magnitude, log10.(rtn), ribbon = σ_rtn ./rtn ./log(10.0))

plot!(ylim = [1, 12], yticks = [1,2,4,6,8,10,12], xticks = [0,0.4,0.8], xlim = [0, 1])


direct_a_m = readdlm("direct_1e7_a_m_dt_01.csv")
# timestep of 0.1, segments of ΔT=50 -> 500 moving average samples
# This needs to be the same.
direct_event_magnitude, direct_rtn, direct_σ_rtn= return_curve(direct_a_m[:], T_a-T, ones(length(direct_a_m[:])));

plot!(direct_event_magnitude, log10.(direct_rtn), ribbon = direct_σ_rtn./ direct_rtn, label = "Direct")
plot!(legend = :bottomright)

### Analytic estimate does not work yet
σ = 0.142 # Computed standard deviation of moving average of x(t), from a direct integration,
# over time T=50. It has a mean of 0. This is independent of dt if dt is small enough.
# If Y = moving average, Y~ N(0, σ^2) = P_0
μ_k = k*σ^2.0*T # Mean of moving average distribution resulting from algorithm. 

# P(MA > a) related to erfc(a) = 0.5*erfc(a/√2/σ)
# To assign a return time, we need to know the rate of sampling MA ~ P_O(MA)

analytic(a, σ, t, μ) = t/(0.5*SpecialFunctions.erfc((a-μ)/√2/σ))

analytic_rtn = analytic.(final_a,σ, 50.0, 0.0)
plot!(final_a, log10.(analytic_rtn),color = "purple", label = "analytic estimate")
