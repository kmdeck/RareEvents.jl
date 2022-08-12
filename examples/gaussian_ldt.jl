# This script explores the effect of averaging block length
# in the LDT fitting method of estimating probabilities
# of rare events, using a iid samples from a Gaussian distribution.

using RareEvents
using SpecialFunctions
using Plots
using Random
using Statistics

# Initialize plots
plot1 = plot()
plot!(plot1, xlabel = "a", ylabel = "I(a)")
plot2 = plot()
plot!(plot2, xlabel = "a", ylabel = "P(a)")


bin = 0.05
magnitude = Array(-2.0:bin:2.0)
averaging_lengths = [3,10,30,100, 300]
probability = zeros(length(magnitude)-1)
n_blocks = 10000
for averaging_length in averaging_lengths
    draws = randn(n_blocks*averaging_length)
    blocks = block_mean(draws, averaging_length)
    for i in 1:length(magnitude)-1
        mask = (blocks .< magnitude[i+1]) .& (blocks .> magnitude[i])
        probability[i] = sum(mask) ./ n_blocks
    end
    rate  = -log.(probability) ./ averaging_length
    
    plot!(plot1, (magnitude[2:end] .+ magnitude[1:end-1])./2.0, rate .- minimum(rate), label = string(" n = ", string(averaging_length)))
    plot!(plot2, (magnitude[2:end] .+ magnitude[1:end-1])./2.0, probability, label = string(" n = ", string(averaging_length)))
end

savefig(plot1, "./ldt_rate_with_n.png")
savefig(plot2, "./ldt_prob_with_n.png")
