# This script explores the effect of block length
# in the GEV fitting method of estimating probabilities
# of rare events, using a iid samples from a Gaussian distribution.

using RareEvents
using SpecialFunctions
using Plots
using Random
using Statistics

nsamples = Int(1e4)
samples = randn(nsamples)
bin = 0.1
magnitude = Array(0.0:bin:5.0)
gaussian_cdf = 1.0 .- 0.5 .*(1 .+erf.(magnitude ./2 .^0.5))

# Initialize plots
plot1 = plot()
plot!(plot1,xrange = [0,5.0])
plot!(plot1,yaxis = :log)
plot!(plot1,ylabel = "Log10(1- cdf)")
plot!(plot1,xlabel = "magnitude")
plot!(plot1,title = "Effect of block length on GEV fit; Fixed data size")
plot!(plot1, magnitude, gaussian_cdf, label = "Gaussian")


plot2=plot()
plot!(plot2, xlabel = "magnitude", ylabel = "P(X_{block length} ≤ a)", title = "Non-convergence of block maxima CDF")
plot!(plot2, legend = :topleft)

plot3=plot()
plot!(plot3, xlabel = "magnitude", ylabel = "P((X_{block length}-μ_n)/σ_n ≤ a)", title = "Convergence of shifted/scaled block maxima CDF")

# Loop over block length; for each block length, add to plots.
block_lengths = [10,100,1000,10000]
for block_length in block_lengths
    draws = randn(1000*block_length)
    blocks = block_maxima(draws, block_length)
    params, nll = fit_gev(blocks, [std(blocks), mean(blocks), 0.05])
    cdf= (gev_cdf.(magnitude, params[1], params[2], params[3])).^(1.0/block_length)

    plot!(plot1, magnitude[cdf .< 1], 1.0 .- cdf[cdf .<1], label = string("GEV, block length = ", string(block_length)))
    plot!(plot2, sort(blocks), Array(1:1:length(blocks))./length(blocks), label = string("block length = ", string(block_length)))
    plot!(plot3, (sort(blocks)  .- params[2])./params[3], Array(1:1:length(blocks))./length(blocks), label = string("block length = ", string(block_length)))

end

savefig(plot1, "gaussian_cdf.png")
savefig(plot2, "gaussian_block_maxima_cdf.png")
savefig(plot3, "gaussian_scaled_block_maxima_cdf.png")
