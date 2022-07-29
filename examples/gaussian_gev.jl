using RareEvents
using SpecialFunctions
using Plots
using Random
using Statistics

nsamples = Int(1e4)
samples = randn(nsamples)
bin = 0.1
magnitude = Array(0.0:bin:5.0)

plot1 = plot()
gaussian_cdf = 1.0 .- 0.5 .*(1 .+erf.(magnitude ./2 .^0.5))
plot!(plot1, magnitude, gaussian_cdf, label = "Gaussian")
plot!(xrange = [0,5.0])
plot!(yaxis = :log)
plot!(ylabel = "Log10(1- cdf)")
plot!(xlabel = "magnitude")
plot!(title = "Effect of block length on GEV fit; Fixed data size")
plot2=plot()
plot!(plot2, xlabel = "magnitude", ylabel = "P(X_{block length} ≤ a)", title = "Non-convergence of block maxima CDF")

plot3=plot()
plot!(plot3, xlabel = "magnitude", ylabel = "P((X_{block length}-μ_n)/σ_n ≤ a)", title = "Convergence of shifted/scaled block maxima CDF")
plot!(plot2, legend = :topleft)
block_lengths = [10,100,1000,10000]
for block_length in block_lengths
    draws = randn(1000*block_length)
    blocks = block_maxima(draws, block_length)
    params, nll = fit_gev(blocks, [std(blocks), mean(blocks), 0.05])
    plot!(plot2, sort(blocks), Array(1:1:length(blocks))./length(blocks), label = string("block length = ", string(block_length)))
    plot!(plot3, (sort(blocks)  .- params[2])./params[3], Array(1:1:length(blocks))./length(blocks), label = string("block length = ", string(block_length)))
    cdf= (gev_cdf.(magnitude, params[1], params[2], params[3])).^(1.0/block_length)
    plot!(plot1, magnitude[cdf .< 1], 1.0 .- cdf[cdf .<1], label = string("GEV, block length = ", string(block_length)))
end

savefig(plot1, "gaussian_cdf.png")
savefig(plot2, "gaussian_block_maxima_cdf.png")
savefig(plot3, "gaussian_scaled_block_maxima_cdf.png")



## MCMC
block_length = 10
draws = randn(1000*block_length)
blocks = block_maxima(draws, block_length)
params, nll = fit_gev(blocks, [std(blocks), mean(blocks), 0.05])
chain = evolve_mcmc(blocks, params, [0.01, 0.01, 0.005], log_likelihood_gev, 5000)


# propagate uncertainty to 
for s in 1:100
    index = Int64(ceil(rand()*length(chain)))
    p = chain[index]
    cdf= (gev_cdf.(magnitude, p[1], p[2], p[3])).^(1.0/block_length)
    plot!(plot1, magnitude, 1.0 .- cdf .+ eps(Float64), label = "")
end
