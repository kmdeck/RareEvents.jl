using Test
using StatsBase
using Random
using RareEvents: log_likelihood_gev, fit_gev, gev_pdf, gev_cdf

@testset "GEV cumulative distribution" begin
    σ = 1.0
    μ = 0.0
    @test gev_cdf(1.0, σ,μ, -2.0) == 1.0
    @test gev_cdf(-1.0, σ, μ, 2.0) == 0.0
    x = [0.1,0.2,0.3]
    z = 1.0 .+ 1.0 .*x
    @test gev_cdf.([0.1,0.2,0.3], σ, μ, 1.0) == exp.(.-z.^(-1.0/1.0))
    @test gev_cdf.([0.1,0.2,0.3], σ, μ, 0.0) == exp.(.-exp.(.-x))    
end

