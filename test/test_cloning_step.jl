using Test
using StatsBase
using Random
import RareEvents: sample_and_rewrite_history!, orig_sample_and_rewrite_history!, compute_ncopies!

nensemble = 500
nsteps = 1
idx_current = nsteps

@testset "old v new" begin
    rng = MersenneTwister(1234);
    ensemble_new = [ones(nsteps)*k for k in 1:nensemble]
    frequencies = Array(1:nensemble) # P(k) = k/norm
    sample_and_rewrite_history!(ensemble_new, frequencies, idx_current, rng)
    
    rng = MersenneTwister(1234);
    ensemble_old = [ones(nsteps)*k for k in 1:nensemble]
    frequencies = Array(1:nensemble) # P(k) = k/norm
    orig_sample_and_rewrite_history!(ensemble_old, frequencies, idx_current, rng)
    diff = sort(vcat(ensemble_new...)) .== sort(vcat(ensemble_old...))
    @test sum(diff) .== nensemble

end



@testset "Probabilistic test" begin
    rng = MersenneTwister(1234);
    ensemble = [ones(nsteps)*k for k in 1:nensemble]
    frequencies = Array(1:nensemble) # P(k) = k/norm
    sample_and_rewrite_history!(ensemble, frequencies, idx_current, rng)
    
    normalization = sum(frequencies)#(nensemble^2.0-1^2.0)/2.0
    probabilities = frequencies ./ normalization
    expected_mean = sum(frequencies .* probabilities)#(nensemble^3.0 - 1^3.0)/3.0/normalization
    σ_mean = sqrt(sum((frequencies .- expected_mean).^2.0 .* probabilities))./sqrt(nensemble)
    @test abs(mean(ensemble)[1] .-expected_mean) < σ_mean*3.0
end

@testset "ncopies = 1 -> no change" begin
    rng = MersenneTwister(1234);
    ensemble = [ones(nsteps)*k for k in 1:nensemble]
    orig_ensemble = copy(ensemble)
    ncopies = Int64.(ones(nensemble))
    sample_and_rewrite_history!(ensemble, ncopies, idx_current, rng)
    diff = sort(vcat(ensemble...)) .== sort(vcat(orig_ensemble...))
    @test sum(diff) .== nensemble
end


@testset "ncopies" begin
    rng= MersenneTwister(1);
    nensemble = 10
    draw = rand(rng, nensemble)
    ncopies = ones(nensemble)
    scores = Array(1:1:nensemble)
    expected_ncopies = Int.(floor.(scores ./mean(scores) .+ draw))
    rng= MersenneTwister(1);
    compute_ncopies!(ncopies, scores,mean(scores), nensemble, rng)
    similar = ncopies .== expected_ncopies
    @test sum(similar) == nensemble
end
