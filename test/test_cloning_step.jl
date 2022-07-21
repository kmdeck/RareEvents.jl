using Test
using StatsBase
using Random
import RareEvents: sample_and_rewrite_history!


nensemble = 500
nsteps = 1
ensemble = [ones(nsteps)*k for k in 1:nensemble]
frequencies = Array(1:nensemble) # P(k) = k/norm
idx_current = nsteps
sample_and_rewrite_history!(ensemble, frequencies, idx_current)

normalization = sum(frequencies)#(nensemble^2.0-1^2.0)/2.0
probabilities = frequencies ./ normalization
expected_mean = sum(frequencies .* probabilities)#(nensemble^3.0 - 1^3.0)/3.0/normalization
σ_mean = sqrt(sum((frequencies .- expected_mean).^2.0 .* probabilities))./sqrt(nensemble)
@test abs(mean(ensemble)[1] .-expected_mean) < σ_mean*3.0