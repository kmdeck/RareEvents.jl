export return_curve, ensemble_statistics, count_events_per_bin, likelihood_ratio, block_mean, block_maxima

"""
     return_curve(a_m::Vector{FT},
                  ΔT::FT,
                  lr::Vector{FT}
                  ) where {FT <:AbstractFloat}

Estimates the return curve, returning an event magnitude 
array and a return time array, by looking at the maximum 
value of M timeseries of interest
of length N = ΔT/dt, where `dt` is the timestep beween saved values.

The input arguments are: (1) the `a_m` vector consisting of
the maxima of M trajectory segments
, (2) the timespan ΔT  over which the maxima
are computed, and (3) a vector of likelihood raiots `lr`,
equal to the ratio of the target distribution to the importance
distribution
.
This estimates the return curve using what I would predict
from a Poisson process, Return Time (a) = ΔT*M/q(a), where 
q is the observed number of segments with maxima
exceeding the threshold a. This computes the expected
number q(a) for an array of `a` values, and hence gets
an array of return times.

Under the original distribution p_0, we would have
q(a)/M = 1/M * ∑_i=1^M θ(a_i>a). If we sort the maxima
a_i in descending order, we can obtain a curve by noting that
q(a_j)/M = 1/M * ∑_i=1^j 1 = rank of a_j/M.

Using importance sampling: supposed we instead of M samples
from a distribution p_k. Then the expected number of events 
to exceed the threshold a under the model is given by
q(a)/M = 1/M * ∑_i=1^M θ(a_i>a) (p_0/p_k)_i, or
q(a_j)/M = 1/M * ∑_i=1^j (p_0/p_k)_i, if the a_j
are ranked (with their associated probablity ratios) 
in descending order.
"""
function return_curve(a_m::Vector{FT},
                      ΔT::FT,
                      lr::Vector{FT}
                      ) where {FT<:AbstractFloat}
    sort_indices = reverse(sortperm(a_m))
    sorted = a_m[sort_indices]
    lr_sorted = lr[sort_indices] 
    M = length(a_m)
    # γa = P(X > a)
    γ = cumsum(lr_sorted)./M

    ceiling = γ .>=1
    γ[ceiling] .= 0.9999
    
    return_time_naive = ΔT ./  γ
    return_time_paper = -ΔT ./ log.(1.0 .- γ)
    γ² = cumsum(lr_sorted.^2.0)./M
    σ_γ = sqrt.(γ² .-  γ.^2.0)/sqrt(M)
    σ_rtn = return_time_naive .* σ_γ./γ
    return sorted,  return_time_naive, return_time_paper, σ_rtn
end

function count_events_per_bin(samples::Vector{FT},
                 likelihood_ratio::FT,
                 magnitude::Vector{FT}
                 ) where {FT <: AbstractFloat}
    Na = zeros(length(magnitude))
    for i in 1:length(magnitude)-1
        mask = (samples .< magnitude[i+1]) .& ( samples .>= magnitude[i])
        Na[i] = sum(sum(mask, dims = 2) .* likelihood_ratio)
    end
    return Na
end



function ensemble_statistics(ensemble, state_index)
    M = map(x -> x[state_index], reduce(hcat,ensemble))
    means = mean(M, dims = 2)
    stds = std(M, dims = 2)
    return means[:], stds[:]
end

function likelihood_ratio(sim, trajectory, λ, T_a)
    return exp(T_a*λ)/score(sim,trajectory)
end


"""
    block_maxima(x::Vector, m::Int)

Computes the non-overlapping moving maximum of a vector 
x, using a window of length m (in indices).
"""
function block_maxima(x::Vector, m::Int)
    blocks = collect(Base.Iterators.partition(x, m))
    maxima = map(y-> maximum(y), blocks)
    output_length = Int(floor(length(x)/m))
    return maxima[1:output_length]
end

"""
    block_mean(x::Vector, m::Int)

Computes the non-overlapping moving average of a vector 
x, using a window of length m (in indices).
"""
function block_mean(x::Vector, m::Int)
    blocks = collect(Base.Iterators.partition(x, m))
    maxima = map(y-> mean(y), blocks)
    output_length = Int(floor(length(x)/m))
    return maxima[1:output_length]
end
