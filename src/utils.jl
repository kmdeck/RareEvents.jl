export moving_average!, return_curve, ensemble_statistics, N_event, likelihood_ratio

"""
    moving_average!(A::Vector,timeseries::Vector, window::Int)

Computes the moving average 1/T ∫_t^(t+T) timeseries(t') dt',
which is equal to the mean of timeseries[i:i+window],
where window = T/dt, and A is the array where the resulting
timeseries of moving averages is stored.

For a timeseries which is a vector of vectors, this returns the 
moving average for each dimension.
"""
function moving_average!(A::Matrix,timeseries::Vector, window::Int)
    for i in 1:size(A)[1]
        A[i,:] = mean(timeseries[i:i+window-1])
    end
end
"""
     return_curve(a_m::Vector{FT},
                  ΔT::FT,
                  p::Vector{FT}
                  ) where {FT <:AbstractFloat}

Estimates the return curve, returning an event magnitude 
array and a return time array, by looking at the maximum 
value of M timeseries of interest
of length N = ΔT/dt, where `dt` is the timestep beween saved values.

The input arguments are: (1) the `a_m` vector consisting of
the maxima of M trajectory segments
, (2) the timespan ΔT  over which the maxima
are computed, and (3) a vector of probabilities `p`,
corresponding to the probability of each segment under the model.

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
                      likelihood_ratio::Vector{FT}
                      ) where {FT<:AbstractFloat}
    sort_indices = reverse(sortperm(a_m))
    sorted = a_m[sort_indices]
    likelihood_ratio_sorted = likelihood_ratio[sort_indices] 
    M = length(a_m)
    average_fraction_exceeding = cumsum(likelihood_ratio_sorted)./M
    return_time_naive = ΔT ./  average_fraction_exceeding[1:end-1]
    avg_f_squared = cumsum(likelihood_ratio_sorted.^2.0)./M
    σ_avg_f_exceeding = sqrt.(avg_f_squared .-  average_fraction_exceeding.^2.0)/sqrt(M)
    σ_rtn_naive = return_time_naive .* σ_avg_f_exceeding[1:end-1]./average_fraction_exceeding[1:end-1]
    return sorted[1:end-1],  return_time_naive, σ_rtn_naive
end

function N_event(a_mn::Matrix{FT},
                 likelihood_ratio::FT,
                 magnitude::Vector{FT}
                 ) where {FT <: AbstractFloat}
    Na = Matrix{FT}(undef,length(magnitude)-1,size(a_mn)[2])
    for i in 1:length(magnitude)-1
        mask = (a_mn .< magnitude[i+1]) .& ( a_mn .>= magnitude[i])
        Na[i,:] = sum(mask, dims = 1) * likelihood_ratio
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
