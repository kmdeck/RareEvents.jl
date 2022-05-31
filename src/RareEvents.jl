module RareEvents
using Statistics
using StatsBase
using Distributed
using Random
# Code questions
# This algorithm is fairly convoluted; how to make more readable?
# Speed up
# Repackage the trajectory output in ensemble, hard to access/plot?. What about for different output types (CC Field Vector..)
# make less dep on DiffEQ eventually store a model? rather than a problem

# Conceptual questions
# Why does k = 0.3 give me ~a of 0.3, not 0.6? a~k?
# How to derive their formula?

export RareEventSampler, run!, moving_average!, return_curve, ensemble_statistics, N_event
"""
    moving_average!(A::Vector{FT},timeseries::Vector{FT}, window::Int)
                   where {FT<:AbstractFloat}

Computes the moving average 1/T ∫_t^(t+T) timeseries(t') dt',
which is equal to the mean of timeseries[i:i+window],
where window = T/dt, and A is the array where the resulting
timeseries of moving averages is stored.
"""
function moving_average!(A::Vector{FT},timeseries::Vector{FT}, window::Int) where {FT<:AbstractFloat}
    for i in 1:length(A)
        A[i] = mean(timeseries[i:i+window])
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

Lastly, the return curve formula for the paper agrees with this
simple estimate for rare events (small fraction exceeding), but
with poor sampling statistics, can give logarithms(negative numbers)
for frequent events. I also am not sure how to derive it.
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
    #return_time_paper = ΔT ./ log.(1.0 .- average_fraction_exceeding[1:end-1])
    return sorted[1:end-1],  return_time_naive, σ_rtn_naive
end

function N_event(a_mn::Matrix{FT},
                 likelihood_ratio::Vector{FT},
                 magnitude::Vector{FT}
                 ) where {FT <: AbstractFloat}
    Na = zeros(length(magnitude))
    for i in 1:length(magnitude)-1
        mask = (a_mn .< magnitude[i+1]) .& ( a_mn .>= magnitude[i])
        Na[i] = sum(sum(mask, dims = 2) .* likelihood_ratio)
    end
    return Na
end

    
"""
    RareEventSampler{FT<:AbstractFloat, ST,P, ES}

The struct holding the necessary information for
running the rare event algorithm.

Note that you should pick dt, τ, and the length of the
integration such that
τ/dt = Integer
(tf-t0)/dt = Integer.
"""
struct RareEventSampler{FT<:AbstractFloat, ST,P}
    "How often output is saved from the integrations"
    dt::FT
    "The total time span for each trajectory"
    tspan::Tuple{FT,FT}
    "The number of trajectories in the ensemble"
    N::Int
    "Time interval after which a new trajectory is proposed: Must be integer * dt"
    τ::FT
    "The ensemble, which stores all output from all trajectories"
    ensemble::ST
    "The dynamical system evolve function, for a single trajectory"
    evolve_single_trajectory::P
    "The score function to use when resampling trajectories"
    score::Function
    "The standard deviation of the white noise added to each trajectory"
    ϵ::FT
    "Vector storing the normalization of the scores R at each resample"
    R::Vector{FT}
end

"""
    RareEventSampler{FT}(dt::FT,
                         d::Int,
                         u0::U,
                         tspan::Tuple{FT,FT},
                         N::Int,
                         Nτ::Int,
                         evolve_single_trajectory::P,
                         score::Function,
                         ϵ::FT) where{FT,U, P}

Constructor for the RareEventSampler object.

The user should pass
- a timestep `dt`, which is also the frequency of output,
- the initial condition `u0` for the state vector (currently assumed to be the same for
all elements of the ensemble),
-  the time span tuple `tspan = (t0,tf)`, which 
should be satisfy `(tf-t0)` being an integer number of timesteps and an integer
number of the resampling time `τ`,
- the number of ensemble members `N`
- the `evolve` function for a single trajectory
- the `score` function
- `ϵ` the amplitude of the white noise to add to each element of the state vector

The `evolve` function takes in initial conditions and a tspan, as well as integration
keyword arguments, as specific to your problem, and returns the solution at intervals of `dt`.

 Note that to run with pmap, we actually require the following:
`evolve_wrapper(tspan) = (u0) ->  evolve_single_trajectory(u0, tspan; alg_kwargs...)`
with `evolve_wrapper` passed in to the RareEventSampler.

Lastly, the score function acts on single trajectories and outputs a single real number.
 
"""
function RareEventSampler{FT}(dt::FT,
                              u0::U,
                              tspan::Tuple{FT,FT},
                              N::Int,
                              Nτ::Int,
                              evolve_single_trajectory::P,
                              score::Function,
                              ϵ::FT) where{FT,U, P}
    τ = dt*Nτ
    # The user should pass a tspan that requires an integer number of τ,
    # an an integer number of dt:
    @assert (tspan[2]-tspan[1])/dt- Int((tspan[2]-tspan[1])/dt) == 0.0
    @assert (tspan[2]-tspan[1])/τ- Int((tspan[2]-tspan[1])/τ) == 0.0
    
    Nsteps = Int((tspan[2]-tspan[1])/dt)+1
    trajectory = Vector{typeof(u0)}([similar(u0) for _ in 1:Nsteps])
    ensemble = Vector{typeof(trajectory)}([copy(trajectory) for _ in 1:N])
    [ensemble[k][1] .= u0 for k in 1:N];

    R = zeros(Int((Nsteps-1)/Nτ))

    args = (dt, tspan, N, τ, ensemble, evolve_single_trajectory, score, ϵ, R)
    return RareEventSampler{FT, typeof(ensemble), typeof(evolve_single_trajectory)}(args...)
end

# Begin at i = 0
function run!(sim)
    u1 = deepcopy([sim.ensemble[k][1] for k in 1:sim.N])
    u2 = deepcopy([sim.ensemble[k][1] for k in 1:sim.N])
    t0 = sim.tspan[1]
    iterations = Int(round(sim.tspan[2]/sim.τ))
    Nτ = Int(sim.τ/sim.dt)
    map(0:(iterations-1)) do i
        t2 = t0+(i+1)*sim.τ
        t1 =  t0+ i*sim.τ
        i1 = 1+(i)*Nτ
        i2 = 1+(i+1)*Nτ
        solutions = integrate(sim, u1,t1,t2)
        copies = update_ensemble_and_score!(sim, solutions, i1, i2, i)
        sample_and_rewrite_history!(sim, u2, copies, i1, i2, i)
        i = i+1
        u1 = copy(u2)
    end
end

function integrate(sim, u1,t1,t2)
    # Currently 30% slower than EnsembleProblem in DifferentialEquations
    # It is 2x as slow to run distributed = true vs distributed = false
     return pmap(sim.evolve_single_trajectory((t1,t2)),u1;distributed = false)
end

function update_ensemble_and_score!(sim, solutions,i1,i2, i)
    # We need to loop over the original ensemble to get normalized weights
    scores = zeros(sim.N)
    for k in 1:sim.N
        sim.ensemble[k][i1:i2] .= solutions[k]
        scores[k] = sim.score(solutions[k])[1]
    end
    sim.R[i+1] = mean(scores)
    weights = scores ./ mean(scores)
    copies = Int.(floor.(weights .+ rand(sim.N)))
    return copies
end

function sample_and_rewrite_history!(sim, u2, copies, i1, i2, i)
    # We need to loop over the data to sample from it N times
    N = sim.N
    array_u = [[sim.ensemble[k][1:i2],] for k in 1:N]
    # If copies is zero, it is cut. Then shuffle the cloned set.
    # Selecting from it in order = a random sample
    copied_sample = shuffle!(vcat(sample.(array_u, copies)...))
    N_c = sum(copies)
    scores = zeros(N)

    # Dimensionality of state vector
    d = length(u2[1])
    # We need to loop over it to reset the ensemble to the samples.
    if N_c > N
        for k in 1:N
            sim.ensemble[k][1:i2] .= copied_sample[k]
            scores[k] = sim.score(sim.ensemble[k][i1:i2])[1]
            u2[k] = sim.ensemble[k][i2]+ randn(d)*sim.ϵ
        end
    else
        for k in 1:N_c
            sim.ensemble[k][1:i2] .= copied_sample[k]
            scores[k] = sim.score(sim.ensemble[k][i1:i2])[1]
            u2[k] = sim.ensemble[k][i2]+ randn(d)*sim.ϵ
            if k+N_c <= N # start over at the beginning if sum(copies) <N
                sim.ensemble[k+N_c][1:i2] .= copied_sample[k]
                scores[k+N_c] = sim.score(sim.ensemble[k][i1:i2])[1]
                u2[k+N_c] = sim.ensemble[k][i2]+ randn(d)*sim.ϵ
            end
        end
    end
    # This should come before the resampling occurs
    #sim.R[i+1] = mean(scores)
end

function ensemble_statistics(ensemble, state_index)
    M = map(x -> x[state_index], reduce(hcat,ensemble))
    means = mean(M, dims = 2)
    stds = std(M, dims = 2)
    return means[:], stds[:]
end
end
