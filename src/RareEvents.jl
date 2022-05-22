module RareEvents
using Statistics
using StatsBase
using DifferentialEquations
using DifferentialEquations.EnsembleAnalysis
using Random
using Plots
# Code questions
# This algorithm is fairly convoluted; how to make more readable?
# Why doesnt the "clean" approach result in the same output?
# Speed up
# Repackage the trajectory output in ensemble, hard to access/plot.
# make less dep on DiffEQ eventually store a model? rather than a problem

# Conceptual questions
# Why does k = 0.3 give me ~a of 0.3, not 0.6? a~k?
# How to derive their formula?
# Why does their formula not work all the time? (poor statistics)
# why do the curves disagree on the low end (poor statistics here!)
# Do return curves depend on k? They shouldnt with enough samples.
# Picking k picks magnitude of event, most samples are from there.
# so k = 0.3 wont do well for magnitude a = 0.6 events, because those are still rare
# similarily, k =0.5 wont do well for a<0.2 events.

# Try 2d related system and see if another score function can ID extremes
# of each.

# Determine analytic result for O-U process and compare.

# Try without using the moving average

export RareEventSampler, run!, moving_average!, return_curve, ensemble_statistics
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
     return_curve(segment_matrix::Matrix{FT},
                  ΔT::FT,
                  p::Vector{FT}
                  ) where {FT <:AbstractFloat}

Estimates the return curve, returning an event magnitude 
array and a return time array, by looking at the maximum 
value of M timeseries of interest
of length N = ΔT/dt, where `dt` is the timestep beween saved values.

The input arguments are: (1) the `segment_matrix` consisting of
segments of trajectories, size = MxN,
with values equal to the quantity you are estimating return
periods for, (2) the timespan ΔT = dt*N over which the maxima
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
    σ_avg_f_exceeding = sqrt.((cumsum(likelihood_ratio_sorted.^2.0)./M .-  average_fraction_exceeding.^2.0)./M)
    σ_rtn_naive = return_time_naive .* σ_avg_f_exceeding[1:end-1]./average_fraction_exceeding[1:end-1]
    #return_time_paper = ΔT ./ log.(1.0 .- average_fraction_exceeding[1:end-1])
    return sorted[1:end-1],  return_time_naive, σ_rtn_naive
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
struct RareEventSampler{FT<:AbstractFloat, ST,P, ES}
    "How often output is saved from the integrations"
    dt::FT
    "The number of prognostic variables of the dynamical system"
    d::Int
    "The total time span for each trajectory"
    tspan::Tuple{FT,FT}
    "The number of trajectories in the ensemble"
    N::Int
    "Time interval after which a new trajectory is proposed"
    τ::FT
    "The ensemble, which stores all output from all trajectories"
    ensemble::ST
    "The dynamical system problem"
    problem::P
    "The score function to use when resampling trajectories"
    score::Function
    "The standard deviation of the white noise added to each trajectory"
    ϵ::FT
    "Vector storing the normalization of the scores R at each resample"
    R::Vector{FT}
    "ensemble score; not currently used or filled"
    ensemble_score::ES
end

function RareEventSampler{FT}(dt::FT,
                              d::Int,
                              tspan::Tuple{FT,FT},
                              N::Int,
                              Nτ::Int,
                              problem::P,
                              score::Function,
                              ϵ::FT) where{FT,P}
    Nsteps = Int(round((tspan[2]-tspan[1])/dt))+1
    R = zeros(Int(round((Nsteps-1)/Nτ)))# be careful
    state = FT.(zeros(d)) .+ NaN
    
    trajectory = Vector{typeof(state)}([copy(state) for _ in 1:Nsteps])
    ensemble = Vector{typeof(trajectory)}([copy(trajectory) for _ in 1:N])
    [ensemble[k][1] .= problem.u0 for k in 1:N];
    ensemble_score = Vector{Vector{FT}}([copy(R) for _ in 1:N])
    τ = dt*Nτ
    args = (dt, d, tspan, N, τ, ensemble,problem, score, ϵ, R, ensemble_score)
    return RareEventSampler{FT, typeof(ensemble), typeof(problem), typeof(ensemble_score)}(args...)
end

# Work on speeding this up?
# We need to loop over the original ensemble to get normalized weights
# We need to loop over the data to sample from it N times
# We need to loop over it to reset the ensemble to the samples.

# Begin at i = 0
function run!(sim, u0)
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
        # update ensemble members and u2 - the initial condition
        # for the next segment t1:t2 - in place according to algorithm
        integrate_and_update!(sim, u2, u1,t2,t1, i2, i1, i)
        i = i+1
        u1 = copy(u2)
    end
end

"""
    integrate_and_update!(sim, u1, t1, t2, i1, i2)
    
Integrate the ensemble of trajectories from u1, at time
t1, to time t2. Score the trajectories based on their 
behavior from t1 to t2, accept or cut them, and return
the initial condition u2 for the next interval.
"""
function integrate_and_update!(sim, u2, u1,t2,t1,i2,i1, i)
    # Update the `problem` in place with the new IC = u1 and new tspan
    # `i` indicates the ensemble member.
    function prob_func(prob,i,repeat)
        remake(prob, u0 = u1[i], tspan = (t1,t2))
    end
    # Create and run the ensemble
    ensemble_problem = EnsembleProblem(sim.problem; prob_func = prob_func)
    ensemble_solution = solve(ensemble_problem,EnsembleThreads(),trajectories=sim.N,saveat = t1:sim.dt:t2)

    # Get the scores for these segments of trajectories
    scores = zeros(sim.N)
    for k in 1:sim.N
        sim.ensemble[k][i1:i2] .= ensemble_solution.u[k].u
        scores[k] = sim.score(ensemble_solution.u[k].u)[1]
    end
    weights = scores ./ mean(scores)
    copies = Int.(floor.(weights .+ rand(sim.N)))
    # Create copied/cut set; sample from it, replace trajectory history
    # and return the final u(t2) = u2 of the accepted set, plus the noise
    # also set the R in `sim` with the `R` of the accepted set.
    sample_and_rewrite_history!(sim, u2, copies, i1, i2, i)
end

function sample_and_rewrite_history!(sim, u2, copies, i1, i2, i)
    N = sim.N
    array_u = [[sim.ensemble[k][1:i2],] for k in 1:N]
    # If copies is zero, it is cut. Then shuffle the cloned set.
    # Then we dont need to sample from it,
    # as selecting from it in order = a random sample
    copied_sample = shuffle!(vcat(sample.(array_u, copies)...))
    N_c = sum(copies)
    scores = zeros(N)
    if N_c > N
        for k in 1:N
            sim.ensemble[k][1:i2] .= copied_sample[k]
            scores[k] = sim.score(sim.ensemble[k][i1:i2])[1]
            u2[k] = sim.ensemble[k][i2]+ randn(sim.d)*sim.ϵ
        end
    else
        for k in 1:N_c
            sim.ensemble[k][1:i2] .= copied_sample[k]
            # This doesnt work because we need to replace the scores
            # if the whole trajectory is replaced
         #   sim.ensemble_score[k][i+1] = sim.score(sim.ensemble[k][i1:i2])[1]

            scores[k] = sim.score(sim.ensemble[k][i1:i2])[1]
            u2[k] = sim.ensemble[k][i2]+ randn(sim.d)*sim.ϵ
            if k+N_c <= N # start over at the beginning if sum(copies) <N
                sim.ensemble[k+N_c][1:i2] .= copied_sample[k]
                scores[k+N_c] = sim.score(sim.ensemble[k][i1:i2])[1]
                u2[k+N_c] = sim.ensemble[k][i2]+ randn(sim.d)*sim.ϵ
            end
        end
    end
    sim.R[i+1] = mean(scores)
end

# This has to be modified to plot a particular index of the state vector
function ensemble_statistics(ensemble, state_index)
    M = map(x -> x[state_index], reduce(hcat,ensemble))
    means = mean(M, dims = 2)
    stds = std(M, dims = 2)
    return means[:], stds[:]
end

#include("./old_code.jl")

end
