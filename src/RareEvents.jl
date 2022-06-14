module RareEvents
using Statistics
using StatsBase
using Distributed
using Random

export RareEventSampler, run!

    
"""
    RareEventSampler{FT<:AbstractFloat, ST,P, ES}
The struct holding the necessary information for
running the rare event algorithm.
"""  

struct RareEventSampler{FT<:AbstractFloat, ST,P}
    "How often output is saved from the integrations"
    dt::FT
    "Resampling time"
    τ::FT
    "The integration time span for each trajectory"
    tspan::Tuple{FT,FT}
    "The dynamical system evolve function, for a single trajectory"
    evolve_single_trajectory::P
    "The number of trajectories in the ensemble"
    nensemble::Int
    "The ensemble, which stores all output from all trajectories"
    ensemble::ST
    "The metric function,  which maps a trajectory to a scalar measure of extremity"
    metric::Function
    "The coefficient used in the weight function, e^[k∫metric(X) dt]"
    k::FT
    "Vector storing the normalization of the weights at each resample"
    weight_norm::Vector{FT}
    "The standard deviation of the white noise added to each trajectory upon resampling"
    ϵ::FT
    "Flag indicating if the user wishes to run in a distributed fashion"
    distributed::Bool
end

"""
	RareEventSampler{FT}(
	    dt::FT,
	    τ::FT,
	    tspan::Tuple{FT,FT},
	    u0::U,
	    evolve_single_trajectory::P,
	    nensemble::Int,
	    metric::Function,
	    k::FT,
	    ϵ::FT;
            distributed = false
	) where {FT,U,P}
Constructor for the RareEventSampler object, which is used to implement
the GKLT algorithm (Giardina, Kurcha, Lecomte, and Tailleu, 2011).
The user should pass
- a timestep `dt`, which is also the frequency of output,
- the resample time `τ`, which should be an integer number of timesteps,
-  the time span tuple `tspan = (t0,tf)`, which 
should be satisfy `(tf-t0)` being an integer number of timesteps and an integer
number of the resampling time `τ`,
- the initial condition vector `u0`, which is of length `nensemble` and has elements of the same
type as the state vector,
- the `evolve` function for a single trajectory
- the number of ensemble members `N`
- the `metric` function, which takes a trajectory as input and maps to a scalar,
- the parameter `k` which controls the typical magnitude of the metric for trajectories
returned by the algorithm,
- `ϵ` the amplitude of the white noise to add to each element of the state vector
- `distributed` either true or false, indicating if you wish to run in a distributed manner or not.
The `evolve` function takes in initial conditions and a tspan, as well as algorithm
keyword arguments, and returns the solution at intervals of `dt`.Note that to run 
with pmap, we actually require the following: 
`evolve_wrapper(tspan) = (u0) ->  evolve_single_trajectory(u0, tspan; alg_kwargs...)`
with `evolve_wrapper` passed in to the `RareEventSampler`. 
"""
function RareEventSampler{FT}(
    dt::FT,
    τ::FT,
    tspan::Tuple{FT,FT},
    u0::U,
    evolve_single_trajectory::P,
    nensemble::Int,
    metric::Function,
    k::FT,
    ϵ::FT;
    distributed = false
) where {FT,U,P}
    t_integration = tspan[2] - tspan[1] # total runtime for algorithm

    # Check that tspan is integer multiple for preallocation purposes
    @assert rem(t_integration, dt,RoundUp) == 0
    @assert rem(t_integration, τ, RoundUp) == 0

    # Ensemble initilization with different initial conditions
    nsteps = div(t_integration, dt, RoundUp) + 1
    ensemble = [[step == 1 ? u0[member] : similar(u0[member]) for step in 1:nsteps] for member in 1:nensemble]

    # Preallocation for weight normalization at each resample time
    weight_norm = zeros(Int(div(t_integration, τ, RoundUp)))

    args = (dt, τ, tspan, evolve_single_trajectory, nensemble, ensemble, metric,k, weight_norm, ϵ, distributed)
    return RareEventSampler{FT,typeof(ensemble),typeof(evolve_single_trajectory)}(args...)
end

"""
    score(res::RareEventSampler, trajectory)
Scores the trajectory based on the `metric` function
and the parameter `k` chosen by the user, as
``
e^[∫k metric(trajectory) dt]
``
"""
function score(res::RareEventSampler, trajectory)
    k = res.k
    dt = res.dt
    integrand = map(res.metric, trajectory)
    return exp(k*sum(integrand[2:end]+integrand[1:end-1])/2.0*dt)
end

"""
    run!(sim::RareEventSampler)
Runs the rare event algorithm according to the properties of `sim`.
"""
function run!(sim::RareEventSampler)
    # Preallocation
    u1 = deepcopy([sim.ensemble[k][1] for k in 1:sim.nensemble])
    u2 = deepcopy([sim.ensemble[k][1] for k in 1:sim.nensemble])
    scores = zeros(sim.nensemble)
    ncopies = zeros(Int, sim.nensemble)
    
    t0 = sim.tspan[1]
    t_integration = sim.tspan[2] - t0
    nresample = Int(div(t_integration, sim.τ, RoundUp))
    nsteps_per_resample = Int(div(sim.τ, sim.dt, RoundUp))
    
    map(0:(nresample-1)) do i
        t2 = t0+(i+1)*sim.τ
        t1 =  t0+ i*sim.τ
        i1 = 1+(i)*nsteps_per_resample 
        i2 = 1+(i+1)*nsteps_per_resample 
        trajectories = integrate(sim, u1,t1,t2)
        score_trajectories!(sim, scores, trajectories, i1, i2)
        compute_ncopies!(sim, ncopies, scores, i)
        sample_and_rewrite_history!(sim, u2, ncopies, i1, i2, i)
        i = i+1
        u1 = copy(u2)
    end
end


function integrate(sim::RareEventSampler, u1,t1,t2)
     return pmap(sim.evolve_single_trajectory((t1,t2)),u1;distributed = sim.distributed)
end

function score_trajectories!(sim::RareEventSampler, scores, trajectories,i1,i2)
    # We need to loop over the original ensemble to get the scores
    # so that we can get the mean score, before resampling.
    for k in 1:sim.nensemble
        sim.ensemble[k][i1:i2] .= trajectories[k]
        scores[k] = score(sim,trajectories[k])
    end
end

function compute_ncopies!(sim, ncopies,scores, i)
    sim.weight_norm[i+1] = mean(scores)
    weights = scores ./ mean(scores)
    ncopies .= Int.(floor.(weights .+ rand(sim.nensemble)))
end

function sample_and_rewrite_history!(sim, u2, ncopies, i1, i2, i)
    # Make this more readable. we should just shuffle and draw from an
    # index array.
    
    # We need to loop over the data to sample from it N times
    N = sim.nensemble
    array_u = [[sim.ensemble[k][1:i2],] for k in 1:N]
    # If copies is zero, it is cut. Then shuffle the cloned set.
    # Selecting from it in order = a random sample
    copied_sample = shuffle!(vcat(sample.(array_u, ncopies)...))
    N_c = sum(ncopies)

    # Dimensionality of state vector
    d = length(u2[1])
    # We need to loop over it to reset the ensemble to the samples.
    if N_c > N
        for k in 1:N
            sim.ensemble[k][1:i2] .= copied_sample[k]
            u2[k] = sim.ensemble[k][i2] + randn(d)*sim.ϵ
        end
    else
        for k in 1:N_c
            sim.ensemble[k][1:i2] .= copied_sample[k]
            u2[k] = sim.ensemble[k][i2] + randn(d)*sim.ϵ
            if k+N_c <= N # start over at the beginning if sum(copies) <N
                sim.ensemble[k+N_c][1:i2] .= copied_sample[k]
                u2[k+N_c] = sim.ensemble[k+N_c][i2] + randn(d)*sim.ϵ
            end
        end
    end
end

include("utils.jl")
end
