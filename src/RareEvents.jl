module RareEvents
using Statistics
using StatsBase
using Distributed
using Random


export RareEventSampler, run!
#include("trees.jl")
    
"""
    RareEventSampler{FT<:AbstractFloat, ST,P, ES}
The struct holding the necessary information for
running the rare event algorithm.
"""  

struct RareEventSampler{FT<:AbstractFloat, ST,P}
    "The timestep, and how often output is saved from the integrations"
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
    "Random seed"
    seed::Int64
    "History of which trajectories were copied"
    history::Matrix{Int}
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
    ϵ::FT,
    seed::Int64;
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

    # Preallocation for history matrix
    history = Int.(zeros((nensemble, Int(div(t_integration, τ, RoundUp)))))
    args = (dt, τ, tspan, evolve_single_trajectory, nensemble, ensemble, metric,k, weight_norm, ϵ, distributed, seed, history)
    #trees = [IntNode{Int64}(k) for k in 1:nensemble]
    return RareEventSampler{FT,typeof(ensemble),typeof(evolve_single_trajectory)}(args...)
end

"""
    run!(sim::RareEventSampler)
Runs the rare event algorithm according to the properties of `sim`.
"""
function run!(sim::RareEventSampler)
    # Preallocation
    scores = zeros(sim.nensemble)
    ncopies = ones(Int, sim.nensemble)
    ids = Array(1:1:sim.nensemble)

    t_start = sim.tspan[1]
    t_integration = sim.tspan[2] - t_start
    nresample = Int(div(t_integration, sim.τ, RoundUp))
    nsteps_per_resample = Int(div(sim.τ, sim.dt, RoundUp))

    # Initialize
    rng = MersenneTwister(sim.seed)

    t1 = t_start
    t2 = t_start+sim.τ
    i1 = 1
    i2 = 1+1*nsteps_per_resample
    cloned_id_set = Array(1:1:sim.nensemble)
    sim.history .= repeat(ids, 1,nresample)
    # dimensionality of state
    d = length(sim.ensemble[1][1])
    map(0:(nresample-1)) do i

        for k in 1:sim.nensemble
            # "rewrite history"
            # Note that if i = 0, cloned_id_set is such that no trajectory is rewritten.
            if cloned_id_set[k] == k
                nothing
                # We keep at least one copy
                #append_child!(trees[k], IntNode{Int64}(k))
            else
                # for those that are replaced...
                idx_clone = cloned_id_set[k]
                #append_child!(trees[idx_clone], IntNode{Int64}(k))
                sim.ensemble[k][1:i1] .= sim.ensemble[idx_clone][1:i1]
                sim.history[k, 1:i] .= sim.history[idx_clone,1:i]
            end
            # Perturb
            # Note: We don't need to do this at i = 0, but it doesn't matter.
            perturbed_ic = sim.ensemble[k][i1] + randn(rng, d)*sim.ϵ
            # Integrate
            trajectory = sim.evolve_single_trajectory((t1,t2))(perturbed_ic)
            # Store
            sim.ensemble[k][i1:i2] .= trajectory
            # Score
            scores[k] = score(trajectory, sim.k, sim.dt, sim.metric)
        end    

        # Central
        sim.weight_norm[i+1] = mean(scores)
        compute_ncopies!(ncopies, scores, sim.weight_norm[i+1], sim.nensemble, rng)
        sample_ids!(cloned_id_set, ids, ncopies, rng)
        
        # Increment resample index
        i = i+1
        # Update start and end points of integrations
        t1 = t2
        t2 = t_start+(i+1)*sim.τ
        i1 = i2
        i2 = 1+(i+1)*nsteps_per_resample
    end
end

# Do it the easy way first so we have a comparison
function reconstruct_trajectory(sim::RareEventSampler, k::Int, nsteps_per_resample::Int)
    # allocation
    reconstructed_trajectory = similar(sim.ensemble[k])
    # history 
    history = sim.history[k,:]

    # reconstruct
    map(0:(nresample-1)) do i
        i1 = 1
        i2 = 1+(i+1)*nsteps_per_resample  
        reconstructed_trajectory[i1:i2] .= sim.ensemble[history[i+1]][i1:i2]
    end
    return reconstructed_trajectory
end

    

"""
    score(trajectory, k, dt, metric)
Scores the trajectory based on the `metric` function
and the parameter `k` chosen by the user, as
``
e^[∫k metric(trajectory) dt]
``
"""
function score(trajectory, k, dt, metric)
    integrand = map(metric, trajectory)
    return exp(k*sum(integrand[2:end]+integrand[1:end-1])/2.0*dt)
end

"""
    compute_ncopies!(ncopies::Vector{Int},scores::Vector{Real}, norm::Real, nensemble::Int, rng)

    Computes an integer-valued array with elements equal to the number of copies of the ensemble
members needed to perform the resampling step; updates `ncopies` in place.
"""
function compute_ncopies!(ncopies::Vector{Int},scores::Vector{FT}, norm::FT, nensemble::Int, rng) where {FT <: Real}
    weights = scores ./ norm
    ncopies .= Int.(floor.(weights .+ rand(rng, nensemble)))
    nothing
end

"""
    sample_ids!(mapped_id_set::Vector{Int}, ids::Vector{Int}, frequencies::Vector{Real}, rng)

Update `mapped_id_set` in place, by sampling from `ids` with frequency `frequencies`.

Most of the effort is in ensuring that the returned `mapped_id_set` equals `ids` in the
locations where a member is sampled once. For example, if `ids` = [1, 2, 3, 4, 5]` and
frequencies = [2, 2, 0, 0, 1], we want mapped_id_set = [1, 2, 1, 2, 5]  or [1, 2, 2, 1, 5].
That is, the order of the returned mapped_id_set matters, because we only want to rewrite
 elements of the ensemble where mapped_id_set .!= ids. 

If we decide to not rewrite trajectories as the algorithm proceeds, and just do it at the end,
this is not required, I think.
"""
function sample_ids!(mapped_id_set::Vector{Int}, ids::Vector{Int}, frequencies::Vector{Int}, rng)
    nensemble = length(frequencies)
    ncopies = sum(frequencies)
    # First make a list with id[k] repeated frequencies[k] times
    copied_id_set = shuffle!(rng, vcat([repeat([k], frequencies[k]) for k in 1:nensemble]...))
    if ncopies < nensemble
        ids_chosen = vcat(copied_id_set, copied_id_set[1:nensemble-ncopies]) # Not quite the same as sampling with replacement
    else
        ids_chosen = copied_id_set[1:nensemble] # the same as sampling w/o replacement
    end
    
    # Now we have the set of ids we want to carry on with. Within this set,
    # we may not have nsenemble unique members; some ids appear more than once.

    # Determine which ids of the original set are cut, and which require ADDITIONAL copies (which must be cloned)
    n_appearances = StatsBase.counts(ids_chosen, nensemble)
    ids_to_be_cut =  ids[n_appearances .== 0]
    ids_to_be_cloned =  ids[n_appearances .> 1]

    # Determine how many clones to make of those that need them
    n_clones = n_appearances[ids_to_be_cloned] .- 1
    
    # Create a list where the cloned ids appear as many times as they are cloned.
    # By construction, this has the same length as ids_to_be_cut.
    cloned_id_set = vcat([repeat([ids_to_be_cloned[k]], n_clones[k]) for k in 1:length(ids_to_be_cloned)]...)
    mapped_id_set .= ids
    mapped_id_set[ids_to_be_cut] .= cloned_id_set
    return mapped_id_set
end

include("utils.jl")
include("gev_utils.jl")
end
