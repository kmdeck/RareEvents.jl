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

    args = (dt, τ, tspan, evolve_single_trajectory, nensemble, ensemble, metric,k, weight_norm, ϵ, distributed, seed)
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
    scores = zeros(sim.nensemble)
    ncopies = ones(Int, sim.nensemble)
    
    t_start = sim.tspan[1]
    t_integration = sim.tspan[2] - t_start
    nresample = Int(div(t_integration, sim.τ, RoundUp))
    nsteps_per_resample = Int(div(sim.τ, sim.dt, RoundUp))
    rng = MersenneTwister(sim.seed)
    
    map(0:(nresample-1)) do i
        t2 = t_start+(i+1)*sim.τ
        t1 = t_start+ i*sim.τ
        i1 = 1+(i)*nsteps_per_resample 
        i2 = 1+(i+1)*nsteps_per_resample

        trajectories = integrate(sim, u1,t1,t2)
        score_trajectories!(sim, scores, trajectories, i1, i2, i)
        compute_ncopies!(ncopies, scores, sim.nensemble, rng)
        sample_and_rewrite_history!(sim.ensemble, ncopies, i2, rng)
        perturb_trajectories!(u1, sim.ensemble, sim.nensemble, i2, sim.ϵ)# may want rng here?

        i = i+1
    end
end


function integrate(sim::RareEventSampler, u_prev,t_prev,t_curr)
     return pmap(sim.evolve_single_trajectory((t_prev,t_curr)),u_prev;distributed = sim.distributed)
end

function score_trajectories!(sim::RareEventSampler, scores, trajectories, idx_prev::Int,idx_current::Int, idx_resample::Int)
    for k in 1:sim.nensemble
        sim.ensemble[k][idx_prev:idx_current] .= trajectories[k]
        scores[k] = score(sim,trajectories[k])
    end
    sim.weight_norm[idx_resample+1] = mean(scores)
    nothing
end

function compute_ncopies!(ncopies::Array,scores::Array, nensemble::Int, rng)
    weights = scores ./ mean(scores)
    ncopies .= Int.(floor.(weights .+ rand(rng, nensemble)))
    nothing
end

function sample_ids(ensemble::Vector, frequencies::Array, rng)
    nensemble = length(frequencies)
    ids = Array(1:1:nensemble)
    ncopies = sum(frequencies)
    # First make a list with id[k] repeated frequencies[k] times
    copied_id_set = shuffle!(rng, vcat([repeat([k], frequencies[k]) for k in 1:nensemble]...))
    if ncopies < nensemble
        ids_chosen = vcat(copied_id_set, copied_id_set[1:nensemble-ncopies]) # Not quite the same as sampling with replacement
    else
        ids_chosen = copied_id_set[1:nensemble] # should be the same as sampling w/o replacement
    end
    
    # be aware!! this is very sensitive. 
    # For example, this does not return the same distribution
#    if sum(frequencies) < nensemble
#        ids_chosen = sample(copied_id_set, nensemble)
#    else
#        ids_chosen = sample(copied_id_set, nensemble, replace = false)
#    end

    
    # Now we have the set of ids we want to carry on with. Within this set,
    # we may not have nsenemble unique members; some ids appear more than once.

    # Determine which ids of the original set are cut, and which require ADDITIONAL copies (which must be cloned)
    chosen_ncopies = counts(ids_chosen, nensemble)
    ids_to_be_cut =  ids[chosen_ncopies .== 0]
    ids_to_be_cloned =  ids[chosen_ncopies .> 1]

    # Determine how many clones to make of those that need them
    nclones = chosen_ncopies[ids_to_be_cloned] .- 1
    
    # Create a list where the cloned ids appear as many times as they are cloned.
    # By construction, this has the same length as ids_to_be_cut.
    cloned_id_set = vcat([repeat([ids_to_be_cloned[k]], nclones[k]) for k in 1:length(ids_to_be_cloned)]...)
    return ids_to_be_cut, cloned_id_set
end


function rewrite_history!(ensemble::Vector, ids_to_be_cut::Array, cloned_id_set::Array, idx_current::Int)
    for j in 1:length(ids_to_be_cut)
        idx_cut = ids_to_be_cut[j]
        idx_clone = cloned_id_set[j]
        ensemble[idx_cut][1:idx_current] .= ensemble[idx_clone][1:idx_current]
    end
    nothing
end


function sample_and_rewrite_history!(ensemble::Vector, frequencies::Array, idx_current::Int, rng)
    ids_to_be_cut, cloned_id_set = sample_ids(ensemble, frequencies, rng)
    rewrite_history!(ensemble, ids_to_be_cut, cloned_id_set, idx_current)
    nothing
end


function perturb_trajectories!(u_current::Array, ensemble::Vector, nensemble::Int, idx_current::Int, ϵ::Real)
    d = length(u_current[1])
    for j in 1:nensemble
        u_current[j] = ensemble[j][idx_current] + randn(d)*ϵ
    end
    nothing
end

include("utils.jl")
include("gev_utils.jl")

# I'm keeping this around for now; we use it in tests.
function orig_sample_and_rewrite_history!(ensemble::Vector, frequencies::Array, idx_current::Int,rng)
    N = length(frequencies)
    N_c = sum(frequencies)

    historical_trajectories = [[ensemble[k][1:idx_current],] for k in 1:N]

    copies = shuffle!(rng, vcat(repeat.(historical_trajectories, frequencies)...))
    
    if N_c > N
        for k in 1:N
            ensemble[k][1:idx_current] .= copies[k]
        end
    else
        for k in 1:N_c
            ensemble[k][1:idx_current] .= copies[k]
            if k+N_c <= N # start over at the beginning if sum(copies) <N
                ensemble[k+N_c][1:idx_current] .= copies[k]
            end
        end
    end
end

end
