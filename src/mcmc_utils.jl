
function accept_or_reject(x::Vector, x_proposed::Vector, data::Vector, log_likelihood::Function)
    u = rand()
    if log(u) <= log_likelihood(data, x_proposed) - log_likelihood(data, x)
        return x_proposed
    else
        return x
    end
end

function mcmc_sample(data, x::Vector, k::Int, σ_k::Float64, log_likelihood::Function)
    x_proposed = copy(x)
    x_proposed[k] += randn()*σ_k
    x_next = accept_or_reject(x, x_proposed, data, log_likelihood)
    return vcat(x_next, log_likelihood(data, x_next))
end

function evolve_mcmc(data::Vector, x0::Vector, σ::Vector, log_likelihood::Function, nsteps::Int; save_at::Int)
    nparams = length(x0)
    chain = [zeros(nparams+1) for _ in 1:(div(nsteps,save_at)+1)]
    chain[1] .= vcat(x0, log_likelihood(data, x0))
    for i in 2:nsteps
        k = Int64(ceil(rand()*3))
        if i % save_at == 0
            curr = div(i, save_at)+1
            prev = curr -1
            chain[curr] .= mcmc_sample(data, chain[prev][1:nparams],k, σ[k], log_likelihood)
        end
    end
    return chain
end

export evolve_mcmc
