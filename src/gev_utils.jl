using Optim

export log_likelihood_gev, fit_gev, gev_pdf, gev_cdf


"""
    gev_pdf(x::FT, σ::FT, μ::FT, ξ::FT) where {FT}

Computes the probability density function for the gev 
distribution given a point x, and parameters σ, μ, ξ.
"""
function gev_pdf(x::FT, σ::FT, μ::FT, ξ::FT) where {FT}
    y = (x-μ)/σ
    z = FT(1.0)+ξ*y
    if (abs(ξ) > FT(1e-6)) && (abs(z) > FT(1e-6))
        return exp(-z^(-FT(1)/ξ))*z^(-(FT(1)+FT(1)/ξ))./σ
    else
        return exp(-y)*exp(-exp(-y))./σ
    end
    
end 

"""
    log_likelihood_gev(data::Vector, parameters::Vector{FT}) where {FT}

Computes the log likelihood of a data set `data` under the GEV
distribution, assuming parameters = [σ, μ, ξ].
"""
function log_likelihood_gev(data::Vector, parameters::Vector{FT}) where {FT}
    σ, μ,ξ = parameters
    y = (data .-μ)./σ 
    z = FT(1.0) .+ ξ .* y

    m = length(data)
    first_constraint = sum(z .> 0) == m
    second_constraint = σ > 0
    if !first_constraint || !second_constraint
        return -FT(1.0)/eps(FT)
    else
        # Per Cole's textbook
        if abs(ξ) > FT(1e-6)
            return -m*log(σ) - (FT(1) + FT(1)/ξ)*sum(log.(z)) - sum(z.^(-FT(1)/ξ))
        else
            return -m*log(σ) - sum(y) - sum(exp.(-FT(1) .*y))
        end
    end
    nothing
end

"""
    fit_gev(blocks::Vector, initial_guess::Vector{FT}) where{FT}

Computes the best fit parameters for the gev distribution, given 
a data vector `blocks` and an initial guess vector [σ, μ, ξ].
"""
function fit_gev(blocks::Vector, initial_guess::Vector{FT}) where{FT}
    function wrapper(guess::Vector{FT}) where{FT}
        return -(log_likelihood_gev(blocks,guess))
    end
    results = optimize(wrapper, initial_guess, NelderMead())
    return Optim.minimizer(results), Optim.minimum(results)
end

"""
    gev_cdf(x::FT, σ::FT, μ::FT, ξ::FT) where {FT}

Computes the cumulative probability density function for the gev 
distribution given a point x, and parameters σ, μ, ξ.
"""
function gev_cdf(x::FT, σ::FT, μ::FT, ξ::FT) where {FT}
    y = (x-μ)/σ
    z = FT(1.0)+ξ*y
    if ξ < 0.0 && z <= 0
        return FT(1)
    elseif ξ > 0.0 && z <= 0
        return FT(0)
    elseif abs(ξ) > FT(1e-6)
        return exp(-z^(-FT(1)/ξ))
    else
        return exp(-exp(-y))
    end
    
end
