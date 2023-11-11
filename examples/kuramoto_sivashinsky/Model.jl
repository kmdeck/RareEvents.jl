# Code for setting up the model equations to solve
# ∂u/∂t = -∂²u/∂x² - ∂⁴u/∂x⁴ - 1/2 ∂u²/∂x
# or, ∂u/∂t = Lu + N(u),
# where L = -∂²/∂x² - ∂⁴/∂x⁴ and N(u) = - 1/2 ∂u²/∂x,
# in Fourier space using FourierFlows.jl

# This copies the examples in FourierFlows.

# Let u(x,t) = ∫ũ(k,t) exp{i*k*x} dk
# Then ∂ⁿu∂xⁿ = (ik)ⁿ u(x,t)
# Subsitute this in for the linear operator,
# and multiply each side of the equation by  exp{-i*k*x}
# and integrate over x.

# We have:
# - ∫ ∂u/∂t exp{-i*k*x} dx = ∂∫ u exp{-i*k*x} dx/∂t = ∂ũ/∂t
# - ∫Lu exp{-i*k*x} dx = Lũ = (k²-k⁴)ũ
# - ∫ N(u) exp{-i*k*x} dx = -1/2 i k ∫u² exp{-i*k*x} dx
# or
# ∂ũ/∂t = (k²-k⁴)ũ - 1/2 ik[u^2]̃
module Model
using FourierFlows
using LinearAlgebra: ldiv!, mul!

export Vars, Equation, updatevars!, Params, set_u!
# Allocate space in Vars for solution and intermediate
# quantities.
# k denotes a fourier transformed variable
struct Vars{Aphys, Atrans} <: AbstractVars
    "The solution u"
    u :: Aphys
    "The Fourier transform of u"
    uk :: Atrans
    "The Fourier transform of u²"
    uuk :: Atrans
end

function Vars(grid)
  Dev = typeof(grid.device)
  T = eltype(grid)
  @devzeros Dev T grid.nx u
  @devzeros Dev Complex{T} grid.nkr uk uuk
  return Vars(u, uk, uuk,)
end

"""
    Equation(grid)

Define the linear and nonlinear components of the PDE,
in a format defined by FourierFlows.
"""
function Equation(grid)
    T = eltype(grid)
    dev = grid.device
    # Define nonlinear function using signature required by FourierFlows
    function calcN!(N, sol, t, clock, vars, params, grid)
        @. N[:, 1] = - im * grid.kr *vars.uuk/2
        dealias!(N, grid)
        return nothing
    end
    # Define linear function
    L = zeros(dev, T, (grid.nkr, 1))
    @. L[:, 1] = grid.kr^2 - grid.kr^4
    
    return FourierFlows.Equation(L, calcN!, grid)
end

"""
    updatevars!(prob)

Update the intermediate variables required to compute the tendency,
stored in prob.vars, using the solution prob.sol.
"""
function updatevars!(prob)
    vars, grid, sol = prob.vars, prob.grid, prob.sol
    @. vars.uk = sol[:, 1]
    # use deepcopy() because irfft destroys its input
    ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uk))
    mul!(vars.uuk, grid.rfftplan, deepcopy(vars.u .* vars.u))
  return nothing
end

"""
    Params <: AbstractParams

This model has no parameters but we are required to pass in
an argument of type `AbstractParams`.
"""
struct Params <: AbstractParams end

"""
    set_u!(prob, u0)

This function takes in an initial condition and updates the 
field prob.sol and prob.vars.
"""
function set_u!(prob, u0)
  vars, grid, sol = prob.vars, prob.grid, prob.sol
  
  A = typeof(vars.u) # determine the type of vars.u
  
  ## below, e.g., A(u0) converts u0 to the same type as vars expects
  ## (useful when u0 is a CPU array but grid.device is GPU)
  mul!(vars.uk, grid.rfftplan, A(u0))
  @. sol[:, 1] = vars.uk
  updatevars!(prob)
  return nothing
end

end
