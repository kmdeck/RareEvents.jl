using FourierFlows
using Plots
using HDF5
include("Model.jl")
using .Model: Vars, Equation, Params, updatevars!, set_u!
# Run the simulation
dev = CPU()
stepper = "FilteredETDRK4"
dt = 0.1
nx, Lx = 128, 100
grid = OneDGrid(; nx, Lx, x0 = 0.0)

vars = Vars(grid)
equation = Equation(grid)

prob = FourierFlows.Problem(equation, stepper, dt, grid, vars, Params())
u0 = @. sin(16π * grid.x ./ Lx)
set_u!(prob, u0)
updatevars!(prob)

# Integration time, timestep, nsteps
# The autocorrelation time τ ~10
# We want 3τ = 128 steps of dt_save to generate 128x128 "images" to train with
# dt_save ∼0.2
# 30k samples -> 3e4*(dt_save*128) = 7.68e5 
#tspan = (0.0,1e3)
tspan = (0.0,7.68e5)
nsteps = Int((tspan[2]-tspan[1])/dt)
# Saving interval, steps per interval, total # of solutions saved
dt_save = 0.2
n_steps_per_save = Int(round(dt_save/dt))
savesteps = 0:n_steps_per_save:nsteps

# Preallocate the solution array
trajectory = zeros((nx, Int(nsteps/n_steps_per_save)));
for j = 1:nsteps
    updatevars!(prob)
    if j ∈ savesteps
        save_index = Int(j/n_steps_per_save)
        trajectory[:,save_index] .= vars.u[:]
     end
    stepforward!(prob)
end
spinup = 200.0
n_savesteps_in_spinup = Int(spinup / dt_save)
trajectory_nospinup = trajectory[:,n_savesteps_in_spinup:end]
#=
Plots.heatmap(trajectory)
Plots.savefig("sample_trajectory.png")
lags = Array(1:1:(size(trajectory_nospinup)[2]-1)) # in units of steps
ac = StatsBase.autocor(transpose(trajectory_nospinup), lags; demean = true)
mean_ac = mean(ac, dims = 2)[:]
Plots.plot(lags *dt_save, mean_ac, label = "", ylabel = "Autocorrelation Coeff", xlabel = "Lag (time)")
Plots.savefig("autocorr.png")
τ = maximum(lags[mean_ac .> 0.1])*dt_save
@show τ
Plots.heatmap(trajectory_nospinup[:,1:128], xlabel = "Time (30units)", ylabel = "Spatial DOF")
Plots.savefig("sample_image.png")
=#
#Save
fname = "./kuramoto_sivashinksy.hdf5"
fid = h5open(fname, "w")
fid[string("nx_$nx","Lx_$Lx","dt_$dt","dt_save_$dt_save")] = trajectory_nospinup
close(fid)
