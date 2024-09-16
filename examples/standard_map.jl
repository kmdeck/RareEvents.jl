struct StandardMap{FT<: AbstractFloat}
    K::FT
end

function mapping_system(model::StandardMap)
    function update!(u)
        u[4] = u[4] + model.K*cos(u[1])*u[3] # δI
        u[3] = u[3] + u[4] # δθ
        u[2] = mod(u[2] + model.K*sin(u[1]),2π) # I
        u[1] = mod(u[1] + u[2],2π) # θ
    end
    return update!
end

function evolve_mapping_system(model::StandardMap, u0, tspan)
    update! = mapping_system(model)
    nsteps = Int(tspan[2]-tspan[1])
    solution = map(0:nsteps) do (i)
        update!(u0)
        copy(u0)
    end
    return solution
end
