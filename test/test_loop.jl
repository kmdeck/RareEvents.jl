using Test
using RareEvents: sample_and_rewrite_history!
using Random

function original_loop!(ensemble, nresample, nsteps_per_resample, nensemble,rng)
    map(0:(nresample-1)) do i
        i1 = 1+(i)*nsteps_per_resample 
        i2 = 1+(i+1)*nsteps_per_resample
        
        trajectories = [randn(rng, i2-i1+1) for _ in 1:nensemble]
        for k in 1:nensemble
            ensemble[k][i1:i2] .= trajectories[k]
        end
        
        ncopies = Int64.(round.(rand(rng,nensemble))) .+1
        sample_and_rewrite_history!(ensemble, ncopies, i2, rng)
        i = i+1
    end
    nothing
end

function new_loop!(ensemble, nresample, nsteps_per_resample, nensemble,rng)
    i1 = 1
    i2 = 1+1*nsteps_per_resample
    ncopies = Int64.(zeros(nensemble))
    map(0:(nresample-1)) do i
        # can be done per worker
        if i > 0
            sample_and_rewrite_history!(ensemble, ncopies, i1, rng)
        end
        trajectories = [randn(rng, i2-i1+1) for _ in 1:nensemble]
        for k in 1:nensemble
            ensemble[k][i1:i2] .= trajectories[k]
        end
        
        # Central
        ncopies .= Int64.(round.(rand(rng,nensemble))) .+1
        # Increment resample index
        i = i+1
        # Update start and end points of integrations
        i1 = i2
        i2 = 1+(i+1)*nsteps_per_resample
    end
    nothing
end

@testset "Comparing loops" begin
    nsteps_per_resample = 1
    nresample = 100
    nensemble = 10
    u1 = zeros(nensemble)

    rng = MersenneTwister(5)
    ensemble = [similar(copy(zeros(nresample+1))) for _ in 1:nensemble]

    original_loop!(ensemble, nresample, nsteps_per_resample, nensemble,rng)

    rng = MersenneTwister(5)
    ensemble2 = [similar(copy(zeros(nresample+1))) for _ in 1:nensemble]
    
    new_loop!(ensemble2, nresample, nsteps_per_resample, nensemble,rng)
    @test ensemble == ensemble2
end
