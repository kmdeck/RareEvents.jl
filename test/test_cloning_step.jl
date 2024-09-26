using Test
using StatsBase
using Random
import RareEvents: compute_ncopies!, sample_ids!

nensemble = 500

@testset "sample_ids! unit test" begin
    rng = MersenneTwister(1234);
    frequencies = Array(1:nensemble) # P(k) = k/norm
    ids = Array(1:1:nensemble)
    cloned_ids = similar(ids)
    sample_ids!(cloned_ids, ids, frequencies, rng)
    rng = MersenneTwister(1234);
    ids_chosen = shuffle!(rng, vcat([repeat([k], frequencies[k]) for k in 1:nensemble]...))[1:nensemble]
    n_appearances = StatsBase.counts(ids_chosen, nensemble)
    ids_to_be_cut =  ids[n_appearances .== 0]
    ids_to_be_cloned =  ids[n_appearances .> 1]
    ids_left_alone = ids[n_appearances .== 1]
    # Determine how many clones to make of those that need them
    n_clones = n_appearances[ids_to_be_cloned] .- 1
    @test cloned_ids[ids_left_alone] == ids[ids_left_alone]
    @test sum(cloned_ids[ids_to_be_cut] .== ids[ids_to_be_cut]) == 0
    final_counts = StatsBase.counts(cloned_ids, nensemble)
    @test ids_to_be_cloned == ids[final_counts .> 1]
    @test final_counts[final_counts .> 1] .- 1 == n_clones
end

@testset "sample_ids!: ncopies = 1 limiting case" begin
    rng = MersenneTwister(1234);
    ncopies = Int64.(ones(nensemble))
    ids = Array(1:1:nensemble)
    cloned_ids = similar(ids)
    sample_ids!(cloned_ids, ids, ncopies, rng)
    @test cloned_ids == ids
end


@testset "ncopies unit test" begin
    rng= MersenneTwister(1);
    nensemble = 10
    draw = rand(rng, nensemble)
    ncopies = Int.(ones(nensemble))
    scores = Array(1.0:1:nensemble)
    expected_ncopies = Int.(floor.(scores ./mean(scores) .+ draw))
    rng= MersenneTwister(1);
    compute_ncopies!(ncopies, scores, mean(scores), nensemble, rng)
    similar = ncopies .== expected_ncopies
    @test sum(similar) == nensemble
end
