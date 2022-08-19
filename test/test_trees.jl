using Test
using StatsBase
using Random
import RareEvents: compute_ncopies!, sample_ids!, IntNode, append_child!, children, parent, nodevalue, leaf

nensemble = 5
nresample = 5
# not appending correctly. 
@testset "foo" begin
    trees = [IntNode{Int64}(k) for k in 1:nensemble]
    rng = MersenneTwister(1234);
    frequencies = Array(1:nensemble) # P(k) = k/norm
    ids = Array(1:1:nensemble)
    history = Int.(zeros((nensemble, nresample)))
    history .= repeat(ids, 1,nresample)

    for i in 1:nresample
        cloned_id_set = similar(ids)
        sample_ids!(cloned_id_set, ids, frequencies, rng)

        for k in 1:nensemble
            if cloned_id_set[k] == k
                # We keep at least one copy
                append_child!(trees[k], IntNode{Int64}(k))
            else
                # for those that are replaced...
                idx_clone = cloned_id_set[k]
                append_child!(trees[idx_clone], IntNode{Int64}(k))
                history[k, 1:i] .= history[idx_clone,1:i]
            end
        end
    end
end


