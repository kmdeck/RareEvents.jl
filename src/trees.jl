
mutable struct IntNode{T<:Int}
    value::T
    children::Vector{IntNode{T}}
    parent::IntNode{T}
    IntNode{T}(value::T) where {T} = new{T}(value)
end

children(n::IntNode) = try n.children catch nothing end
nodevalue(n::IntNode) = n.value;
parent(n::IntNode) = try n.parent catch nothing end

function append_child!(n::IntNode, child::IntNode)
    if children(n) isa Nothing
        n.children = [child]
    else
        n.children = vcat(n.children, child)
    end
end
