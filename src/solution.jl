struct Dimensions
    num_scalars::Int64
    side_dimensions::Vector{Int64}
    offsets::Vector{Int64}
end

num_matrices(d::Dimensions) = length(d.side_dimensions)
Base.length(d::Dimensions) = d.offsets[end]

struct ScalarIndex
    value::Int64
end

struct MatrixIndex
    value::Int64
end
Base.broadcastable(i::MatrixIndex) = Ref(i)

abstract type AbstractSolution{T} <: AbstractVector{T} end

function Base.getindex(
    s::AbstractSolution,
    i::Union{Type{ScalarIndex},MatrixIndex},
)
    return view(s, i)
end

struct VectorizedSolution{T} <: AbstractSolution{T}
    x::Vector{T}
    dim::Dimensions
end

function LinearAlgebra.dot(x::VectorizedSolution, z::VectorizedSolution)
    return LinearAlgebra.dot(x.x, z.x)
end

num_matrices(x::VectorizedSolution) = num_matrices(x.dim)

Base.similar(s::VectorizedSolution) = VectorizedSolution(similar(s.x), s.dim)

Base.size(s::VectorizedSolution) = (length(s.dim),)

function Base.to_index(s::VectorizedSolution, ::Type{ScalarIndex})
    return Base.OneTo(s.dim.num_scalars)
end

function Base.to_index(s::VectorizedSolution, mi::MatrixIndex)
    i = mi.value
    return (1+s.dim.offsets[i]):s.dim.offsets[i+1]
end

function Base.view(s::VectorizedSolution, ::Type{ScalarIndex})
    return view(s.x, Base.to_index(s, ScalarIndex))
end

# `s[i] .= ...` calleds `copyto!(view(s, i), Broadcasted(...))`
function Base.view(s::VectorizedSolution, i::MatrixIndex)
    v = view(s.x, Base.to_index(s, i))
    dim = s.dim.side_dimensions[i.value]
    X = reshape(v, dim, dim)
    return X
end

Base.setindex!(s::VectorizedSolution, v, i::Integer) = setindex!(s.x, v, i)
Base.getindex(s::VectorizedSolution, i::Integer) = getindex(s.x, i)

struct ShapedSolution{T,MT<:AbstractMatrix{T}} <: AbstractSolution{T}
    scalars::Vector{T}
    matrices::Vector{MT}
end

function Base.size(s::ShapedSolution)
    return (length(s.scalars) + sum(length, s.matrices, init = 0),)
end
num_matrices(s::ShapedSolution) = length(s.matrices)

function LinearAlgebra.norm2(s::ShapedSolution{T}) where {T}
    # `LinearAlgebra.generic_norm2` starts by computing the ∞ norm and do a rescaling, we don't do that here
    return √(LinearAlgebra.dot(s, s))
end

function LinearAlgebra.dot(a::ShapedSolution{T}, b::ShapedSolution{T}) where {T}
    return LinearAlgebra.dot(a.scalars, b.scalars) +
           sum(eachindex(a.matrices); init = zero(T)) do i
        return LinearAlgebra.dot(a.matrices[i], b.matrices[i])
    end
end

Base.view(s::ShapedSolution, ::Type{ScalarIndex}) = s.scalars
Base.view(s::ShapedSolution, i::MatrixIndex) = s.matrices[i.value]
