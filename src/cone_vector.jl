struct ConeDimensions
    num_scalars::Int64
    side_dimensions::Vector{Int64}
    offsets::Vector{Int64}
end

num_matrices(d::ConeDimensions) = length(d.side_dimensions)
Base.length(d::ConeDimensions) = d.offsets[end]

struct ScalarIndex
    value::Int64
end

struct MatrixIndex
    value::Int64
end
Base.broadcastable(i::MatrixIndex) = Ref(i)

abstract type AbstractConeVector{T} <: AbstractVector{T} end

function Base.getindex(
    s::AbstractConeVector,
    i::Union{Type{ScalarIndex},MatrixIndex},
)
    return view(s, i)
end

struct VectorizedConeVector{T} <: AbstractConeVector{T}
    x::Vector{T}
    dim::ConeDimensions
end

function LinearAlgebra.dot(x::VectorizedConeVector, z::VectorizedConeVector)
    return LinearAlgebra.dot(x.x, z.x)
end

num_matrices(x::VectorizedConeVector) = num_matrices(x.dim)

function Base.similar(s::VectorizedConeVector)
    return VectorizedConeVector(similar(s.x), s.dim)
end

Base.size(s::VectorizedConeVector) = (length(s.dim),)

function Base.to_index(s::VectorizedConeVector, ::Type{ScalarIndex})
    return Base.OneTo(s.dim.num_scalars)
end

function Base.to_index(s::VectorizedConeVector, mi::MatrixIndex)
    i = mi.value
    return (1+s.dim.offsets[i]):s.dim.offsets[i+1]
end

function Base.view(s::VectorizedConeVector, ::Type{ScalarIndex})
    return view(s.x, Base.to_index(s, ScalarIndex))
end

# `s[i] .= ...` calleds `copyto!(view(s, i), Broadcasted(...))`
function Base.view(s::VectorizedConeVector, i::MatrixIndex)
    v = view(s.x, Base.to_index(s, i))
    dim = s.dim.side_dimensions[i.value]
    X = reshape(v, dim, dim)
    return X
end

Base.setindex!(s::VectorizedConeVector, v, i::Integer) = setindex!(s.x, v, i)
Base.getindex(s::VectorizedConeVector, i::Integer) = getindex(s.x, i)

struct ShapedConeVector{T,MT<:AbstractMatrix{T}} <: AbstractConeVector{T}
    scalars::Vector{T}
    matrices::Vector{MT}
end

function Base.size(s::ShapedConeVector)
    return (length(s.scalars) + sum(length, s.matrices, init = 0),)
end
num_matrices(s::ShapedConeVector) = length(s.matrices)

if isdefined(LinearAlgebra, :norm_recursive_check)
    LinearAlgebra.norm_recursive_check(::ShapedConeVector) = nothing
end

function LinearAlgebra.norm2(s::ShapedConeVector{T}) where {T}
    # `LinearAlgebra.generic_norm2` starts by computing the ∞ norm and do a rescaling, we don't do that here
    return √(LinearAlgebra.dot(s, s))
end

function LinearAlgebra.dot(
    a::ShapedConeVector{T},
    b::ShapedConeVector{T},
) where {T}
    return LinearAlgebra.dot(a.scalars, b.scalars) +
           sum(eachindex(a.matrices); init = zero(T)) do i
        return LinearAlgebra.dot(a.matrices[i], b.matrices[i])
    end
end

Base.view(s::ShapedConeVector, ::Type{ScalarIndex}) = s.scalars
Base.view(s::ShapedConeVector, i::MatrixIndex) = s.matrices[i.value]

function Base.getindex(s::ShapedConeVector, i::Integer)
    @boundscheck checkbounds(s, i)
    if i <= length(s.scalars)
        return s.scalars[i]
    end
    i -= length(s.scalars)
    for matrix in s.matrices
        if i <= length(matrix)
            return matrix[i]
        end
        i -= length(matrix)
    end
    return throw(BoundsError(s, i))
end
