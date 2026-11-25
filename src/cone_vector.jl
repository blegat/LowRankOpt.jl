struct ConeDimensions
    num_scalars::Int64
    second_order_dimensions::Vector{Int64}
    side_dimensions::Vector{Int64}
    offsets::Vector{Int64}
end
function ConeDimensions(n, dimensions, offsets)
    return ConeDimensions(n, Int[], dimensions, offsets)
end
function num_second_order_cones(dim::ConeDimensions)
    return length(dim.second_order_dimensions)
end

num_matrices(d::ConeDimensions) = length(d.side_dimensions)
Base.length(d::ConeDimensions) = d.offsets[end]

struct ScalarIndex
    value::Int64
end

struct SOCIndex
    value::Int64
end
Base.broadcastable(i::SOCIndex) = Ref(i)

struct MatrixIndex
    value::Int64
end
Base.broadcastable(i::MatrixIndex) = Ref(i)

abstract type AbstractConeVector{T} <: AbstractVector{T} end

function Base.getindex(
    s::AbstractConeVector,
    i::Union{Type{ScalarIndex},SOCIndex,MatrixIndex},
)
    return view(s, i)
end

struct VectorizedConeVector{T,V<:AbstractVector{T}} <: AbstractConeVector{T}
    x::V
    dim::ConeDimensions
end

function UnsafeArrays.unsafe_uview(s::VectorizedConeVector)
    return VectorizedConeVector(UnsafeArrays.uview(s.x), s.dim)
end

function LinearAlgebra.dot(x::VectorizedConeVector, z::VectorizedConeVector)
    return LinearAlgebra.dot(x.x, z.x)
end

num_matrices(x::VectorizedConeVector) = num_matrices(x.dim)
num_second_order_cones(x::VectorizedConeVector) = num_second_order_cones(x.dim)

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

function Base.to_index(s::VectorizedConeVector, index::SOCIndex)
    first =
        s.dim.num_scalars +
        sum(view(s.dim.second_order_dimensions, 1:(index.value-1))) +
        1
    return first:(first+s.dim.second_order_dimensions[index.value]-1)
end
function Base.view(s::VectorizedConeVector, index::SOCIndex)
    return view(s.x, Base.to_index(s, index))
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
    second_order::Vector{Vector{T}}
    matrices::Vector{MT}
end

function ShapedConeVector(scalars::Vector{T}, matrices) where {T}
    return ShapedConeVector(scalars, Vector{T}[], matrices)
end
function ShapedConeVector{T,MT}(scalars, matrices) where {T,MT}
    return ShapedConeVector{T,MT}(scalars, Vector{T}[], matrices)
end
num_second_order_cones(x::ShapedConeVector) = length(x.second_order)
_soc_index(i)::SOCIndex = SOCIndex(i)
function second_order_indices(model)
    return MOI.Utilities.LazyMap{SOCIndex}(
        _soc_index,
        Base.OneTo(num_second_order_cones(model)),
    )
end

function Base.size(s::ShapedConeVector)
    return (
        length(s.scalars) +
        sum(length, s.second_order, init = 0) +
        sum(length, s.matrices, init = 0),
    )
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
           sum(
               LinearAlgebra.dot.(a.second_order, b.second_order);
               init = zero(T),
           ) +
           sum(eachindex(a.matrices); init = zero(T)) do i
               return LinearAlgebra.dot(a.matrices[i], b.matrices[i])
           end
end

Base.view(s::ShapedConeVector, ::Type{ScalarIndex}) = s.scalars
Base.view(s::ShapedConeVector, i::SOCIndex) = s.second_order[i.value]
Base.view(s::ShapedConeVector, i::MatrixIndex) = s.matrices[i.value]

# Leave integer getindex undefined. Locating one entry requires scanning the
# preceding blocks, so generic elementwise operations could take quadratic time.
# Traverse whole blocks when writing a flat vector.
function _collect_blocks(f, ::Type{T}, s::ShapedConeVector) where {T}
    result = Vector{T}(undef, length(s))
    offset = 0
    for block in Iterators.flatten(((s.scalars,), s.second_order, s.matrices))
        for value in block
            offset += 1
            result[offset] = f(value)
        end
    end
    return result
end

Base.collect(s::ShapedConeVector{T}) where {T} = _collect_blocks(identity, T, s)
function Base.:-(s::ShapedConeVector{T}) where {T}
    return _collect_blocks(-, Base.promote_op(-, T), s)
end
