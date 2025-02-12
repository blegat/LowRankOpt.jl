"""
    SetDotProducts(set::MOI.AbstractSet, vectors::AbstractVector)

Given a set `set` of dimension `d` and `m` vectors `a_1`, ..., `a_m` given in `vectors`, this is the set:
``\\{ ((\\langle a_1, x \\rangle, ..., \\langle a_m, x \\rangle) \\in \\mathbb{R}^{m} : x \\in \\text{set} \\}.``
"""
struct SetDotProducts{S,A,V<:AbstractVector{A}} <: MOI.AbstractVectorSet
    set::S
    vectors::V
end

function Base.:(==)(s1::SetDotProducts, s2::SetDotProducts)
    return s1.set == s2.set && s1.vectors == s2.vectors
end

function Base.copy(s::SetDotProducts)
    return SetDotProducts(copy(s.set), copy(s.vectors))
end

MOI.dimension(s::SetDotProducts) = length(s.vectors)

function MOI.Bridges.Constraint.conversion_cost(
    ::Type{SetDotProducts{S,A1,Vector{A1}}},
    ::Type{SetDotProducts{S,A2,Vector{A2}}},
) where {S,A1,A2}
    return MOI.Bridges.Constraint.conversion_cost(A1, A2)
end

function Base.convert(::Type{SetDotProducts{S,A,V}}, set::SetDotProducts) where {S,A,V}
    return SetDotProducts{S,A,V}(set.set, convert(V, set.vectors))
end

"""
    LinearCombinationInSet(set::MOI.AbstractSet, matrices::AbstractVector)

Given a set `set` of dimension `d` and `m` vectors `a_1`, ..., `a_m` given in `vectors`, this is the set:
``\\{ (y \\in \\mathbb{R}^{m} : \\sum_{i=1}^m y_i a_i \\in \\text{set} \\}.``
"""
struct LinearCombinationInSet{S,A,V<:AbstractVector{A}} <: MOI.AbstractVectorSet
    set::S
    vectors::V
end

function Base.:(==)(s1::LinearCombinationInSet, s2::LinearCombinationInSet)
    return s1.set == s2.set && s1.vectors == s2.vectors
end

function Base.copy(s::LinearCombinationInSet)
    return LinearCombinationInSet(copy(s.set), copy(s.vectors))
end

MOI.dimension(s::LinearCombinationInSet) = length(s.vectors)

function MOI.Bridges.Constraint.conversion_cost(
    ::Type{LinearCombinationInSet{S,A1,Vector{A1}}},
    ::Type{LinearCombinationInSet{S,A2,Vector{A2}}},
) where {S,A1,A2}
    return MOI.Bridges.Constraint.conversion_cost(A1, A2)
end

function Base.convert(
    ::Type{LinearCombinationInSet{S,A,V}},
    set::LinearCombinationInSet,
) where {S,A,V}
    return LinearCombinationInSet{S,A,V}(set.set, convert(V, set.vectors))
end

function MOI.dual_set(s::SetDotProducts)
    return LinearCombinationInSet(s.set, s.vectors)
end

function MOI.dual_set_type(::Type{SetDotProducts{S,A,V}}) where {S,A,V}
    return LinearCombinationInSet{S,A,V}
end

function MOI.dual_set(s::LinearCombinationInSet)
    return SetDotProducts(s.side_dimension, s.vectors)
end

function MOI.dual_set_type(::Type{LinearCombinationInSet{S,A,V}}) where {S,A,V}
    return SetDotProducts{S,A,V}
end

abstract type AbstractFactorization{T,F} <: AbstractMatrix{T} end

function Base.size(m::AbstractFactorization)
    n = size(m.factor, 1)
    return (n, n)
end

"""
    struct Factorization{
        T,
        F<:Union{AbstractVector{T},AbstractMatrix{T}},
        D<:Union{T,AbstractVector{T}},
    } <: AbstractMatrix{T}
        factor::F
        scaling::D
    end

Matrix corresponding to `factor * Diagonal(diagonal) * factor'`.
If `factor` is a vector and `diagonal` is a scalar, this corresponds to
the matrix `diagonal * factor * factor'`.
If `factor` is a matrix and `diagonal` is a vector, this corresponds to
the matrix `factor * Diagonal(scaling) * factor'`.
"""
struct Factorization{
    T,
    F<:Union{AbstractVector{T},AbstractMatrix{T}},
    D<:Union{T,AbstractVector{T}},
} <: AbstractFactorization{T,F}
    factor::F
    scaling::D
    function Factorization(
        factor::AbstractMatrix{T},
        scaling::AbstractVector{T},
    ) where {T}
        if length(scaling) != size(factor, 2)
            error(
                "Length `$(length(scaling))` of diagonal does not match number of columns `$(size(factor, 2))` of factor",
            )
        end
        return new{T,typeof(factor),typeof(scaling)}(factor, scaling)
    end
    function Factorization(factor::AbstractVector{T}, scaling::T) where {T}
        return new{T,typeof(factor),typeof(scaling)}(factor, scaling)
    end
end

function Base.getindex(m::Factorization, i::Int, j::Int)
    return sum(
        m.factor[i, k] * m.scaling[k] * m.factor[j, k]' for
        k in eachindex(m.scaling)
    )
end

"""
    struct PositiveSemidefiniteFactorization{
        T,
        F<:Union{AbstractVector{T},AbstractMatrix{T}},
    } <: AbstractFactorization{T,F}
        factor::F
    end

Matrix corresponding to `factor * Diagonal(diagonal) * factor'`.
If `factor` is a vector and `diagonal` is a scalar, this corresponds to
the matrix `diagonal * factor * factor'`.
If `factor` is a matrix and `diagonal` is a vector, this corresponds to
the matrix `factor * Diagonal(scaling) * factor'`.
"""
struct PositiveSemidefiniteFactorization{
    T,
    F<:Union{AbstractVector{T},AbstractMatrix{T}},
} <: AbstractFactorization{T,F}
    factor::F
end

function Base.getindex(m::PositiveSemidefiniteFactorization, i::Int, j::Int)
    return sum(m.factor[i, k] * m.factor[j, k]' for k in axes(m.factor, 2))
end

function MOI.Bridges.Constraint.conversion_cost(
    ::Type{<:AbstractMatrix},
    ::Type{<:AbstractMatrix},
)
    return Inf
end

function MOI.Bridges.Constraint.conversion_cost(
    ::Type{<:Factorization{T,F}},
    ::Type{PositiveSemidefiniteFactorization{T,F}},
) where {T,F}
    return 1.0
end

function Base.convert(
    ::Type{Factorization{T,F,T}},
    f::PositiveSemidefiniteFactorization{T,F},
) where {T,F<:AbstractVector}
    return Factorization{T,F}(f.factor, one(T))
end

function Base.convert(
    ::Type{Factorization{T,F,Vector{T}}},
    f::PositiveSemidefiniteFactorization{T,F},
) where {T,F<:AbstractVector}
    return Factorization{T,F,Vector{T}}(f.factor, ones(T, size(f.factor, 2)))
end

struct TriangleVectorization{T,M<:AbstractMatrix{T}} <: AbstractVector{T}
    matrix::M
end

function MOI.Bridges.Constraint.conversion_cost(
    ::Type{TriangleVectorization{T,M1}},
    ::Type{TriangleVectorization{T,M2}},
) where {T,M1,M2}
    return MOI.Bridges.Constraint.conversion_cost(M1, M2)
end

function Base.size(v::TriangleVectorization)
    n = size(v.matrix, 1)
    return (MOI.Utilities.trimap(n, n),)
end

function Base.getindex(v::TriangleVectorization, k::Int)
    return getindex(v.matrix, MOI.Utilities.inverse_trimap(k)...)
end
