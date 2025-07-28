# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

@enum(SetInclusion, WITHOUT_SET, WITH_SET)

"""
    SetDotProducts{W}(set::MOI.AbstractVectorSet, vectors::AbstractVector)

Given a set `set` of dimension `d` and `m` vectors `a_1`, ..., `a_m` given in `vectors`,
if `W` is `WITHOUT_SET`, this is the set:
``\\{ ((\\langle a_1, x \\rangle, ..., \\langle a_m, x \\rangle) \\in \\mathbb{R}^{m} : x \\in \\text{set} \\}.``
if `W` is `WITH_SET`, this is the set:
``\\{ ((\\langle a_1, x \\rangle, ..., \\langle a_m, x \\rangle, x) \\in \\mathbb{R}^{m + d} : x \\in \\text{set} \\}.``
"""
struct SetDotProducts{W,S<:MOI.AbstractVectorSet,V<:AbstractVector} <:
       MOI.AbstractVectorSet
    set::S
    vectors::Vector{V}
end
function SetDotProducts{W}(
    set::S,
    vector::Vector{V},
) where {W,S<:MOI.AbstractVectorSet,V<:AbstractVector}
    return SetDotProducts{W,S,V}(set, vector)
end

function Base.:(==)(s1::SetDotProducts, s2::SetDotProducts)
    return s1.set == s2.set && s1.vectors == s2.vectors
end

function Base.copy(s::SetDotProducts{W}) where {W}
    return SetDotProducts{W}(copy(s.set), copy(s.vectors))
end

MOI.dimension(s::SetDotProducts{WITHOUT_SET}) = length(s.vectors)
function MOI.dimension(s::SetDotProducts{WITH_SET})
    return length(s.vectors) + MOI.dimension(s.set)
end

function MOI.Bridges.Constraint.conversion_cost(
    ::Type{SetDotProducts{W,S,V1}},
    ::Type{SetDotProducts{W,S,V2}},
) where {W,S,V1,V2}
    return MOI.Bridges.Constraint.conversion_cost(V1, V2)
end

function Base.convert(
    ::Type{SetDotProducts{W,S,V}},
    set::SetDotProducts,
) where {W,S,V}
    return SetDotProducts{W,S,V}(set.set, convert(Vector{V}, set.vectors))
end

"""
    LinearCombinationInSet{W}(set::MOI.AbstractVectorSet, matrices::AbstractVector)

Given a set `set` of dimension `d` and `m` vectors `a_1`, ..., `a_m` given in `vectors`, this is the set:
if `W` is `WITHOUT_SET`, this is the set:
``\\{ y \\in \\mathbb{R}^{m} : \\sum_{i=1}^m y_i a_i \\in \\text{set} \\}.``
if `W` is `WITH_SET`, this is the set:
``\\{ (y, c) \\in \\mathbb{R}^{m} : \\sum_{i=1}^m y_i a_i - c \\in \\text{set} \\}.``
"""
struct LinearCombinationInSet{W,S<:MOI.AbstractVectorSet,V} <:
       MOI.AbstractVectorSet
    set::S
    vectors::Vector{V}
end
function LinearCombinationInSet{W}(
    set::S,
    vector::Vector{V},
) where {W,S<:MOI.AbstractVectorSet,V<:AbstractVector}
    return LinearCombinationInSet{W,S,V}(set, vector)
end

function Base.:(==)(s1::LinearCombinationInSet, s2::LinearCombinationInSet)
    return s1.set == s2.set && s1.vectors == s2.vectors
end

function Base.copy(s::LinearCombinationInSet{W}) where {W}
    return LinearCombinationInSet{W}(copy(s.set), copy(s.vectors))
end

MOI.dimension(s::LinearCombinationInSet{WITHOUT_SET}) = length(s.vectors)
function MOI.dimension(s::LinearCombinationInSet{WITH_SET})
    return length(s.vectors) + MOI.dimension(s.set)
end

function MOI.Bridges.Constraint.conversion_cost(
    ::Type{LinearCombinationInSet{W,S,V1}},
    ::Type{LinearCombinationInSet{W,S,V2}},
) where {W,S,V1,V2}
    return MOI.Bridges.Constraint.conversion_cost(V1, V2)
end

function Base.convert(
    ::Type{LinearCombinationInSet{W,S,V}},
    set::LinearCombinationInSet,
) where {W,S,V}
    return LinearCombinationInSet{W,S,V}(
        set.set,
        convert(Vector{V}, set.vectors),
    )
end

function MOI.dual_set(s::SetDotProducts{W}) where {W}
    return LinearCombinationInSet{W}(s.set, s.vectors)
end

function MOI.dual_set_type(::Type{SetDotProducts{W,S,V}}) where {W,S,V}
    return LinearCombinationInSet{W,S,V}
end

function MOI.dual_set(s::LinearCombinationInSet{W}) where {W}
    return SetDotProducts{W}(s.set, s.vectors)
end

function MOI.dual_set_type(::Type{LinearCombinationInSet{W,S,V}}) where {W,S,V}
    return SetDotProducts{W,S,V}
end

function MOI.Utilities.set_dot(
    x::AbstractVector,
    y::AbstractVector,
    set::Union{SetDotProducts{WITH_SET},LinearCombinationInSet{WITH_SET}},
)
    n = length(set.vectors)
    # MOI defines a custom `view(::CanonicalVector, ::AbstractUnitRange)`
    # so we should use `view` in order to keep the `CanonicalVector`
    # structure.
    return LinearAlgebra.dot(view(x, 1:n), view(y, 1:n)) +
           MOI.Utilities.set_dot(
        view(x, (n+1):length(x)),
        view(y, (n+1):length(y)),
        set.set,
    )
end

function MOI.Utilities.dot_coefficients(
    x::AbstractVector,
    set::Union{SetDotProducts{WITH_SET},LinearCombinationInSet{WITH_SET}},
)
    c = copy(x)
    n = length(set.vectors)
    c[(n+1):end] = MOI.Utilities.dot_coefficients(x[(n+1):end], set.set)
    return c
end

function MOI.side_dimension(set::Union{SetDotProducts,LinearCombinationInSet})
    return MOI.side_dimension(set.set)
end
