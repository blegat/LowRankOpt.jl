# Copyright (c) 2026: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    BlockColumns(block_ids, values, nblocks)

Store a component of the constraint matrices on selected PSD blocks.
`values[j, k]` is the matrix for constraint `j` on PSD block `block_ids[k]`.
The block identifiers must be strictly increasing. Other blocks have no
contribution. Missing blocks are skipped, without requiring `zero(M)` for
the matrix element type `M`.

Constraints are rows and stored PSD blocks are columns, so each block's
constraint matrices are contiguous. This container stores whole columns;
it does not represent sparsity within a column.

For example, with `n` equality constraints and three PSD blocks,
`BlockColumns([1, 3], values, 3)` stores an `n × 2` matrix of coefficients.
It contributes to blocks 1 and 3 and contributes nothing to block 2.
A model can sum this with another component by passing `(component1,
component2)` as its constraint data.
"""
struct BlockColumns{M<:AbstractMatrix}
    block_ids::Vector{Int}
    values::Matrix{M}
    nblocks::Int
    function BlockColumns(
        block_ids,
        values::Matrix{M},
        nblocks::Integer,
    ) where {M<:AbstractMatrix}
        ids = Int.(block_ids)
        length(ids) == size(values, 2) || throw(
            DimensionMismatch(
                "one block identifier is required per stored column",
            ),
        )
        all(
            i -> 1 <= ids[i] <= nblocks && (i == 1 || ids[i-1] < ids[i]),
            eachindex(ids),
        ) || throw(
            ArgumentError(
                "block identifiers must be strictly increasing and within 1:nblocks",
            ),
        )
        return new{M}(ids, values, Int(nblocks))
    end
end

Base.size(A::BlockColumns) = (size(A.values, 1), A.nblocks)
Base.size(A::BlockColumns, d::Integer) = d == 2 ? A.nblocks : size(A.values, d)
function Base.:(==)(A::BlockColumns, B::BlockColumns)
    return A.nblocks == B.nblocks &&
           A.block_ids == B.block_ids &&
           A.values == B.values
end

# Legacy input has PSD blocks as rows and constraints as columns.
function BlockColumns(A::Matrix{<:AbstractMatrix})
    return BlockColumns(1:size(A, 1), permutedims(A), size(A, 1))
end

function _block_slot(A::BlockColumns, i::Integer)
    k = searchsortedfirst(A.block_ids, i)
    return k <= length(A.block_ids) && A.block_ids[k] == i ? k : 0
end

function _block_column(A::BlockColumns, i::Integer)
    k = _block_slot(A, i)
    return iszero(k) ? nothing : view(A.values, :, k)
end

_as_component(A::BlockColumns) = A
_as_component(A::Matrix{<:AbstractMatrix}) = BlockColumns(A)

@inline _foreach_column(f, ::Tuple{}, i) = nothing
@inline function _foreach_column(f, components::Tuple, i)
    component = first(components)
    slot = _block_slot(component, i)
    if !iszero(slot)
        f(component.values, slot)
    end
    _foreach_column(f, Base.tail(components), i)
    return nothing
end

@inline _sum_columns(f, ::Tuple{}, i, init) = init
@inline function _sum_columns(f, components::Tuple, i, init)
    column = _block_column(first(components), i)
    value = isnothing(column) ? init : init + f(column)
    return _sum_columns(f, Base.tail(components), i, value)
end

# Construction runs once. Split a heterogeneous legacy array into concrete
# components, omitting blocks that contribute nothing to that component.
_component_zero(A::AbstractMatrix) = zero(A)
_component_zero(A::Factorization) = Factorization(zero(A.factor), A.scaling)
_matrix_iszero(A::AbstractMatrix) = iszero(A)
function _matrix_iszero(A::Factorization)
    return all(iszero, A.factor) || all(iszero, A.scaling)
end

function _typed_component(::Type{M}, A) where {M<:AbstractMatrix}
    ids = Int[]
    for i in axes(A, 1)
        if any(j -> A[i, j] isa M && !_matrix_iszero(A[i, j]), axes(A, 2))
            push!(ids, i)
        end
    end
    values = Matrix{M}(undef, size(A, 2), length(ids))
    for (k, i) in enumerate(ids)
        template = A[i, findfirst(j -> A[i, j] isa M, axes(A, 2))]
        for j in axes(A, 2)
            values[j, k] = A[i, j] isa M ? A[i, j] : _component_zero(template)
        end
    end
    return BlockColumns(ids, values, size(A, 1))
end

function _matrix_components(A::AbstractMatrix)
    types =
        unique(typeof(value) for value in A if !(value isa FillArrays.Zeros))
    return Tuple(_typed_component(M, A) for M in types)
end
