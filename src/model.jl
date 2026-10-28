# Adapted from Loraine.jl

import MutableArithmetics as MA
import MathOptInterface as MOI
import NLPModels
import UnsafeArrays

abstract type AbstractModel{T} <: NLPModels.AbstractNLPModel{T,Vector{T}} end
Base.broadcastable(model::AbstractModel) = Ref(model)

function NLPModels.cons!(
    model::AbstractModel,
    x::AbstractVector,
    cx::AbstractVector,
)
    NLPModels.jprod!(model, x, x, cx)
    cx .-= cons_constant(model)
    return cx
end

function NLPModels.jprod!(
    model::AbstractModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    jprod!(model, x, v[ScalarIndex], Jv, ScalarIndex)
    for i in matrix_indices(model)
        add_jprod!(model, v[i], Jv, i)
    end
    return Jv
end

"""
    Model

Model representing the primal-dual pair of problems:
```math
\\begin{aligned}
\\min {} & \\sum_{i=1}^\\text{nlmi}
\\langle C_i, X_i \\rangle + \\langle d_\\text{lin}, x \\rangle &
\\max {} & b^\\top y
\\\\
\\text{s.t. } & \\sum_{i=1}^\\text{nlmi} \\langle A_{i,j}, X_i \\rangle + (C_\\text{lin} x)_j = b_j
\\qquad
\\forall j \\in \\{1,\\ldots,m\\} &
\\text{s.t. } & \\sum_{j=1}^m y_j A_{i,j} \\preceq C_i
\\qquad
\\forall i \\in \\{1,\\ldots,\\text{nlmi}\\}
\\\\
& x \\ge 0, X_i \\succeq 0
\\qquad
\\forall i \\in \\{1,\\ldots,\\text{nlmi}\\} &
& C_\\text{lin}^\\top y \\le d_\\text{lin}
\\end{aligned}
```
This corresponds to [this primal-dual pair](https://plato.asu.edu/dimacs/node2.html).
The fields of the `struct` as related to the arrays of the above formulation as follows:

* The ``i``th PSD constraint is of size `msize[i] × msisze[i]`
* The matrix ``C_i`` is given by `C[i]`.
* `A` is a tuple of [`BlockColumns`](@ref) components. The matrix
  ``A_{i,j}`` is the sum of their entries for PSD block `i` and constraint
  `j`. A dense outer matrix in the original block-by-constraint layout is
  accepted and split into homogeneous components. Tuple entries can also
  use dense outer matrices in that layout.
"""
mutable struct Model{T,C<:AbstractMatrix{T},A<:Tuple} <: AbstractModel{T}
    meta::NLPModels.NLPModelMeta{T,Vector{T}}
    dim::Dimensions
    C::Vector{C}
    A::A
    b::Vector{T}
    d_lin::SparseArrays.SparseVector{T,Int64}
    C_lin::SparseArrays.SparseMatrixCSC{T,Int64}
    msizes::Vector{Int64}

    function Model(
        C::Vector{CT},
        A::Tuple,
        b::Vector{T},
        d_lin::SparseArrays.SparseVector{T,Int64},
        C_lin::SparseArrays.SparseMatrixCSC{T,Int64},
        msizes::Vector{Int64},
    ) where {T,CT<:AbstractMatrix{T}}
        components = map(_as_component, A)
        for component in components
            size(component) == (length(b), length(C)) || throw(
                DimensionMismatch(
                    "constraint components must have one row per constraint and one logical column per PSD block",
                ),
            )
            for (k, i) in enumerate(component.block_ids),
                j in axes(component.values, 1)

                coefficient = component.values[j, k]
                coefficient isa AbstractMatrix{T} || throw(
                    ArgumentError(
                        "constraint coefficients must have element type $T",
                    ),
                )
                size(coefficient) == (msizes[i], msizes[i]) || throw(
                    DimensionMismatch(
                        "constraint matrix has the wrong size for PSD block $i",
                    ),
                )
            end
        end
        model = new{T,CT,typeof(components)}()
        model.C = C
        model.A = components
        model.b = b
        model.d_lin = d_lin
        model.C_lin = C_lin
        model.msizes = msizes
        n = num_scalars(model)
        model.meta = NLPModels.NLPModelMeta{T,Vector{T}}(
            Int(n + sum(abs2, msizes, init = 0)),
            ncon = length(b),
        )
        offsets = n .+ [0; cumsum(abs2.(msizes))]
        model.dim = Dimensions(n, msizes, offsets)
        return model
    end
end

# Preserve the existing constructor for one dense outer matrix of matrices.
function Model(C, A::Matrix{<:AbstractMatrix}, b, d_lin, C_lin, msizes)
    return Model(C, _matrix_components(A), b, d_lin, C_lin, msizes)
end

function NLPModels.unconstrained(model::Model)
    return iszero(model.meta.ncon)
end

# TODO the scalar actually have lower bounds and the SDP variables too
#      but these are not box constraints
NLPModels.has_bounds(::Model) = false

num_scalars(model::Model) = length(model.d_lin)

num_matrices(model::Model) = length(model.C)

# Julia has troubles inferring the return type of constructors
_matrix_index(i)::MatrixIndex = MatrixIndex(i)

function matrix_indices(model::Union{Model,AbstractSolution})
    return MOI.Utilities.LazyMap{MatrixIndex}(
        _matrix_index,
        Base.OneTo(num_matrices(model)),
    )
end

side_dimension(model::Model, i::MatrixIndex) = model.msizes[i.value]

#######################
###### Objective ######
#######################

function obj(model::Model, X::AbstractMatrix, i::MatrixIndex)
    return LinearAlgebra.dot(model.C[i.value], X)
end

function obj(model::Model, x::AbstractVector, ::Type{MatrixIndex})
    result = zero(eltype(x))
    for i in matrix_indices(model)
        result += obj(model, x[i], i)
    end
    return result
end

function obj(model::Model, x::AbstractVector, ::Type{ScalarIndex})
    return LinearAlgebra.dot(model.d_lin, x[ScalarIndex])
end

function NLPModels.obj(model::Model, x::AbstractVector)
    return obj(model, x, ScalarIndex) + obj(model, x, MatrixIndex)
end

function NLPModels.grad!(model::Model, _::AbstractVector, g::AbstractVector)
    copyto!(g[ScalarIndex], model.d_lin)
    for i in matrix_indices(model)
        copyto!(g[i], model.C[i.value])
    end
    return g
end

dual_obj(model::Model, y::AbstractVector) = LinearAlgebra.dot(model.b, y)

function jtprod!(
    model::Model,
    y::AbstractVector,
    vJ::AbstractVector,
    ::Type{ScalarIndex},
)
    return LinearAlgebra.mul!(vJ, model.C_lin', y)
end

function dual_cons!(model::Model, y::AbstractVector, res, ::Type{ScalarIndex})
    copyto!(res, model.d_lin)
    return LinearAlgebra.mul!(res, model.C_lin', y, -1, true)
end

grad(model::Model, ::Type{ScalarIndex}) = model.d_lin
grad(model::Model, i::MatrixIndex) = model.C[i.value]

#########################
###### Constraints ######
#########################

cons_constant(model::Model) = model.b

# Should be only used with `norm`
jac(model::Model, ::Type{ScalarIndex}) = model.C_lin
function jac(model::Model, j::Integer, i::MatrixIndex)
    return _coefficient_sum(
        model.A,
        j,
        i.value,
        side_dimension(model, i),
        eltype(model.b),
    )
end

function _coefficient_sum(::Tuple{}, j, i, d, ::Type{T}) where {T}
    return FillArrays.Zeros{T}(d, d)
end
@inline function _coefficient_sum(
    components::Tuple,
    j,
    i,
    d,
    ::Type{T},
) where {T}
    column = _block_column(first(components), i)
    tail = _coefficient_sum(Base.tail(components), j, i, d, T)
    if isnothing(column)
        return tail
    elseif tail isa FillArrays.Zeros
        return column[j]
    end
    return column[j] + tail
end

function jac(model::Model, j::Integer, ::Type{ScalarIndex})
    return model.C_lin[j, :]
end
function norm_jac(model::Model{T}, i::MatrixIndex) where {T}
    iszero(model.meta.ncon) && return zero(T)
    return LinearAlgebra.norm((jac(model, j, i) for j in 1:model.meta.ncon))
end

#######################
###### J product ######
#######################

function add_jprod!(
    model::Model,
    V::AbstractMatrix,
    Jv::AbstractVector,
    i::MatrixIndex,
)
    _foreach_column(model.A, i.value) do values, slot
        return _add_column_jprod!(values, slot, V, Jv)
    end
    return Jv
end

function _coefficient_dot(A, V)
    return LinearAlgebra.dot(A, V)
end
# `dot(Diagonal, Matrix)` copies the matrix diagonal on supported Julia
# versions. Contract the entries directly to keep the generic Schur path
# allocation-free for diagonal coefficients too.
function _coefficient_dot(A::LinearAlgebra.Diagonal, V::AbstractMatrix)
    return sum(
        eachindex(A.diag);
        init = zero(promote_type(eltype(A), eltype(V))),
    ) do k
        return conj(A.diag[k]) * V[k, k]
    end
end
function _coefficient_dot(::FillArrays.Zeros{T}, V) where {T}
    return zero(promote_type(T, eltype(V)))
end
function _coefficient_dot(A::AbstractFactorization, V::AbstractMatrix)
    return LinearAlgebra.dot(V, A)
end
function _coefficient_dot(A::AbstractFactorization, V::AbstractFactorization)
    return LinearAlgebra.dot(A, V)
end
function _add_column_jprod!(values, slot, V, Jv)
    for j in eachindex(Jv)
        Jv[j] += _coefficient_dot(values[j, slot], V)
    end
    return nothing
end

function jprod!(
    model::Model,
    _::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
    ::Type{ScalarIndex},
)
    return LinearAlgebra.mul!(Jv, model.C_lin, v)
end
