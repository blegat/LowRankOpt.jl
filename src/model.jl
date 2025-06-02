# Adapted from Loraine.jl

import SparseArrays
import LinearAlgebra
import MutableArithmetics as MA
import MathOptInterface as MOI
import NLPModels

struct Solution{T} <: AbstractVector{T}
    scalars::Vector{T}
    matrices::Vector{LinearAlgebra.Symmetric{T,Matrix{T}}}
end

function Base.zero(::Type{Solution{T}}, num_scalars::Integer, side_dimensions) where {T}
    return Solution{T}(
        zeros(T, num_scalars),
        [LinearAlgebra.Symmetric(zeros(T, d, d)) for d in side_dimensions]
    )
end

struct Meta{T} <: NLPModels.AbstractNLPModelMeta{T,Solution{T}}
  nvar::Int
  x0::Solution{T}
  ncon::Int
  y0::Vector{T}
  minimize::Bool
end

"""
    Model

Model representing the problem:
```math
\\begin{aligned}
\\max {} & b^\\top y - b_\\text{const}
\\\\
& \\sum_{j=1}^n y_j A_{i,j} \\preceq C_i
\\qquad
\\forall i \\in \\{1,\\ldots,\\text{nlmi}\\}
\\\\
& C_\\text{lin}^\\top y \\le d_\\text{lin}
\\end{aligned}
```
The fields of the `struct` as related to the arrays of the above formulation as follows:

* The ``i``th PSD constraint is of size `msize[i] × msisze[i]`
* The matrix ``C_i`` is given by `C[i]`.
* The matrix ``A_{i,j}`` is given by `-A[i,j]`.
"""
mutable struct Model{T,A<:AbstractMatrix{T}} <: NLPModels.AbstractNLPModel{T,Vector{T}}
    meta::Meta{T}
    C::Vector{SparseArrays.SparseMatrixCSC{T,Int}}
    A::Matrix{A}
    b::Vector{T}
    b_const::T
    d_lin::SparseArrays.SparseVector{T,Int64}
    C_lin::SparseArrays.SparseMatrixCSC{T,Int64}
    msizes::Vector{Int64}

    function Model(
        C::Vector{SparseArrays.SparseMatrixCSC{T,Int}},
        A::Matrix{AT},
        b::Vector{T},
        b_const::T,
        d_lin::SparseArrays.SparseVector{T,Int64},
        C_lin::SparseArrays.SparseMatrixCSC{T,Int64},
        msizes::Vector{Int64},
    ) where {T,AT<:AbstractMatrix{T}}
        model = new{T,AT}()
        model.C = C
        model.A = A
        model.b = b
        model.b_const = b_const
        model.d_lin = d_lin
        model.C_lin = C_lin
        model.msizes = msizes
        model.meta = Meta{T}(
            num_scalars(model) + sum(
                Base.Fix1(side_dimension, model),
                matrix_indices(model);
                init = 0
            ),
            zero(Solution{T}, num_scalars(model), msizes),
            length(b),
            zero(b),
            true,
        )
        return model
    end
end

function NLPModels.unconstrained(model::Model)
    return iszero(num_constraints(model))
end

# TODO the scalar actually have lower bounds and the SDP variables too
#      but these are not box constraints
NLPModels.has_bounds(::Model) = false

struct ScalarIndex
    value::Int64
end

num_scalars(model::Model) = length(model.d_lin)

function scalar_indices(model::Model)
    return MOI.Utilities.LazyMap{ScalarIndex}(ScalarIndex, Base.OneTo(num_scalars(model)))
end

struct MatrixIndex
    value::Int64
end

num_matrices(model::Model) = length(model.C)

function matrix_indices(model::Model)
    return MOI.Utilities.LazyMap{MatrixIndex}(MatrixIndex, Base.OneTo(num_matrices(model)))
end

side_dimension(model::Model, i::MatrixIndex) = model.msizes[i.value]

struct ConstraintIndex
    value::Int64
end
num_constraints(model::Model) = length(model.b)
function constraint_indices(model::Model)
    return MOI.Utilities.LazyMap{ConstraintIndex}(ConstraintIndex, Base.OneTo(num_constraints(model)))
end

# Should be only used with `norm`
NLPModels.jac(model::Model, ::Type{ScalarIndex}) = model.C_lin
NLPModels.jac(model::Model, i::ConstraintIndex, j::MatrixIndex) = model.A[j.value, i.value]
NLPModels.jac(model::Model, i::ConstraintIndex, ::Type{ScalarIndex}) = model.C_lin[i.value,:]
function norm_jac(model::Model{T}, i::MatrixIndex) where {T}
    if isempty(model.A)
        return zero(T)
    end
    return norm(model.A[i.value, :])
end

function NLPModels.obj(model::Model, X, i::MatrixIndex)
    return -LinearAlgebra.dot(model.C[i.value], X)
end

function NLPModels.obj(model::Model, x, ::Type{MatrixIndex})
    result = zero(eltype(x))
    for i in matrix_indices(model)
        result += NLPModels.obj(model, x[i], i)
    end
    return result
end

function NLPModels.obj(model::Model, x, ::Type{ScalarIndex})
    return -LinearAlgebra.dot(model.d_lin, x[ScalarIndex])
end

function NLPModels.obj(model::Model, x)
    return model.b_const + NLPModels.obj(model, x, MatrixIndex) + NLPModels.obj(model, x, ScalarIndex)
end

function NLPModels.grad!(model::Model, _, g)
    copyto!(g[ScalarIndex], model.d_lin)
    for i in matrix_indices(model)
        copyto!(g[i], model.C[i.value])
    end
    return g
end

dual_obj(model::Model, y) = -dot(model.b, y) + model.b_const

function jtprod(model::Model, ::Type{ScalarIndex}, y)
    return -model.C_lin' * y
end

function dual_cons(model::Model, ::Type{ScalarIndex}, y, S)
    return model.d_lin - S + jtprod(model, ScalarIndex, y)
end

function buffer_for_jtprod(model::Model)
    if iszero(num_matrices(model))
        return
    end
    return map(Base.Fix1(buffer_for_jtprod, model), matrix_indices(model))
end

function buffer_for_jtprod(model::Model, mat_idx::MatrixIndex)
    if iszero(num_constraints(model))
        return
    end
    # FIXME: at some point, switch to dense
    return sum(
        abs.(model.A[mat_idx.value, j])
        for j in 1:num_constraints(model)
    )
end

function _add_mul!(A::SparseArrays.SparseMatrixCSC, B::SparseArrays.SparseMatrixCSC, α)
    for col in axes(A, 2)
        range_A = SparseArrays.nzrange(A, col)
        it_A = iterate(range_A)
        for k in SparseArrays.nzrange(B, col)
            row_B = SparseArrays.rowvals(B)[k]
            while SparseArrays.rowvals(A)[it_A[1]] < row_B
                it_A = iterate(range_A, it_A[2])
            end
            @assert row_B == SparseArrays.rowvals(A)[it_A[1]]
            SparseArrays.nonzeros(A)[it_A[1]] += SparseArrays.nonzeros(B)[k] * α
        end
    end
end

_zero!(A::SparseArrays.SparseMatrixCSC) = fill!(SparseArrays.nonzeros(A), 0.0)

function jtprod!(buffer, model::Model, mat_idx::MatrixIndex, y)
    if iszero(num_constraints(model))
        return MA.Zero()
    end
    _zero!(buffer)
    for j in eachindex(y)
        _add_mul!(buffer, model.A[mat_idx.value, j], y[j])
    end
    return buffer
end

function dual_cons!(buffer, model::Model, mat_idx::MatrixIndex, y, S)
    i = mat_idx.value
    return jtprod!(buffer[i], model, mat_idx, y) + model.C[i] - S[i]
end

NLPModels.grad(model::Model, ::Type{ScalarIndex}) = -model.d_lin
NLPModels.grad(model::Model, i::MatrixIndex) = model.C[i.value]

cons_constant(model::Model) = model.b

function NLPModels.cons!(model::Model, x, cx)
    NLPModels.jprod!(model, x, x, cx)
    cx .*= -1
    cx .+= model.b
    return cx
end

function add_jprod!(model::Model, i::MatrixIndex, V, Jv)
    for j in 1:num_constraints(model)
        Jv[j] -= LinearAlgebra.dot(model.A[i.value, j], V)
    end
end

function NLPModels.jprod!(model::Model, _, v, Jv)
    LinearAlgebra.mul!(Jv, model.C_lin, v[ScalarIndex])
    for i in matrix_indices(model)
        add_jprod!(model, i, v[i], Jv)
    end
    return Jv
end
