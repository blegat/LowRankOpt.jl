# Adapted from Loraine.jl

import MutableArithmetics as MA
import MathOptInterface as MOI
import NLPModels
import UnsafeArrays

abstract type AbstractDualModel{T} <: NLPModels.AbstractNLPModel{T,Vector{T}} end
Base.broadcastable(model::AbstractDualModel) = Ref(model)

function NLPModels.cons!(
    model::AbstractDualModel,
    x::AbstractVector,
    cx::AbstractVector,
)
    NLPModels.jprod!(model, x, x, cx)
    cx .-= cons_constant(model)
    return cx
end

function NLPModels.jprod!(
    model::AbstractDualModel,
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
    DualModel(model::Model)

Explicit cone-variable adapter of `model`. Its nonnegative scalar variables
`z` and PSD matrix variables `X[i]` satisfy
`(C_lin * z)[j] + sum(dot(A[i,j], X[i]) for i) = b[j]`. It minimizes
`dot(d_lin, z) + sum(dot(C[i], X[i]))`, the negative of the maximization dual
objective, excluding the original objective constant.

The adapter shares affine coefficients with its parent where possible and
owns the metadata for its cone variables. [`primal_model`](@ref) returns the
original model, preserving its objective constant and native constraints.
"""
struct DualModel{T,C<:AbstractMatrix{T},A<:Tuple,M<:Model{T}} <:
       AbstractDualModel{T}
    model::M
    meta::NLPModels.NLPModelMeta{T,Vector{T}}
    dim::ConeDimensions
    C::Vector{C}
    A::A
    b::Vector{T}
    d_lin::SparseArrays.SparseVector{T,Int64}
    C_lin::SparseArrays.SparseMatrixCSC{T,Int64}
    msizes::Vector{Int64}
end

# Explicitly supply transformed coefficients while retaining the original
# model, e.g. when finite scalar bounds are expanded into cone variables.
function DualModel(
    model::Model{T},
    C::Vector{CT},
    A::Tuple,
    d_lin::SparseArrays.SparseVector{T},
    C_lin::SparseArrays.SparseMatrixCSC{T},
    msizes::AbstractVector{<:Integer},
) where {T,CT<:AbstractMatrix{T}}
    d_lin = convert(SparseArrays.SparseVector{T,Int64}, d_lin)
    C_lin = convert(SparseArrays.SparseMatrixCSC{T,Int64}, C_lin)
    msizes = convert(Vector{Int64}, msizes)
    n = length(d_lin)
    meta = NLPModels.NLPModelMeta{T,Vector{T}}(
        Int(n + sum(abs2, msizes; init = 0));
        ncon = length(model.b),
    )
    dim = ConeDimensions(n, msizes, n .+ [0; cumsum(abs2.(msizes))])
    return DualModel(model, meta, dim, C, A, model.b, d_lin, C_lin, msizes)
end

"""
    dual_model(model::Model)

Create a cone-variable adapter sharing the original model's coefficients.
The original model is available as `primal_model(adapter)`.
"""
function dual_model(model::Model; equalities = true)
    if num_scalars(model) == length(model.d_lin) &&
       iszero(num_second_order_cones(model))
        return DualModel(
            model,
            model.C,
            model.A,
            model.d_lin,
            model.C_lin,
            model.msizes,
        )
    end
    C, d, _, _ = scalar_bounds(model; equalities)
    costs, coefficients, sizes = lifted_soc_data(model)
    return DualModel(
        model,
        costs,
        coefficients,
        SparseArrays.sparsevec(d),
        C,
        sizes,
    )
end

DualModel(model::Model) = dual_model(model)
dual_model(model::AbstractDualModel) = model
primal_model(model::DualModel) = model.model

# Convenience for callers that explicitly request the cone-variable form.
function DualModel(C::Vector, A, b, d_lin, C_lin, msizes)
    return DualModel(Model(C, A, b, d_lin, C_lin, msizes))
end

function NLPModels.unconstrained(model::DualModel)
    return iszero(model.meta.ncon)
end

# TODO the scalar actually have lower bounds and the SDP variables too
#      but these are not box constraints
NLPModels.has_bounds(::DualModel) = false

num_scalars(model::DualModel) = length(model.d_lin)

num_matrices(model::DualModel) = length(model.C)

# Julia has troubles inferring the return type of constructors
_matrix_index(i)::MatrixIndex = MatrixIndex(i)

function matrix_indices(model::Union{DualModel,AbstractConeVector})
    return MOI.Utilities.LazyMap{MatrixIndex}(
        _matrix_index,
        Base.OneTo(num_matrices(model)),
    )
end

side_dimension(model::DualModel, i::MatrixIndex) = model.msizes[i.value]

#######################
###### Objective ######
#######################

function obj(model::DualModel, X::AbstractMatrix, i::MatrixIndex)
    return LinearAlgebra.dot(model.C[i.value], X)
end

function obj(model::DualModel, x::AbstractVector, ::Type{MatrixIndex})
    result = zero(eltype(x))
    for i in matrix_indices(model)
        result += obj(model, x[i], i)
    end
    return result
end

function obj(model::DualModel, x::AbstractVector, ::Type{ScalarIndex})
    return LinearAlgebra.dot(model.d_lin, x[ScalarIndex])
end

function NLPModels.obj(model::DualModel, x::AbstractVector)
    return obj(model, x, ScalarIndex) + obj(model, x, MatrixIndex)
end

function NLPModels.grad!(model::DualModel, _::AbstractVector, g::AbstractVector)
    copyto!(g[ScalarIndex], model.d_lin)
    for i in matrix_indices(model)
        copyto!(g[i], model.C[i.value])
    end
    return g
end

dual_obj(model::DualModel, y::AbstractVector) = LinearAlgebra.dot(model.b, y)

function jtprod!(
    model::DualModel,
    y::AbstractVector,
    vJ::AbstractVector,
    ::Type{ScalarIndex},
)
    return LinearAlgebra.mul!(vJ, model.C_lin', y)
end

function dual_cons!(
    model::DualModel,
    y::AbstractVector,
    res,
    ::Type{ScalarIndex},
)
    copyto!(res, model.d_lin)
    return LinearAlgebra.mul!(res, model.C_lin', y, -1, true)
end

grad(model::DualModel, ::Type{ScalarIndex}) = model.d_lin
grad(model::DualModel, i::MatrixIndex) = model.C[i.value]

#########################
###### Constraints ######
#########################

cons_constant(model::DualModel) = model.b

# Should be only used with `norm`
jac(model::DualModel, ::Type{ScalarIndex}) = model.C_lin
function jac(model::DualModel, j::Integer, i::MatrixIndex)
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

function jac(model::DualModel, j::Integer, ::Type{ScalarIndex})
    return model.C_lin[j, :]
end
function norm_jac(model::DualModel{T}, i::MatrixIndex) where {T}
    iszero(model.meta.ncon) && return zero(T)
    return LinearAlgebra.norm((jac(model, j, i) for j in 1:model.meta.ncon))
end

#######################
###### J product ######
#######################

function add_jprod!(
    model::DualModel,
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
    model::DualModel,
    _::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
    ::Type{ScalarIndex},
)
    return LinearAlgebra.mul!(Jv, model.C_lin, v)
end
