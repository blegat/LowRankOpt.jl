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
* The matrix ``A_{i,j}`` is given by `A[i,j]`.
"""
mutable struct Model{T,C<:AbstractMatrix{T},A<:AbstractMatrix{T}} <:
               AbstractModel{T}
    meta::NLPModels.NLPModelMeta{T,Vector{T}}
    dim::Dimensions
    C::Vector{C}
    A::Matrix{A}
    b::Vector{T}
    d_lin::SparseArrays.SparseVector{T,Int64}
    C_lin::SparseArrays.SparseMatrixCSC{T,Int64}
    msizes::Vector{Int64}

    function Model(
        C::Vector{CT},
        A::Matrix{AT},
        b::Vector{T},
        d_lin::SparseArrays.SparseVector{T,Int64},
        C_lin::SparseArrays.SparseMatrixCSC{T,Int64},
        msizes::Vector{Int64},
    ) where {T,CT<:AbstractMatrix{T},AT<:AbstractMatrix{T}}
        model = new{T,CT,AT}()
        model.C = C
        model.A = A
        model.b = b
        model.d_lin = d_lin
        model.C_lin = C_lin
        model.msizes = msizes
        n = num_scalars(model)
        model.meta = NLPModels.NLPModelMeta{T,Vector{T}}(
            n + sum(abs2, msizes, init = 0),
            ncon = length(b),
        )
        offsets = n .+ [0; cumsum(abs2.(msizes))]
        model.dim = Dimensions(n, msizes, offsets)
        return model
    end
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
    return model.A[i.value, j]
end
function jac(model::Model, j::Integer, ::Type{ScalarIndex})
    return model.C_lin[j, :]
end
function norm_jac(model::Model{T}, i::MatrixIndex) where {T}
    if isempty(model.A)
        return zero(T)
    end
    return LinearAlgebra.norm(model.A[i.value, :])
end

#######################
###### J product ######
#######################

"""
    _rank_one_rowview_batch(row)

If every entry `j` of `row` is a rank-1 `Factorization` whose `factor` is
exactly `view(U, j, :)` for a single common parent matrix `U` (with a
0-dim scalar `scaling`), return `(U, weights)` where
`weights[j] == only(row[j].scaling)`. Otherwise return `nothing`.

This is the structural detector for the sample-from-a-parent shape produced
by `SumOfSquares.Bridges.Variable.LowRankBridge`: every per-Lagrange-point
rank-1 factor `u_j` is a row of `U = MB.transformation_to(gram, lagrange)`.
When the detector fires, `add_jprod!` / `BurerMonteiro.add_jtprod!` use a
single batched `mul!(buf, U, ...)` instead of `n` per-constraint inner
products — which, for `U::MB.TrigEvalMatrix`, collapses `n` per-row
dot products into one batched FFT per column of the rank-`r` factor.
"""
function _rank_one_rowview_batch(row::AbstractVector)
    isempty(row) && return nothing
    n = length(row)
    e1 = first(row)
    e1 isa Factorization || return nothing
    e1.factor isa SubArray || return nothing
    ndims(e1.factor) == 1 || return nothing
    ndims(e1.scaling) == 0 || return nothing
    U = parent(e1.factor)
    U isa AbstractMatrix || return nothing
    size(U, 1) == n || return nothing
    T = eltype(U)
    weights = Vector{T}(undef, n)
    @inbounds for j in 1:n
        e = row[j]
        e isa Factorization || return nothing
        f = e.factor
        f isa SubArray || return nothing
        parent(f) === U || return nothing
        ndims(e.scaling) == 0 || return nothing
        idx = parentindices(f)
        length(idx) == 2 || return nothing
        idx[1] == j || return nothing
        idx[2] isa Base.Slice || return nothing
        weights[j] = only(e.scaling)
    end
    return U, weights
end

# `V = factor * factor'` (rank-`r` PSD with `Ones` scaling). Cons-path
# shape: `BurerMonteiro.cons!` calls `jprod!(model, X, X, cx)` which lands
# here with `V = X[i]::positive_semidefinite_factorization`.
function _add_jprod_batched!(
    Jv::AbstractVector,
    U::AbstractMatrix,
    weights::AbstractVector,
    V::Factorization{T,<:AbstractMatrix{T},<:FillArrays.Ones{T,1}},
) where {T}
    Z = U * V.factor                       # one batched FFT for `U::TrigEvalMatrix`
    @inbounds for j in eachindex(weights)
        Jv[j] += weights[j] * sum(abs2, view(Z, j, :))
    end
    return true
end

# `V = α * (left * right')` (rank-`r` asymmetric with constant `Fill` scaling).
# Jprod-path shape: `BurerMonteiro.jprod!` lands here with
# `V = _OuterProduct(X, V)[i]::AsymmetricFactorization(_, _, Fill(2, r))`.
function _add_jprod_batched!(
    Jv::AbstractVector,
    U::AbstractMatrix,
    weights::AbstractVector,
    V::AsymmetricFactorization{T,<:AbstractMatrix{T},<:FillArrays.Fill{T,1}},
) where {T}
    ZL = U * left_factor(V)
    ZR = U * right_factor(V)
    α = V.scaling.value
    @inbounds for j in eachindex(weights)
        s = zero(T)
        @simd for k in axes(ZL, 2)
            s = muladd(ZL[j, k], ZR[j, k], s)
        end
        Jv[j] += α * weights[j] * s
    end
    return true
end

function add_jprod!(
    model::Model,
    V::AbstractMatrix,
    Jv::AbstractVector,
    i::MatrixIndex,
)
    batch = _rank_one_rowview_batch(view(model.A, i.value, :))
    if batch !== nothing
        U, w = batch
        if _add_jprod_batched!(Jv, U, w, V)
            return
        end
    end
    for j in 1:model.meta.ncon
        Jv[j] += LinearAlgebra.dot(model.A[i.value, j], V)
    end
    return
end

# Fallback: signal "no specialisation" to the caller (which then runs the
# per-constraint loop). Concrete specialisations above return `true`.
_add_jprod_batched!(::AbstractVector, ::AbstractMatrix, ::AbstractVector, ::AbstractMatrix) = false

function jprod!(
    model::Model,
    _::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
    ::Type{ScalarIndex},
)
    return LinearAlgebra.mul!(Jv, model.C_lin, v)
end
