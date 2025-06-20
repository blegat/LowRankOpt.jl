# Adapted from Loraine.jl

import SparseArrays
import LinearAlgebra
import MutableArithmetics as MA
import MathOptInterface as MOI
import NLPModels

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
mutable struct Model{T,A<:AbstractMatrix{T}} <:
               NLPModels.AbstractNLPModel{T,Vector{T}}
    meta::NLPModels.NLPModelMeta{T,Vector{T}}
    dim::Dimensions
    C::Vector{SparseArrays.SparseMatrixCSC{T,Int}}
    A::Matrix{A}
    b::Vector{T}
    d_lin::SparseArrays.SparseVector{T,Int64}
    C_lin::SparseArrays.SparseMatrixCSC{T,Int64}
    msizes::Vector{Int64}

    function Model(
        C::Vector{SparseArrays.SparseMatrixCSC{T,Int}},
        A::Matrix{AT},
        b::Vector{T},
        d_lin::SparseArrays.SparseVector{T,Int64},
        C_lin::SparseArrays.SparseMatrixCSC{T,Int64},
        msizes::Vector{Int64},
    ) where {T,AT<:AbstractMatrix{T}}
        model = new{T,AT}()
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

function matrix_indices(model::Union{Model,AbstractSolution})
    return MOI.Utilities.LazyMap{MatrixIndex}(
        MatrixIndex,
        Base.OneTo(num_matrices(model)),
    )
end

side_dimension(model::Model, i::MatrixIndex) = model.msizes[i.value]

# Should be only used with `norm`
NLPModels.jac(model::Model, ::Type{ScalarIndex}) = model.C_lin
function NLPModels.jac(model::Model, j::Integer, i::MatrixIndex)
    return model.A[i.value, j]
end
function NLPModels.jac(model::Model, j::Integer, ::Type{ScalarIndex})
    return model.C_lin[j, :]
end
function norm_jac(model::Model{T}, i::MatrixIndex) where {T}
    if isempty(model.A)
        return zero(T)
    end
    return LinearAlgebra.norm(model.A[i.value, :])
end

function NLPModels.obj(model::Model, X::AbstractMatrix, i::MatrixIndex)
    return LinearAlgebra.dot(model.C[i.value], X)
end

function NLPModels.obj(model::Model, x::AbstractVector, ::Type{MatrixIndex})
    result = zero(eltype(x))
    for i in matrix_indices(model)
        result += NLPModels.obj(model, x[i], i)
    end
    return result
end

function NLPModels.obj(model::Model, x::AbstractVector, ::Type{ScalarIndex})
    return LinearAlgebra.dot(model.d_lin, x[ScalarIndex])
end

function NLPModels.obj(model::Model, x::AbstractVector)
    return NLPModels.obj(model, x, MatrixIndex) +
           NLPModels.obj(model, x, ScalarIndex)
end

function NLPModels.grad!(model::Model, _::AbstractVector, g::AbstractVector)
    copyto!(g[ScalarIndex], model.d_lin)
    for i in matrix_indices(model)
        copyto!(g[i], model.C[i.value])
    end
    return g
end

dual_obj(model::Model, y::AbstractVector) = LinearAlgebra.dot(model.b, y)

function jtprod(model::Model, ::Type{ScalarIndex}, y::AbstractVector)
    return model.C_lin' * y
end

function buffer_for_jtprod(model::Model)
    if iszero(num_matrices(model))
        return
    end
    return map(Base.Fix1(buffer_for_jtprod, model), matrix_indices(model))
end

function buffer_for_jtprod(model::Model, mat_idx::MatrixIndex)
    if iszero(model.meta.ncon)
        return
    end
    # FIXME: at some point, switch to dense
    return sum(abs.(model.A[mat_idx.value, j]) for j in 1:model.meta.ncon)
end

# Computes `A .+= B * α`
function _add_mul!(
    A::SparseArrays.SparseMatrixCSC,
    B::SparseArrays.SparseMatrixCSC,
    α,
)
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
    _zero!(buffer)
    for j in eachindex(y)
        _add_mul!(buffer, model.A[mat_idx.value, j], y[j])
    end
    return buffer
end

function dual_cons(model::Model, ::Type{ScalarIndex}, y::AbstractVector)
    return model.d_lin - jtprod(model, ScalarIndex, y)
end

function dual_cons!(
    buffer,
    model::Model,
    mat_idx::MatrixIndex,
    y::AbstractVector,
)
    i = mat_idx.value
    return model.C[i] - jtprod!(buffer[i], model, mat_idx, y)
end

NLPModels.grad(model::Model, ::Type{ScalarIndex}) = model.d_lin
NLPModels.grad(model::Model, i::MatrixIndex) = model.C[i.value]

cons_constant(model::Model) = model.b

function NLPModels.cons!(model::Model, x::AbstractVector, cx::AbstractVector)
    NLPModels.jprod!(model, x, x, cx)
    cx .-= model.b
    return cx
end

function add_jprod!(
    model::Model,
    i::MatrixIndex,
    V::AbstractMatrix,
    Jv::AbstractVector,
)
    for j in 1:model.meta.ncon
        Jv[j] += LinearAlgebra.dot(model.A[i.value, j], V)
    end
end

function NLPModels.jprod!(
    model::Model,
    _::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    LinearAlgebra.mul!(Jv, model.C_lin, v[ScalarIndex])
    for i in matrix_indices(model)
        add_jprod!(model, i, v[i], Jv)
    end
    return Jv
end
