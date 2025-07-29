# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

abstract type AbstractFactorization{T,F} <: AbstractMatrix{T} end

function Base.size(m::AbstractFactorization)
    n = size(left_factor(m), 1)
    return (n, n)
end

function Base.getindex(m::AbstractFactorization, i::Int, j::Int)
    left = left_factor(m)
    right = right_factor(m)
    return sum(
        left[i, k] * m.scaling[k] * right[j, k]' for k in eachindex(m.scaling)
    )
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
    D<:Union{AbstractArray{T,0},AbstractVector{T}},
} <: AbstractFactorization{T,F}
    factor::F
    scaling::D
    function Factorization{T,F,S}(
        factor::AbstractMatrix{T},
        scaling::AbstractVector{T},
    ) where {T,F<:AbstractMatrix{T},S<:AbstractVector{T}}
        if length(scaling) != size(factor, 2)
            error(
                "Length `$(length(scaling))` of diagonal does not match number of columns `$(size(factor, 2))` of factor",
            )
        end
        return new{T,F,S}(factor, scaling)
    end
    function Factorization{T,F,S}(
        factor::AbstractVector{T},
        scaling::AbstractArray{T,0},
    ) where {T,F<:AbstractVector{T},S<:AbstractArray{T,0}}
        return new{T,F,S}(factor, scaling)
    end
end

function Factorization(
    factor::AbstractMatrix{T},
    scaling::AbstractVector{T},
) where {T}
    return Factorization{T,typeof(factor),typeof(scaling)}(factor, scaling)
end

function Factorization(
    factor::AbstractVector{T},
    scaling::AbstractArray{T,0},
) where {T}
    return Factorization{T,typeof(factor),typeof(scaling)}(factor, scaling)
end

function Factorization(factor::AbstractVector{T}, scaling::T) where {T}
    return Factorization(factor, fill(scaling, tuple()))
end

left_factor(m::Factorization) = m.factor
right_factor(m::Factorization) = m.factor

function Base.promote_rule(
    ::Type{Factorization{T,M,S1}},
    ::Type{Factorization{T,V,S2}},
) where {T,M<:AbstractMatrix{T},V<:AbstractVector{T},S1,S2}
    return Factorization{T,M,S1}
end

function Base.convert(
    ::Type{Factorization{T,F,V}},
    f::Factorization{S,<:AbstractVector{S},<:AbstractArray{S,0}},
) where {T,S,F<:AbstractMatrix{T},V<:AbstractVector{T}}
    return Factorization{T,F,V}(
        reshape(f.factor, length(f.factor), 1),
        reshape(f.scaling, 1),
    )
end

function Base.convert(
    ::Type{<:Factorization{T,F,V}},
    f::Factorization{S,<:AbstractMatrix{S},<:AbstractVector{S}},
) where {T,S,F<:AbstractVector{T},V<:AbstractArray{T,0}}
    return Factorization{T,F,V}(
        reshape(f.factor, size(f.factor, 1)),
        reshape(f.scaling, tuple()),
    )
end

function MOI.Bridges.Constraint.conversion_cost(
    ::Type{<:Factorization{T,<:AbstractMatrix{T},<:AbstractVector{T}}},
    ::Type{<:Factorization{T,<:AbstractVector{T},<:AbstractArray{T,0}}},
) where {T}
    return 1.0
end

function MOI.Bridges.Constraint.conversion_cost(
    ::Type{<:AbstractMatrix},
    ::Type{<:AbstractMatrix},
)
    return Inf
end

function _add_by_cat(a::Factorization, b::Factorization)
    return Factorization([a.factor b.factor], [a.scaling; b.scaling])
end

# Solvers are recommented to use this constant instead of hardcoding this
# `FillArrays` type so that the solver does not have to explicitly `import`
# `FillArrays` nor explicitly add it to its dependency so that it remains
# a detail that's internal to LowRankOpt that we can easily change later
# The rest of the code of LowRankOpt should also use these two constants
# and not `FillArrays` directly.

import FillArrays

const One{T} = FillArrays.Ones{T,0,Tuple{}}
const Ones{T} = FillArrays.Ones{T,1,Tuple{Base.OneTo{Int}}}

function positive_semidefinite_factorization(
    factor::AbstractVector{T},
) where {T}
    return Factorization(factor, One{T}(tuple()))
end

function positive_semidefinite_factorization(
    factor::AbstractMatrix{T},
) where {T}
    return Factorization(factor, Ones{T}(Base.OneTo(size(factor, 2))))
end

struct AsymmetricFactorization{
    T,
    F<:Union{AbstractVector{T},AbstractMatrix{T}},
    D<:Union{AbstractArray{T,0},AbstractVector{T}},
} <: AbstractFactorization{T,F}
    left::F
    right::F
    scaling::D
    function AsymmetricFactorization{T,F,S}(
        left::AbstractMatrix{T},
        right::AbstractMatrix{T},
        scaling::AbstractVector{T},
    ) where {T,F<:AbstractMatrix{T},S<:AbstractVector{T}}
        if size(left) != size(right)
            error(
                "Size `$(size(left))` of left factor does not match size `$(size(right))` of right factor",
            )
        end
        if length(scaling) != size(left, 2)
            error(
                "Length `$(length(scaling))` of diagonal does not match number of columns `$(size(left, 2))` of factor",
            )
        end
        return new{T,F,S}(left, right, scaling)
    end
    function AsymmetricFactorization{T,F,S}(
        left::AbstractVector{T},
        right::AbstractVector{T},
        scaling::AbstractArray{T,0},
    ) where {T,F<:AbstractVector{T},S<:AbstractArray{T,0}}
        if length(left) != length(right)
            error(
                "Length `$(length(left))` of left factor does not match the length `$(length(right))` of right factor",
            )
        end
        return new{T,F,S}(left, right, scaling)
    end
end

function AsymmetricFactorization(
    left::AbstractMatrix{T},
    right::AbstractMatrix{T},
    scaling::AbstractVector{T},
) where {T}
    return AsymmetricFactorization{T,typeof(left),typeof(scaling)}(
        left,
        right,
        scaling,
    )
end

function AsymmetricFactorization(
    left::AbstractVector{T},
    right::AbstractVector{T},
    scaling::AbstractArray{T,0},
) where {T}
    return AsymmetricFactorization{T,typeof(left),typeof(scaling)}(
        left,
        right,
        scaling,
    )
end

left_factor(m::AsymmetricFactorization) = m.left
right_factor(m::AsymmetricFactorization) = m.right

"""
    symmetrize_factorization(L, R)

Factorization corresponding to the symmetrization `(L * R' + R * L') / 2` of `L * R'`.

## Example

```jldoctest
julia> LowRankOpt.symmetrize_factorization([1, 0], [0, 1])
2×2 LowRankOpt.Factorization{Float64, Matrix{Float64}, Vector{Float64}}:
 0.0  0.5
 0.5  0.0
```
"""
function symmetrize_factorization(L, R; use_krylov = true)
    sym = LinearAlgebra.Symmetric((L * R' + R * L') / 2)
    r = 2size(L, 2)
    if use_krylov
        eigvals, factors = KrylovKit.eigsolve(sym, r)
        factor = reduce(hcat, factors)
    else
        eigvals, factor = LinearAlgebra.eigen(sym)
    end
    σ = sortperm(abs.(eigvals), rev = true)
    keep = σ[1:r]
    return Factorization(factor[:, keep], eigvals[keep])
end

struct TriangleVectorization{T,M<:AbstractMatrix{T}} <: AbstractVector{T}
    matrix::M
end

function Base.convert(
    ::Type{TriangleVectorization{T,M}},
    t::TriangleVectorization,
) where {T,M}
    return TriangleVectorization{T,M}(t.matrix)
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

######## Dot product ########

# We don't want the fallback as it would be terribly slow,
# we prefer an error so that we can as a specialized method for this case
function _dot_error(a, b)
    return error(
        "`dot` is not implemented yet between `$(typeof(a))` and `$(typeof(b))`",
    )
end

function LinearAlgebra.dot(a::AbstractFactorization, b::AbstractFactorization)
    return _dot_error(a, b)
end
function LinearAlgebra.dot(a::AbstractFactorization, b::AbstractMatrix)
    return _dot_error(a, b)
end

_abs2!!(a) = abs2(a)

function _abs2!!(a::AbstractArray)
    for i in eachindex(a)
        a[i] = abs2(a[i])
    end
    return a
end

_lmul_diag!!(::FillArrays.Ones, VtU) = VtU
_rmul_diag!!(VtU, ::FillArrays.Ones) = VtU

function _lmul_diag!!(s::FillArrays.Fill, VtU::Number)
    return s.value * VtU
end

function _lmul_diag!!(s::FillArrays.Fill, VtU)
    return LinearAlgebra.lmul!(s.value, VtU)
end

function _lmul_diag!!(s::AbstractVector, VtU)
    return LinearAlgebra.lmul!(LinearAlgebra.Diagonal(s), VtU)
end

function _rmul_diag!!(VtU::Number, s::FillArrays.Fill)
    return VtU * s.value
end

function _rmul_diag!!(VtU, s::FillArrays.Fill)
    return LinearAlgebra.rmul!(VtU, s.value)
end

function _rmul_diag!!(VtU, s::AbstractVector)
    return LinearAlgebra.rmul!(VtU, LinearAlgebra.Diagonal(s))
end

function LinearAlgebra.dot(a::Factorization, b::Factorization)
    # `⟨UΣU', VΛV'⟩ = ⟨ΣU'VΛ, U'V⟩`
    VtU = a.factor' * b.factor
    VtU = _abs2!!(VtU)
    VtU = _lmul_diag!!(a.scaling, VtU)
    VtU = _rmul_diag!!(VtU, b.scaling)
    return sum(VtU)
end

function LinearAlgebra.dot(a::Factorization, b::AsymmetricFactorization)
    # `⟨XΛX', UΣV'⟩ = ⟨ΛX'V, X'UΣ⟩`
    XtV = right_factor(b)' * a.factor
    XtU = left_factor(b)' * a.factor
    XtV = MA.broadcast!!(*, XtV, XtU)
    XtV = _lmul_diag!!(a.scaling, XtV)
    XtV = _rmul_diag!!(XtV, b.scaling)
    return sum(XtV)
end

function _dot_mat_fact(
    a::AbstractMatrix,
    b::AbstractFactorization{T,<:AbstractVector},
) where {T}
    return b.scaling[] * LinearAlgebra.dot(left_factor(b), a, right_factor(b))
end

function _dot_mat_fact(
    a::AbstractMatrix{T},
    b::AbstractFactorization{U,<:AbstractMatrix},
) where {T,U}
    res = zero(MA.promote_sum_mul(T, U))
    for i in axes(left_factor(b), 2)
        res +=
            b.scaling[i] *
            LinearAlgebra.dot(left_factor(b)[:, i], a, right_factor(b)[:, i])
    end
    return res
end

# Need to redirect to `dot_mat_fact` before adding the two methods to avoid ambiguity
# with `dot(::AbstractFactorization, ::AbstractFactorization)`
function LinearAlgebra.dot(a::AbstractMatrix, b::AbstractFactorization)
    return _dot_mat_fact(a, b)
end

######## Matrix multiplication ########

# `mul!(::Vector, ::SparseVector, ::Number, ::Number, ::Number)` does not have any specialized method
# I cannot add one since it would be type piracy so we define our own `_mul!`
# For this reason, I don't trust default fallbacks if `A` is sparse so I prefer erroring than
# silent performance issues.
function _add_mul!(
    res::AbstractVecOrMat,
    A::SparseArrays.AbstractSparseArray,
    B,
    _,
)
    return error(
        "Missing `_add_mul!` between `$(typeof(res))`, `$(typeof(A))` and `$(typeof(B))`",
    )
end

function _mul!(
    res::AbstractVecOrMat{T},
    A::SparseArrays.AbstractSparseArray,
    B,
    α,
    β,
) where {T}
    if iszero(β)
        # Since `A` is sparse, there may be some entries of `res` that we won't touch so
        # we cannot use `LinearAlgebra.MulAddMul(α, β)` as it won't set them to zero.
        # It's better to just set them to zero now and then use `_add_mul!`.
        fill!(res, zero(T))
    else
        @assert isone(β)
    end
    return _add_mul!(res, A, B, α)
end

function _add_mul!(
    res::AbstractVector,
    x::SparseArrays.SparseVector,
    C::Number,
    α,
)
    @assert axes(res, 1) == axes(x, 1)
    γ = C * α
    for (row, val) in zip(x.nzind, x.nzval)
        res[row] += val * γ
    end
end

function _add_mul!(
    res::AbstractMatrix,
    x::SparseArrays.SparseVector,
    C::LinearAlgebra.AdjOrTrans,
    α,
)
    @assert axes(res, 2) == axes(C, 2)
    for (row, val) in zip(x.nzind, x.nzval)
        γ = val * α
        for j in axes(res, 2)
            res[row, j] += γ * C[j]
        end
    end
end

function _add_mul!(
    res::AbstractVector,
    F::SparseArrays.SparseMatrixCSC,
    C::AbstractVector,
    α,
)
    @assert axes(C, 1) == axes(F, 2)
    @assert axes(res, 1) == axes(F, 1)
    for col in axes(F, 2)
        for i in SparseArrays.nzrange(F, col)
            row = SparseArrays.rowvals(F)[i]
            res[row] += SparseArrays.nonzeros(F)[i] * C[col] * α
        end
    end
end

function _add_mul!(
    res::AbstractMatrix,
    F::SparseArrays.SparseMatrixCSC,
    C::AbstractMatrix,
    α,
)
    @assert axes(res, 2) == axes(C, 2)
    @assert axes(C, 1) == axes(F, 2)
    @assert axes(res, 1) == axes(F, 1)
    for col in axes(F, 2)
        for i in SparseArrays.nzrange(F, col)
            row = SparseArrays.rowvals(F)[i]
            γ = SparseArrays.nonzeros(F)[i] * α
            for j in axes(res, 2)
                res[row, j] += γ * C[col, j]
            end
        end
    end
end

function _mul!(res::AbstractVecOrMat, A::AbstractVecOrMat, B, α, β)
    return LinearAlgebra.mul!(res, A, B, α, β)
end

function _fact_mul!(
    res::AbstractVecOrMat,
    A::AbstractFactorization,
    B::AbstractVecOrMat,
    α::Number,
    β::Number,
)
    # TODO if `scaling` is `FillArrays.Fill`, we could just update `α`
    C = _lmul_diag!!(A.scaling, right_factor(A)' * B)
    lA = left_factor(A)
    return _mul!(res, lA, C, α, β)
end

# We want the same implementation for the two following ones but we can't use
# `AbstractVecOrMat` as it would give ambiguity so we redirect to `_fact_mul!`
function LinearAlgebra.mul!(
    res::AbstractMatrix,
    A::AbstractFactorization,
    B::AbstractMatrix,
    α::Number,
    β::Number,
)
    return _fact_mul!(res, A, B, α, β)
end

function LinearAlgebra.mul!(
    res::AbstractVector,
    A::AbstractFactorization,
    B::AbstractVector,
    α::Number,
    β::Number,
)
    return _fact_mul!(res, A, B, α, β)
end
