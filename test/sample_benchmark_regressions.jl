# Copyright (c) 2026: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.
#
# Self-contained tests for the LowRankOpt changes required by the
# `SumOfSquares/docs/src/tutorials/Getting started/sampling.jl` benchmark
# (`bmlbfgs = Dualization.dual_optimizer(LRO.Optimizer{...BurerMonteiro...})`).

module TestSampleBenchmarkRegressions

using Test
using LinearAlgebra
using SparseArrays
using FillArrays
import MathOptInterface as MOI
import LowRankOpt as LRO

# -----------------------------------------------------------------------------
# 1. `LRO.SetDotProducts` 4-parameter relaxation
#    (`vectors::Vector{V}` → `vectors::Vs<:AbstractVector{V}`)
# -----------------------------------------------------------------------------

function test_set_dot_products_accepts_row_slices()
    # `eachrow(U)::Base.RowSlices` is the canonical non-`Vector` container
    # that the SumOfSquares LowRankBridge now passes — we need the type
    # parameter `Vs` to capture it instead of forcing materialization.
    U = randn(5, 3)
    rows = eachrow(U)
    @test rows isa AbstractVector{<:AbstractVector{Float64}}
    @test !(rows isa Vector)

    psd = MOI.PositiveSemidefiniteConeTriangle(3)
    set = LRO.SetDotProducts{LRO.WITHOUT_SET}(psd, rows)
    @test set isa LRO.SetDotProducts
    # `vectors` field stores the container as-is; no copy.
    @test set.vectors === rows
    @test length(set.vectors) == 5
    @test MOI.dimension(set) == 5
    @test MOI.side_dimension(set) == 3

    # The fourth type parameter `Vs` must be the concrete `RowSlices` type.
    Vs = typeof(set).parameters[4]
    @test Vs <: AbstractVector{<:AbstractVector{Float64}}
    @test Vs === typeof(rows)
end

function test_set_dot_products_vector_backward_compat()
    # The original `Vector{V}` form still constructs and the inferred `Vs`
    # parameter is `Vector{V}`.
    psd = MOI.PositiveSemidefiniteConeTriangle(2)
    V = Vector{Float64}
    vec_v = V[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    set = LRO.SetDotProducts{LRO.WITHOUT_SET}(psd, vec_v)
    @test set.vectors === vec_v
    Vs = typeof(set).parameters[4]
    @test Vs === Vector{V}
end

function test_set_dot_products_equality_and_copy()
    # `==`, `copy` must work uniformly across `Vs` shapes.
    psd = MOI.PositiveSemidefiniteConeTriangle(2)
    U = [1.0 2.0 3.0; 4.0 5.0 6.0]
    s1 = LRO.SetDotProducts{LRO.WITHOUT_SET}(psd, eachrow(U))
    s2 = LRO.SetDotProducts{LRO.WITHOUT_SET}(psd, eachrow(copy(U)))
    @test s1 == s2
    s3 = copy(s1)
    @test s3 == s1
    @test s3 !== s1
end

function test_set_dot_products_dual_set_round_trip()
    # `MOI.dual_set` / `MOI.dual_set_type` must not assume a particular `Vs`.
    psd = MOI.PositiveSemidefiniteConeTriangle(2)
    U = [1.0 0.0 0.0; 0.5 0.5 0.0]
    primal = LRO.SetDotProducts{LRO.WITHOUT_SET}(psd, eachrow(U))
    dual = MOI.dual_set(primal)
    @test dual isa LRO.LinearCombinationInSet
    @test dual.set == primal.set
    @test dual.vectors == primal.vectors
    # `dual_set_type` only needs `{W,S,V}` to round-trip — the `Vs` parameter
    # is consumed without raising.
    rt_type = MOI.dual_set_type(typeof(primal))
    @test rt_type <: LRO.LinearCombinationInSet
end

function test_set_dot_products_convert_collects_into_vector()
    # `convert(SetDotProducts{...Vector{V}}, ...)` must `collect` any
    # `Vs` (e.g. `RowSlices`) into a `Vector{V}` so the canonical bridged
    # form stays available.
    psd = MOI.PositiveSemidefiniteConeTriangle(2)
    U = [1.0 2.0 3.0; 4.0 5.0 6.0]
    src = LRO.SetDotProducts{LRO.WITHOUT_SET}(psd, eachrow(U))
    V = eltype(src.vectors)
    target = LRO.SetDotProducts{LRO.WITHOUT_SET,typeof(psd),V,Vector{V}}
    dst = convert(target, src)
    @test dst isa target
    @test dst.vectors isa Vector{V}
    @test dst.vectors[1] == [1.0, 2.0, 3.0]
    @test dst.vectors[2] == [4.0, 5.0, 6.0]
end

# -----------------------------------------------------------------------------
# 2. `_lmul_diag!! / _rmul_diag!!` methods for `AbstractArray{T,0}` scaling.
#    Exercised through `dot(::Factorization, ::Factorization)` when the
#    scaling is a 0-dim array — the shape produced by the SumOfSquares
#    LowRankBridge's `reshape(T[weights[j]], ())`.
# -----------------------------------------------------------------------------

function test_dot_rank_one_zerodim_scaling_pair()
    # `F1 = s1 * f1 * f1'` and `F2 = s2 * f2 * f2'` with both `s1`, `s2`
    # 0-dim `Array{T,0}`. The dot is `tr(F1 * F2) = s1 * s2 * (f1' f2)²`.
    f1 = [1.0, 2.0, 3.0]
    f2 = [2.0, 0.0, 1.0]
    s1 = reshape([0.5], ())
    s2 = reshape([3.0], ())
    F1 = LRO.Factorization(f1, s1)
    F2 = LRO.Factorization(f2, s2)
    expected = only(s1) * only(s2) * (f1' * f2)^2
    @test dot(F1, F2) ≈ expected
    @test dot(F1, F2) ≈ dot(Matrix(F1), Matrix(F2))
end

function test_dot_rank_one_zerodim_with_lowrank_vector_scaling()
    # 0-dim × vector-scaling: the `_lmul_diag!!(::AbstractArray{<:Any,0}, _)`
    # method must work in mixed combinations.
    f = [1.0, 0.0, 1.0]
    F1 = LRO.Factorization(f, reshape([2.0], ()))
    F = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    F2 = LRO.Factorization(F, [1.0, 0.5])
    @test dot(F1, F2) ≈ dot(Matrix(F1), Matrix(F2))
    # Symmetric: vector scaling × 0-dim scaling.
    @test dot(F2, F1) ≈ dot(Matrix(F2), Matrix(F1))
end

function test_dot_rank_one_zerodim_scalar_VtU()
    # When both factors are vectors, `a.factor' * b.factor` is a scalar (a
    # plain `Number`), not an `AbstractVector`. The 0-dim `_lmul_diag!!` /
    # `_rmul_diag!!` Number methods must accept that.
    f1 = [1.0, 2.0]
    f2 = [3.0, 4.0]
    s1 = reshape([0.25], ())
    s2 = reshape([2.0], ())
    F1 = LRO.Factorization(f1, s1)
    F2 = LRO.Factorization(f2, s2)
    expected = 0.25 * 2.0 * (f1' * f2)^2
    @test dot(F1, F2) ≈ expected
end

# -----------------------------------------------------------------------------
# 3. `concrete_bridge_type` dispatch on the 4-parameter `SetDotProducts`.
#    `AppendSetBridge` / `DotProductsBridge` originally used
#    `Type{LRO.SetDotProducts{W,S,V}}` (concrete UnionAll-of-3-params).
#    After the relaxation the inputs are concrete 4-params, so the
#    signatures need `<:` to dispatch.
# -----------------------------------------------------------------------------

function test_concrete_bridge_type_append_set_bridge()
    T = Float64
    psd = MOI.PositiveSemidefiniteConeTriangle(3)
    V = LRO.TriangleVectorization{T,LRO.Factorization{T,Vector{T},Array{T,0}}}
    Vs = Vector{V}
    SetType = LRO.SetDotProducts{LRO.WITHOUT_SET,typeof(psd),V,Vs}
    bridge_t = MOI.Bridges.Variable.concrete_bridge_type(
        LRO.Bridges.Variable.AppendSetBridge{T},
        SetType,
    )
    @test bridge_t isa Type
    @test bridge_t ===
          LRO.Bridges.Variable.AppendSetBridge{T,typeof(psd),V}
end

function test_concrete_bridge_type_dot_products_bridge()
    T = Float64
    psd = MOI.PositiveSemidefiniteConeTriangle(2)
    V = LRO.TriangleVectorization{T,LRO.Factorization{T,Vector{T},LRO.One{T}}}
    Vs = Vector{V}
    SetType = LRO.SetDotProducts{LRO.WITH_SET,typeof(psd),V,Vs}
    bridge_t = MOI.Bridges.Variable.concrete_bridge_type(
        LRO.Bridges.Variable.DotProductsBridge{T},
        SetType,
    )
    @test bridge_t ===
          LRO.Bridges.Variable.DotProductsBridge{T,typeof(psd),V}
end

function test_concrete_bridge_type_subarray_factor()
    # Same dispatch but with a `SubArray` factor — the shape SumOfSquares'
    # LowRankBridge produces when it calls `view(U, j, :)` on a `Matrix` or
    # `TrigEvalMatrix`. The bridge graph must accept this concrete type.
    T = Float64
    psd = MOI.PositiveSemidefiniteConeTriangle(3)
    U = randn(4, 3)
    F = typeof(view(U, 1, :))
    V = LRO.TriangleVectorization{T,LRO.Factorization{T,F,Array{T,0}}}
    Vs = Vector{V}
    SetType = LRO.SetDotProducts{LRO.WITHOUT_SET,typeof(psd),V,Vs}
    bridge_t = MOI.Bridges.Variable.concrete_bridge_type(
        LRO.Bridges.Variable.AppendSetBridge{T},
        SetType,
    )
    @test bridge_t ===
          LRO.Bridges.Variable.AppendSetBridge{T,typeof(psd),V}
end

# -----------------------------------------------------------------------------
# 4. `_add!(::_MatrixBuilder, row, coef, ::LinearCombinationInSet)` no longer
#    asserts `isone(coef)` — it folds `coef` into the `Factorization` scaling
#    so the dualized SOS chain (which produces non-unit coefficients) is
#    consumable.
# -----------------------------------------------------------------------------

function test_set_dot_products_coef_fold_preserves_value()
    # `LRO.Factorization(F, c * scaling)` is the in-place pattern the
    # `_add!` rewrite uses. Verify the resulting rank-1 matrix matches
    # `c * F.matrix` element-wise.
    f = [1.0, 2.0, 3.0]
    s = reshape([0.5], ())
    F = LRO.Factorization(f, s)
    c = -1.7
    Fc = LRO.Factorization(F.factor, c * F.scaling)
    @test typeof(Fc.scaling) <: AbstractArray{Float64,0}
    @test only(Fc.scaling) ≈ c * only(s)
    @test Matrix(Fc) ≈ c * Matrix(F)
end

# -----------------------------------------------------------------------------
# 5. `ConstraintDual` retrieval workaround.
#    The fix replaces `sol[ScalarIndex][rows]` (which delegates to
#    `MOI.Utilities.VectorLazyMap`'s broken `getindex(::UnitRange)`) with
#    `[scalars[i] for i in rows]`. Scalar `getindex` on the lazy wrapper is
#    correct, so the workaround returns the element-wise mapped values.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 6. `_rank_one_rowview_batch` detector + batched `add_jprod!` /
#    `BurerMonteiro.add_jtprod!` path. When the SOS LowRankBridge produces
#    `Factorization(view(U, j, :), reshape(T[w_j], ()))` for `j = 1:n`, the
#    detector must surface the parent `U` and the weight vector, and the
#    batched evaluators must agree (to floating-point precision) with the
#    per-constraint reference computation. The whole point of this path is
#    a single `mul!(buf, U, X.factor)` instead of `n` separate `dot`s.
# -----------------------------------------------------------------------------

function _rank_one_row_from_matrix(U::AbstractMatrix, w::AbstractVector)
    n = size(U, 1)
    @assert length(w) == n
    return [
        LRO.Factorization(view(U, j, :), reshape([w[j]], ())) for j in 1:n
    ]
end

function test_rank_one_rowview_batch_detects_matching_pattern()
    U = randn(4, 3)
    w = [0.1, -0.5, 1.2, 0.3]
    row = _rank_one_row_from_matrix(U, w)
    out = LRO._rank_one_rowview_batch(row)
    @test out !== nothing
    parent_U, weights = out
    @test parent_U === U
    @test weights == w
end

function test_rank_one_rowview_batch_rejects_materialised_factor()
    # When the bridge materialises rows (the old `collect(view(U, j, :))`
    # path) the factors are independent `Vector{Float64}`s; there is no
    # shared parent to batch on, so the detector must decline.
    U = randn(4, 3)
    w = ones(4)
    row = LRO.Factorization{Float64,Vector{Float64},Array{Float64,0}}[
        LRO.Factorization(collect(view(U, j, :)), reshape([w[j]], ())) for j in 1:4
    ]
    @test LRO._rank_one_rowview_batch(row) === nothing
end

function test_rank_one_rowview_batch_rejects_mismatched_row_index()
    # Detector must reject if the j-th constraint's view doesn't index row `j`.
    U = randn(4, 3)
    w = ones(4)
    row = _rank_one_row_from_matrix(U, w)
    # Swap the first and second entries: their `parentindices` no longer
    # match `(1, :)` and `(2, :)`.
    row[1], row[2] = row[2], row[1]
    @test LRO._rank_one_rowview_batch(row) === nothing
end

function test_rank_one_rowview_batch_rejects_different_parents()
    U1 = randn(3, 3)
    U2 = randn(3, 3)
    row = [
        LRO.Factorization(view(U1, 1, :), reshape([1.0], ())),
        LRO.Factorization(view(U2, 2, :), reshape([1.0], ())),
        LRO.Factorization(view(U1, 3, :), reshape([1.0], ())),
    ]
    @test LRO._rank_one_rowview_batch(row) === nothing
end

function _batched_jprod_ref(row, V::AbstractMatrix)
    return [LinearAlgebra.dot(A, V) for A in row]
end

function test_add_jprod_batched_psd_factorization_matches_reference()
    # `V = factor * factor'` (PSD rank-`r` with `Ones` scaling) — the
    # `BurerMonteiro.cons!` shape.
    rng_U = randn(5, 4)
    w = [0.3, -0.7, 1.1, 0.0, 2.0]
    row = _rank_one_row_from_matrix(rng_U, w)
    factor = randn(4, 2)  # side=4, rank=2
    V = LRO.positive_semidefinite_factorization(factor)
    expected = _batched_jprod_ref(row, V)
    out = LRO._rank_one_rowview_batch(row)
    @test out !== nothing
    U, weights = out
    Jv = zeros(Float64, length(weights))
    @test LRO._add_jprod_batched!(Jv, U, weights, V) === true
    @test Jv ≈ expected
end

function test_add_jprod_batched_asymmetric_factorization_matches_reference()
    # `V = α * (left * right')` (rank-`r` asymmetric with constant `Fill`
    # scaling) — the `BurerMonteiro.jprod!` shape produced by
    # `_OuterProduct(X, V)[i]`.
    rng_U = randn(6, 3)
    w = randn(6)
    row = _rank_one_row_from_matrix(rng_U, w)
    L = randn(3, 2)
    R = randn(3, 2)
    α = 2.0
    V = LRO.AsymmetricFactorization(L, R, FillArrays.Fill(α, 2))
    expected = _batched_jprod_ref(row, V)
    out = LRO._rank_one_rowview_batch(row)
    @test out !== nothing
    U, weights = out
    Jv = zeros(Float64, length(weights))
    @test LRO._add_jprod_batched!(Jv, U, weights, V) === true
    @test Jv ≈ expected
end

function test_add_jprod_batched_unspecialised_V_returns_false()
    # The fallback `_add_jprod_batched!(::Any, ::Any, ::Any, ::AbstractMatrix)`
    # signals "no specialisation" so `add_jprod!` falls back to the
    # per-constraint loop. Cover that contract.
    U = randn(3, 2)
    w = ones(3)
    @test LRO._add_jprod_batched!(
        zeros(3),
        U,
        w,
        Matrix{Float64}(undef, 2, 2),  # a plain `AbstractMatrix`, not a Factorization
    ) === false
end

function test_add_jprod_dispatch_batched_path_matches_unbatched()
    # Build an `LRO.Model` whose single PSD block has rank-1 constraints
    # arranged as row-views of a parent matrix, then check that
    # `add_jprod!` (which now routes through the batched path) agrees with
    # the explicit per-constraint reference.
    T = Float64
    ncon = 5
    side = 4
    U = randn(ncon, side)
    w = randn(ncon)
    row = _rank_one_row_from_matrix(U, w)
    # Wrap in the `Matrix{A}` shape expected by `LRO.Model.A`.
    A = reshape(row, 1, ncon)
    C = LRO.Factorization{T,Vector{T},Array{T,0}}[]
    push!(C, LRO.Factorization(zeros(T, side), reshape([zero(T)], ())))
    b = zeros(T, ncon)
    d_lin = SparseArrays.spzeros(T, 0)
    C_lin = SparseArrays.spzeros(T, ncon, 0)
    msizes = [side]
    model = LRO.Model(C, A, b, d_lin, C_lin, msizes)
    V = LRO.positive_semidefinite_factorization(randn(side, 2))
    Jv = zeros(T, ncon)
    LRO.add_jprod!(model, V, Jv, LRO.MatrixIndex(1))
    expected = _batched_jprod_ref(row, V)
    @test Jv ≈ expected
end

function test_add_jprod_dispatch_unbatched_fallback_still_works()
    # When the constraint matrices DON'T form a row-view batch (e.g. they
    # are independent `Vector{Float64}` factors), the detector returns
    # `nothing` and `add_jprod!` falls back to the per-constraint loop.
    T = Float64
    ncon = 3
    side = 2
    factors = [randn(T, side) for _ in 1:ncon]  # independent Vectors → no batch
    w = randn(T, ncon)
    row = [LRO.Factorization(factors[j], reshape([w[j]], ())) for j in 1:ncon]
    A = reshape(
        convert(
            Vector{LRO.Factorization{T,Vector{T},Array{T,0}}},
            row,
        ),
        1,
        ncon,
    )
    C = LRO.Factorization{T,Vector{T},Array{T,0}}[]
    push!(C, LRO.Factorization(zeros(T, side), reshape([zero(T)], ())))
    b = zeros(T, ncon)
    d_lin = SparseArrays.spzeros(T, 0)
    C_lin = SparseArrays.spzeros(T, ncon, 0)
    msizes = [side]
    model = LRO.Model(C, A, b, d_lin, C_lin, msizes)
    @test LRO._rank_one_rowview_batch(view(model.A, 1, :)) === nothing
    V = LRO.positive_semidefinite_factorization(randn(side, 2))
    Jv = zeros(T, ncon)
    LRO.add_jprod!(model, V, Jv, LRO.MatrixIndex(1))
    expected = [LinearAlgebra.dot(model.A[1, j], V) for j in 1:ncon]
    @test Jv ≈ expected
end

function test_vector_lazy_map_scalar_indexing_pattern()
    base = [1.0, 2.0, 3.0, -2.0]
    lazy = MOI.Utilities.VectorLazyMap{Float64}(abs2, base)
    @test lazy[1] == 1.0
    @test lazy[3] == 9.0
    rows = 2:4
    out = Float64[lazy[i] for i in rows]
    @test out == [4.0, 9.0, 4.0]
end

# -----------------------------------------------------------------------------

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
end

end

TestSampleBenchmarkRegressions.runtests()
