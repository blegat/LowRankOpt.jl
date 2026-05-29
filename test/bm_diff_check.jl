# Copyright (c) 2026: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.
#
# Regression test for the BurerMonteiro.Model derivatives.
#
# Verifies `NLPModels.obj`/`grad`/`cons`/`jprod!`/`jtprod!`/`hprod!` against
# `FiniteDiff` on the small rank-1 LP+PSD instance produced by the SOS
# `_lagrange_jump_model` shape (`p − γ ∈ SOSCone(BoxSampling)`).
#
# Caught a real sign bug: the scalar block of `hprod!` for `Model{true}`
# previously evaluated `−2·v·(obj_weight·d_lin + C_lin'·y)` instead of
# `+2·v·(obj_weight·d_lin − C_lin'·y)`. The bug stayed dormant because the
# only existing inner sub-solver (Percival → LBFGS) never calls `hprod!`.

module TestBurerMonteiroDiff

using Test
using LinearAlgebra
using SparseArrays
using FillArrays
import LowRankOpt as LRO
import NLPModels
import FiniteDiff

# Build a tiny `LRO.Model` whose shape matches what
# `SumOfSquares.Bridges.Variable.LowRankBridge` + Dualization produces for a
# univariate SOS-feasibility problem with `BoxSampling`. Five rank-1
# equality constraints on a single 3×3 PSD block + 2 scalar variables.
function _toy_lro_model(::Type{T} = Float64) where {T}
    side = 3
    rank_factor_columns = 4  # what `"ranks" => [4]` becomes downstream
    ncon = 5
    n_scalars = 2

    # Rank-1 constraint matrices `A[1, j] = u_j * u_j' * w_j` mirroring the
    # `view(U, j, :)` shape the SOS bridge produces.
    U = randn(T, ncon, side)
    weights = ones(T, ncon)
    A = Matrix{LRO.Factorization{T,SubArray{T,1,Matrix{T},Tuple{Int,Base.Slice{Base.OneTo{Int}}},true},Array{T,0}}}(undef, 1, ncon)
    for j in 1:ncon
        A[1, j] = LRO.Factorization(view(U, j, :), reshape(T[weights[j]], ()))
    end

    # Objective: PSD-block `C` zero, linear scalar `d_lin = [1.0, -0.5]`.
    C = LRO.Factorization{T,Vector{T},Array{T,0}}[
        LRO.Factorization(zeros(T, side), reshape([zero(T)], ())),
    ]
    d_lin = SparseArrays.sparsevec(1:n_scalars, T[1.0, -0.5])
    # `C_lin`: how scalar vars enter each constraint.
    C_lin = sparse(T[1.0 0.5; -0.3 0.2; 0.1 0.4; 0.7 -0.1; 0.0 0.6])

    b = randn(T, ncon)
    return LRO.Model(C, A, b, d_lin, C_lin, [side])
end

# Sanity: shape detection. With `square_scalars = true`, every variable is
# free (`lvar = -Inf`).
function _bm_model(::Type{T} = Float64) where {T}
    return LRO.BurerMonteiro.Model{true}(_toy_lro_model(T), [4])
end

# Sample sub-vectors used across the gradient / Jacobian / Hessian checks.
function _sample_vectors(n; seed = 0)
    return [
        ones(n),
        -ones(n),
        2 * ones(n),
        Float64[(i % 3 == 0) ? 1.0 : 0.0 for i in 1:n],
        randn(Random.default_rng(seed), n),
    ]
end

import Random

function test_gradient_matches_finite_diff()
    bm = _bm_model()
    x = randn(bm.meta.nvar)
    f(x) = NLPModels.obj(bm, x)
    gfd = FiniteDiff.finite_difference_gradient(f, x)
    g   = NLPModels.grad(bm, x)
    @test g ≈ gfd rtol = 1e-5 atol = 1e-5
end

function test_jprod_matches_finite_diff()
    bm = _bm_model()
    x = randn(bm.meta.nvar)
    Jfd = FiniteDiff.finite_difference_jacobian(x -> NLPModels.cons(bm, x), x)
    for v in _sample_vectors(bm.meta.nvar)
        @test NLPModels.jprod(bm, x, v) ≈ Jfd * v rtol = 1e-5 atol = 1e-5
    end
end

function test_jtprod_matches_finite_diff()
    bm = _bm_model()
    x = randn(bm.meta.nvar)
    Jfd = FiniteDiff.finite_difference_jacobian(x -> NLPModels.cons(bm, x), x)
    for w in _sample_vectors(bm.meta.ncon)
        @test NLPModels.jtprod(bm, x, w) ≈ Jfd' * w rtol = 1e-5 atol = 1e-5
    end
end

function test_hprod_matches_finite_diff()
    # The bug fix this test pins down: scalar-block Hessian must satisfy
    # `(H · v)_scalar = 2 · v_scalar · (obj_weight · d_lin − C_lin' · y)`,
    # not the prior `−2 · v_scalar · (obj_weight · d_lin + C_lin' · y)`.
    bm = _bm_model()
    x  = randn(bm.meta.nvar)
    y  = randn(bm.meta.ncon)
    obj_weight = 0.7
    L(x) = obj_weight * NLPModels.obj(bm, x) - LinearAlgebra.dot(y, NLPModels.cons(bm, x))
    Hfd = FiniteDiff.finite_difference_hessian(L, x)
    for v in _sample_vectors(bm.meta.nvar)
        @test NLPModels.hprod(bm, x, y, v; obj_weight) ≈ Hfd * v rtol = 1e-5 atol = 1e-5
    end
end

# MadNLP uses the *opposite* Lagrangian sign from NLPModels: it solves with
# `L_MN(x, y) = obj_weight·f(x) + y'·c(x)` (dual feasibility `∇f + Jᵀy = 0`,
# `MadNLP/src/IPM/kernels.jl:247`), whereas NLPModels uses
# `L_NLP(x, y) = obj_weight·f(x) − y'·c(x)`. To get the MadNLP Hessian out of
# NLPModels' `hprod!`, we must negate `y` before passing it in (this is what
# `bm_madnlp_kkt.jl`'s `eval_lag_hess_wrapper!` does via
# `kkt.current_y .= .-l`). The test below locks that sign convention in.
function test_hprod_matches_finite_diff_madnlp_sign()
    bm = _bm_model()
    x  = randn(bm.meta.nvar)
    y  = randn(bm.meta.ncon)
    obj_weight = 0.7
    L_MN(x) = obj_weight * NLPModels.obj(bm, x) + LinearAlgebra.dot(y, NLPModels.cons(bm, x))
    Hfd = FiniteDiff.finite_difference_hessian(L_MN, x)
    for v in _sample_vectors(bm.meta.nvar)
        # Pass `-y` to `NLPModels.hprod` to get the MadNLP-convention Hessian.
        @test NLPModels.hprod(bm, x, -y, v; obj_weight) ≈ Hfd * v rtol = 1e-5 atol = 1e-5
    end
end

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$name" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
end

end

TestBurerMonteiroDiff.runtests()
