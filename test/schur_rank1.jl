# Copyright (c) 2026: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.
#
# Verifies the rank-1 Schur-complement specialization in src/schur.jl:
# when constraint matrices are rank-1 `LRO.Factorization`s, the block
# contribution H[j,k] += d_j d_k (b_j' W b_k)² should match a brute-force
# evaluation of ⟨A_{i,j}, W A_{i,k} W⟩.

module TestSchurRank1

using Test
using LinearAlgebra
using SparseArrays
import LowRankOpt as LRO
import Random

# Build a tiny `LRO.Model` whose constraint matrices share the
# `view(U_i, j, :)` shape produced by the SOS bridge / sample-from-parent
# rank-1 batch (`LRO._rank_one_rowview_batch`).
function _rank1_model(::Type{T}, sizes::Vector{Int}, n::Int; seed = 0) where {T}
    rng = Random.MersenneTwister(seed)
    p = length(sizes)
    Us = [randn(rng, T, n, m) for m in sizes]
    weights = [randn(rng, T, n) for _ in 1:p]
    AT = LRO.Factorization{
        T,
        SubArray{T,1,Matrix{T},Tuple{Int,Base.Slice{Base.OneTo{Int}}},true},
        Array{T,0},
    }
    A = Matrix{AT}(undef, p, n)
    for i in 1:p, j in 1:n
        A[i, j] =
            LRO.Factorization(view(Us[i], j, :), reshape(T[weights[i][j]], ()))
    end
    CT = LRO.Factorization{T,Vector{T},Array{T,0}}
    C = CT[
        LRO.Factorization(zeros(T, m), reshape(T[zero(T)], ())) for m in sizes
    ]
    b = randn(rng, T, n)
    d_lin = sparsevec(Int[], T[], 0)
    C_lin = sparse(zeros(T, n, 0))
    return LRO.Model(C, A, b, d_lin, C_lin, sizes), Us, weights
end

# Brute-force: H[j,k] = Σ_i ⟨A_{i,j}, W_i A_{i,k} W_i⟩, no rank-1 trick.
function _brute_force_H(model, W_blocks::Vector{<:AbstractMatrix})
    n = model.meta.ncon
    T = eltype(model.b)
    H = zeros(T, n, n)
    for idx in LRO.matrix_indices(model)
        i = idx.value
        Wi = W_blocks[i]
        for j in 1:n, k in 1:n
            Aj = Matrix(model.A[i, j])
            Ak = Matrix(model.A[i, k])
            H[j, k] += tr(Aj * Wi * Ak * Wi)
        end
    end
    return H
end

function _random_sym_blocks(::Type{T}, sizes; seed = 1) where {T}
    rng = Random.MersenneTwister(seed)
    return map(sizes) do m
        Wi = randn(rng, T, m, m)
        return (Wi + Wi') / 2
    end
end

# `LRO.add_schur_complement!` for one block matches the brute-force
# Schur contribution.
function test_single_block()
    T = Float64
    sizes = [3]
    n = 4
    model, _, _ = _rank1_model(T, sizes, n)
    buf = LRO.BufferedModelForSchur(model, 1)
    W = _random_sym_blocks(T, sizes)
    H = zeros(T, n, n)
    LRO.add_schur_complement!(buf, LRO.MatrixIndex(1), W[1], H)
    @test H ≈ _brute_force_H(model, W)
end

# Multi-block: each block contributes its own rank-1 trick; summed via
# per-block `add_schur_complement!` calls.
function test_multiple_blocks()
    T = Float64
    sizes = [3, 2, 4]
    n = 5
    model, _, _ = _rank1_model(T, sizes, n)
    buf = LRO.BufferedModelForSchur(model, 1)
    W = _random_sym_blocks(T, sizes)
    H = zeros(T, n, n)
    for idx in LRO.matrix_indices(model)
        LRO.add_schur_complement!(buf, idx, W[idx.value], H)
    end
    @test H ≈ _brute_force_H(model, W)
end

# Single constraint sanity (n = 1): H is the 1×1 matrix
# d² (b' W b)² for that constraint.
function test_single_constraint()
    T = Float64
    sizes = [3]
    n = 1
    model, Us, ws = _rank1_model(T, sizes, n)
    buf = LRO.BufferedModelForSchur(model, 1)
    W = _random_sym_blocks(T, sizes)
    H = zeros(T, n, n)
    LRO.add_schur_complement!(buf, LRO.MatrixIndex(1), W[1], H)
    b = Us[1][1, :]
    d = ws[1][1]
    @test H[1, 1] ≈ d^2 * (b' * W[1] * b)^2
end

# Negative scalings should propagate through `d_j d_k` (the squaring is
# only of `b_j' W b_k`).
function test_negative_scalings()
    T = Float64
    sizes = [2]
    n = 2
    model, Us, ws = _rank1_model(T, sizes, n)
    ws[1] .= [-1.5, 2.3]
    AT = LRO.Factorization{
        T,
        SubArray{T,1,Matrix{T},Tuple{Int,Base.Slice{Base.OneTo{Int}}},true},
        Array{T,0},
    }
    A = Matrix{AT}(undef, 1, n)
    for j in 1:n
        A[1, j] = LRO.Factorization(view(Us[1], j, :), reshape(T[ws[1][j]], ()))
    end
    CT = LRO.Factorization{T,Vector{T},Array{T,0}}
    C = CT[LRO.Factorization(zeros(T, 2), reshape(T[zero(T)], ()))]
    model2 = LRO.Model(C, A, model.b, model.d_lin, model.C_lin, model.msizes)
    buf = LRO.BufferedModelForSchur(model2, 1)
    W = _random_sym_blocks(T, sizes; seed = 7)
    H = zeros(T, n, n)
    LRO.add_schur_complement!(buf, LRO.MatrixIndex(1), W[1], H)
    @test H ≈ _brute_force_H(model2, W)
    # off-diagonal sign comes from d_1 * d_2 = (-1.5) * 2.3 < 0.
    @test sign(H[1, 2]) == sign(ws[1][1] * ws[1][2])
end

# Shim supporting `W[MatrixIndex(i)]` and `W[LRO.ScalarIndex]` indexing
# the way `LRO.VectorizedSolution` does on the solver side.
struct _WShim{T}
    blocks::Vector{Matrix{T}}
    scalars::Vector{T}
end
Base.getindex(W::_WShim, i::LRO.MatrixIndex) = W.blocks[i.value]
Base.getindex(W::_WShim, ::Type{LRO.ScalarIndex}) = W.scalars

# `eval_schur_complement!` (rank-1 specialization) should agree with
# the explicit matrix-vector product `H * y`.
function test_eval_matches_matvec()
    T = Float64
    sizes = [3, 2]
    n = 4
    model, _, _ = _rank1_model(T, sizes, n)
    buf = LRO.BufferedModelForSchur(model, 1)
    W_blocks = _random_sym_blocks(T, sizes)
    W = _WShim(W_blocks, T[])
    rng = Random.MersenneTwister(11)
    y = randn(rng, T, n)
    H = zeros(T, n, n)
    LRO.add_schur_complement!(buf, W, LRO.MatrixIndex, H)
    Hy_ref = H * y
    Hy = zeros(T, n)
    LRO.eval_schur_complement!(buf, W, y, Hy)
    @test Hy ≈ Hy_ref
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

TestSchurRank1.runtests()
