# Copyright (c) 2026: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# Scope: the Schur complement `H[j, k] = ∑ᵢ ⟨Aᵢⱼ, Lᵢ Aᵢₖ Rᵢ⟩` with two
# different scalings `L` and `R`, as for the HKM direction, for every
# component kind, including one-hot components.

module TestHKM

using Test, LinearAlgebra, SparseArrays, Random
import FillArrays
import LowRankOpt as LRO

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

_spd(rng, d) = (R = randn(rng, d, d); R * R' / d + I)

function _block_columns(ids, entries, nblocks)
    return LRO.BlockColumns(ids, reshape(entries, :, length(ids)), nblocks)
end

# Checks `schur_complement!` and `eval_schur_complement!` against
# `tr(Aⱼ L Aₖ R)` computed from the dense matrices.
function _test_schur(model; κ = 1, nscalars = 0)
    rng = MersenneTwister(1)
    T = eltype(model.b)
    m = model.meta.ncon
    buffer = LRO.BufferedModelForSchur(model, κ)
    idx = LRO.matrix_indices(model)
    L = LRO.ShapedSolution(
        rand(rng, T, nscalars),
        [_spd(rng, d) for d in model.msizes],
    )
    R = LRO.ShapedSolution(
        rand(rng, T, nscalars),
        [_spd(rng, d) for d in model.msizes],
    )
    full = [Matrix(LRO.jac(model, j, i)) for i in idx, j in 1:m]
    C = Matrix(model.C_lin)
    expected = [
        sum(
            tr(full[i.value, j] * L[i] * full[i.value, k] * R[i]) for i in idx;
            init = zero(T),
        ) + sum(
            C[j, l] * L[LRO.ScalarIndex][l] * R[LRO.ScalarIndex][l] * C[k, l] for l in 1:nscalars;
            init = zero(T),
        ) for j in 1:m, k in 1:m
    ]
    H = zeros(T, m, m)
    LRO.schur_complement!(buffer, L, R, H)
    @test H ≈ expected
    y = randn(rng, T, m)
    Hy = zeros(T, m)
    LRO.eval_schur_complement!(buffer, L, R, y, Hy)
    @test Hy ≈ expected * y
    # The NT methods are the special case `L = R = W` for the PSD blocks.
    if iszero(nscalars)
        W = L
        LRO.schur_complement!(buffer, W, H)
        H_two_sided = similar(H)
        LRO.schur_complement!(buffer, W, W, H_two_sided)
        @test H ≈ H_two_sided
    end
    if T == Float64 && VERSION >= v"1.11"
        LRO.schur_complement!(buffer, L, R, H)
        @test 0 == @allocated LRO.schur_complement!(buffer, L, R, H)
        @test 0 == @allocated LRO.eval_schur_complement!(buffer, L, R, y, Hy)
    end
    return buffer
end

function _model(C, A, b; scalars = 0)
    T = eltype(b)
    m = length(b)
    d_lin = SparseVector{T,Int64}(
        sparsevec(collect(1:scalars), ones(T, scalars), scalars),
    )
    C_lin = sparse(
        T[iszero(scalars) ? 0 : (j + l) % 2 for j in 1:m, l in 1:scalars],
    )
    return LRO.Model(
        C,
        A,
        b,
        d_lin,
        SparseMatrixCSC{T,Int64}(C_lin),
        Int64[size(c, 1) for c in C],
    )
end

# Sparse matrices with one, a few and many nonzeros so that `κ` selects
# the three regimes of the sparse component.
function test_sparse_regimes()
    rng = MersenneTwister(2)
    d, m = 6, 5
    entries = [
        sparse([2], [2], [1.5], d, d),
        sparse(
            Symmetric(sprandn(rng, d, d, 0.15) + sparse([1], [2], [1.0], d, d)),
        ),
        sparse(Symmetric(randn(rng, d, d))),
        sparse([4], [4], [-2.0], d, d),
        spzeros(d, d),
    ]
    model =
        _model([spzeros(d, d)], reshape(entries, 1, m), zeros(m); scalars = 2)
    for κ in (0, 3, 100)
        buffer = _test_schur(model; κ, nscalars = 2)
        @test only(buffer.schur_buffer[3]) isa LRO._GeneralSchurComponent
    end
end

# The max-cut constraints `Xⱼⱼ = 1` are one-hot with the identity index.
function test_one_hot_identity()
    d = 5
    entries = [sparse([j], [j], [1.0], d, d) for j in 1:d]
    model = _model([spzeros(d, d)], reshape(entries, 1, d), ones(d))
    buffer = _test_schur(model)
    component = only(buffer.schur_buffer[3])
    @test component isa LRO._RankOneSchurComponent
    @test only(component.U) == LRO._Selection(Base.OneTo(d))
    # One-hot factors do not need the dense rank-one Jacobian.
    @test only(buffer.jprod_buffer) isa SparseMatrixCSC
end

# More constraints than the side dimension so the index repeats.
function test_one_hot_vector_index()
    d, m = 3, 7
    entries =
        Any[sparse([mod1(j, d)], [mod1(j, d)], [Float64(j)], d, d) for j in 1:m]
    entries[2] = spzeros(d, d)
    entries = SparseMatrixCSC{Float64,Int}[entries...]
    model =
        _model([spzeros(d, d)], reshape(entries, 1, m), zeros(m); scalars = 1)
    buffer = _test_schur(model; nscalars = 1)
    component = only(buffer.schur_buffer[3])
    @test only(component.U).index isa Vector{Int}
end

# One-hot rank-one factorizations as returned by `detect_rank_one`.
function test_one_hot_factorization()
    d = 4
    entries = [sparse([j], [j], [2.0], d, d) for j in 1:d]
    # A zero matrix is detected as a factorization with an empty factor.
    entries[3] = spzeros(d, d)
    components = LRO.detect_rank_one(reshape(entries, 1, d))
    model = _model([spzeros(d, d)], components, ones(d))
    buffer = _test_schur(model)
    @test only(buffer.schur_buffer[3]).U[1] isa LRO._Selection
end

# Every pair of component kinds on the same PSD blocks: one-hot, dense
# rank-one, sparse, diagonal and low-rank.
function test_component_pairs()
    rng = MersenneTwister(3)
    dims, m = [3, 4], 4
    # `Int32` indices, as `sparsevec` gives on 32-bit systems.
    one_hot = _block_columns(
        [1, 2],
        [
            LRO.Factorization(
                sparsevec(Int32[mod1(j, d)], [1.0], d),
                Float64(j),
            ) for j in 1:m, d in dims
        ],
        2,
    )
    rank_one = _block_columns(
        [1, 2],
        [LRO.Factorization(randn(rng, d), Float64(-j)) for j in 1:m, d in dims],
        2,
    )
    sparse_component = _block_columns(
        [1],
        [sparse(Symmetric(sprandn(rng, 3, 3, 0.5))) for j in 1:m],
        2,
    )
    diagonal = _block_columns([2], [Diagonal(randn(rng, 4)) for j in 1:m], 2)
    low_rank = _block_columns(
        [2],
        [LRO.Factorization(randn(rng, 4, 2), randn(rng, 2)) for j in 1:m],
        2,
    )
    components = (one_hot, rank_one, sparse_component, diagonal, low_rank)
    model =
        _model([spzeros(d, d) for d in dims], components, zeros(m); scalars = 2)
    buffer = _test_schur(model; nscalars = 2)
    @test buffer.schur_buffer[3][1].U[1] isa LRO._Selection
    @test buffer.schur_buffer[3][2].U[1] isa Matrix
    # A block with only one-hot and rank-one components uses the
    # `_add_component_matvec!` path of `eval_schur_complement!`.
    model = _model([spzeros(d, d) for d in dims], (one_hot, rank_one), zeros(m))
    return _test_schur(model)
end

end

TestHKM.runtests()
