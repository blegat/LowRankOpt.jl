# Copyright (c) 2026: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestComponents

using Test, LinearAlgebra, SparseArrays, Random
import LowRankOpt as LRO
import NLPModels
import FillArrays

function _rank_component(rng, ::Type{T}, ids, dims, n) where {T}
    values = [
        LRO.Factorization(randn(rng, T, dims[i]), T((-1)^j * (j + i) / 5))
        for j in 1:n, i in ids
    ]
    return LRO.BlockColumns(ids, values, length(dims))
end

function _fixture(::Type{T}, kind) where {T}
    rng = MersenneTwister(42)
    dims, n = Int64[2, 3, 1], 3
    r1 = _rank_component(rng, T, [1, 2], dims, n)
    r2 = _rank_component(rng, T, [1, 3], dims, n)
    sparse = LRO.BlockColumns(
        [1, 3],
        [spdiagm(T.(1:dims[i]) .+ T(j)) for j in 1:n, i in (1, 3)],
        3,
    )
    diagonal = LRO.BlockColumns(
        [2],
        [Diagonal(T.(1:3) .* T(j / 4)) for j in 1:n, _ in 1:1],
        3,
    )
    lowrank = LRO.BlockColumns(
        [2],
        [
            LRO.Factorization(randn(rng, T, 3, 2), T[0.75, -0.5]) for
            j in 1:n, _ in 1:1
        ],
        3,
    )
    components =
        kind == :rank_one ? (r1, r2) : (sparse, r1, r2, diagonal, lowrank)
    # Build the reference directly from stored entries, independently of jac.
    full = [zeros(T, d, d) for d in dims, _ in 1:n]
    for component in components,
        (k, i) in enumerate(component.block_ids),
        j in 1:n

        full[i, j] .+= component.values[j, k]
    end
    C = [spdiagm(fill(T(i), d)) for (i, d) in enumerate(dims)]
    b = T[1, -2, 3]
    scalars = sparsevec(Int64[], T[], 0)
    scalar_jacobian = spzeros(T, Int64, n, 0)
    model = LRO.Model(C, components, b, scalars, scalar_jacobian, dims)
    reference = LRO.Model(C, full, b, scalars, scalar_jacobian, dims)
    W = LRO.ShapedSolution(T[], [
        let R = randn(rng, T, d, d)
            R * R' + I
        end for d in dims
    ])
    x = LRO.ShapedSolution(
        T[],
        [Matrix(Symmetric(randn(rng, T, d, d))) for d in dims],
    )
    y = randn(rng, T, n)
    return model, reference, full, W, x, y
end

function _iteration!(buffer, W, x, y, H, Hy, Jv, Jty)
    fill!(H, zero(eltype(H)))
    LRO.add_schur_complement!(buffer, W, LRO.MatrixIndex, H)
    LRO.eval_schur_complement!(buffer, W, y, Hy)
    NLPModels.jprod!(buffer, x, x, Jv)
    NLPModels.jtprod!(buffer, x, y, Jty)
    return nothing
end
function _allocations(args...)
    _iteration!(args...)
    return @allocated _iteration!(args...)
end

@testset "Component sums and Schur cross terms" begin
    for T in (Float32, Float64), kind in (:rank_one, :mixed), κ in (0, 1, 100)
        model, reference, full, W, x, y = _fixture(T, kind)
        buffer = LRO.BufferedModelForSchur(model, κ)
        reference_buffer = LRO.BufferedModelForSchur(reference, κ)
        H = zeros(T, 3, 3)
        Hy = zeros(T, 3)
        Jv = zeros(T, 3)
        Jty = LRO.ShapedSolution(T[], similar.(x.matrices))
        for i in LRO.matrix_indices(model), j in 1:3
            @test LRO.jac(model, j, i) ≈ full[i.value, j]
        end
        for i in LRO.matrix_indices(model)
            @test LRO.norm_jac(model, i) ≈
                  sqrt(sum(sum(abs2, full[i.value, j]) for j in 1:3))
            @test LRO.unsafe_jtprod(buffer, y, i) ≈
                  sum(y[j] * full[i.value, j] for j in 1:3)
            @test LRO.unsafe_jtprod(buffer, sparsevec(y), i) ≈
                  sum(y[j] * full[i.value, j] for j in 1:3)
        end
        expected_H = [
            sum(
                dot(full[i.value, j], W[i] * full[i.value, k] * W[i]) for
                i in LRO.matrix_indices(model)
            ) for j in 1:3, k in 1:3
        ]
        _iteration!(buffer, W, x, y, H, Hy, Jv, Jty)
        @test H ≈ expected_H
        @test Hy ≈ expected_H * y
        expected_Jv = [
            sum(dot(full[i.value, j], x[i]) for i in LRO.matrix_indices(model)) for j in 1:3
        ]
        @test Jv ≈ expected_Jv
        unbuffered_Jv = similar(Jv)
        NLPModels.jprod!(model, x, x, unbuffered_Jv)
        @test unbuffered_Jv ≈ Jv
        for i in LRO.matrix_indices(model)
            @test Jty[i] ≈ sum(y[j] * full[i.value, j] for j in 1:3)
        end
        reference_H = similar(H)
        fill!(reference_H, 0)
        LRO.add_schur_complement!(
            reference_buffer,
            W,
            LRO.MatrixIndex,
            reference_H,
        )
        @test H ≈ reference_H
        @test _allocations(buffer, W, x, y, H, Hy, Jv, Jty) == 0
    end
end

@testset "Missing component blocks" begin
    T, n = Float64, 2
    component = LRO.BlockColumns(
        [2],
        reshape(
            [LRO.Factorization([1.0], 2.0), LRO.Factorization([3.0], -1.0)],
            n,
            1,
        ),
        3,
    )
    model = LRO.Model(
        [spzeros(d, d) for d in (2, 1, 3)],
        (component,),
        zeros(n),
        sparsevec(Int64[], T[], 0),
        spzeros(T, Int64, n, 0),
        Int64[2, 1, 3],
    )
    @test size(component) == (2, 3)
    @test isconcretetype(eltype(component.values))
    @test iszero(LRO.jac(model, 1, LRO.MatrixIndex(1)))
    @test size(LRO.jac(model, 1, LRO.MatrixIndex(1))) == (2, 2)
    @test iszero(LRO.jac(model, 2, LRO.MatrixIndex(3)))
    @test LRO.jac(model, 1, LRO.MatrixIndex(2)) == fill(2.0, 1, 1)
    @test_throws ArgumentError LRO.BlockColumns(
        [2, 1],
        repeat(component.values, 1, 2),
        3,
    )
    @test_throws DimensionMismatch LRO.BlockColumns([1, 2], component.values, 3)
    @test_throws DimensionMismatch LRO.Model(
        model.C,
        (LRO.BlockColumns([1], component.values, 2),),
        model.b,
        model.d_lin,
        model.C_lin,
        model.msizes,
    )
end

@testset "Legacy matrix with Zeros entries" begin
    T = Float64
    rng = MersenneTwister(7)
    dims, n = Int64[3, 2, 2], 3
    Z(d) = FillArrays.Zeros{T}(d, d)
    R(d) = LRO.Factorization(randn(rng, T, d), T(0.5))
    S(d) = sparse(Symmetric(sprandn(rng, T, d, d, 0.7) + I))
    # Block 1 mixes all three types, block 2 has no rank-one entry and
    # block 3 has no sparse entry.
    A = Matrix{Union{typeof(Z(1)),typeof(R(1)),typeof(S(1))}}(undef, 3, n)
    A[1, :] = [R(3), S(3), Z(3)]
    A[2, :] = [Z(2), S(2), S(2)]
    A[3, :] = [R(2), Z(2), R(2)]
    full = Matrix.(A)
    C = [spdiagm(fill(T(i), d)) for (i, d) in enumerate(dims)]
    b = T[1, -2, 3]
    scalars = sparsevec(Int64[], T[], 0)
    scalar_jacobian = spzeros(T, Int64, n, 0)
    model = LRO.Model(C, A, b, scalars, scalar_jacobian, dims)
    rank_one, sparse_component = model.A
    @test rank_one.block_ids == [1, 3]
    @test sparse_component.block_ids == [1, 2]
    @test eltype(rank_one.values) == Union{typeof(R(1)),typeof(Z(1))}
    @test eltype(sparse_component.values) == Union{typeof(S(1)),typeof(Z(1))}
    W = LRO.ShapedSolution(T[], [
        let R = randn(rng, T, d, d)
            R * R' + I
        end for d in dims
    ])
    x = LRO.ShapedSolution(
        T[],
        [Matrix(Symmetric(randn(rng, T, d, d))) for d in dims],
    )
    y = randn(rng, T, n)
    buffer = LRO.BufferedModelForSchur(model, 1)
    H = zeros(T, n, n)
    Hy = zeros(T, n)
    Jv = zeros(T, n)
    Jty = LRO.ShapedSolution(T[], similar.(x.matrices))
    _iteration!(buffer, W, x, y, H, Hy, Jv, Jty)
    expected_H = [
        sum(
            dot(full[i.value, j], W[i] * full[i.value, k] * W[i]) for
            i in LRO.matrix_indices(model)
        ) for j in 1:n, k in 1:n
    ]
    @test H ≈ expected_H
    @test Hy ≈ expected_H * y
    @test Jv ≈ [
        sum(dot(full[i.value, j], x[i]) for i in LRO.matrix_indices(model)) for
        j in 1:n
    ]
    @test NLPModels.jprod!(model, x, x, similar(Jv)) ≈ Jv
    for i in LRO.matrix_indices(model)
        @test Jty[i] ≈ sum(y[j] * full[i.value, j] for j in 1:n)
        for j in 1:n
            @test LRO.jac(model, j, i) ≈ full[i.value, j]
        end
    end
    @test _allocations(buffer, W, x, y, H, Hy, Jv, Jty) == 0
end

@testset "Vector factors outside the rank-one kernel" begin
    T = Float64
    rng = MersenneTwister(11)
    d, n = 3, 3
    # A symmetric `AsymmetricFactorization` with vector factors is not a
    # `Factorization` so it uses the general Schur kernel, also when paired
    # with a rank-one component.
    asym = LRO.BlockColumns(
        [1],
        reshape(
            [
                let u = randn(rng, T, d)
                    LRO.AsymmetricFactorization(u, u, fill(T(j)))
                end for j in 1:n
            ],
            n,
            1,
        ),
        1,
    )
    @test size(asym, 1) == n
    @test size(asym, 2) == 1
    rank_one = LRO.BlockColumns(
        [1],
        reshape(
            [LRO.Factorization(randn(rng, T, d), T(-j)) for j in 1:n],
            n,
            1,
        ),
        1,
    )
    # Tuple entries can also be in the legacy block-by-constraint layout.
    legacy = reshape([sparse(T(j) * I, d, d) for j in 1:n], 1, n)
    full = [
        Matrix(asym.values[j, 1]) +
        Matrix(rank_one.values[j, 1]) +
        Matrix(legacy[1, j]) for _ in 1:1, j in 1:n
    ]
    model = LRO.Model(
        [spzeros(T, d, d)],
        (asym, rank_one, legacy),
        zeros(T, n),
        sparsevec(Int64[], T[], 0),
        spzeros(T, Int64, n, 0),
        Int64[d],
    )
    @test model.A[3] == LRO.BlockColumns(legacy)
    buffer = LRO.BufferedModelForSchur(model, 0)
    W = let R = randn(rng, T, d, d)
        R * R' + I
    end
    y = randn(rng, T, n)
    i = LRO.MatrixIndex(1)
    H = zeros(T, n, n)
    LRO.add_schur_complement!(buffer, i, W, H)
    expected_H = [dot(full[j], W * full[k] * W) for j in 1:n, k in 1:n]
    @test H ≈ expected_H
    Hy = zeros(T, n)
    LRO.eval_schur_complement!(buffer, LRO.ShapedSolution(T[], [W]), y, Hy)
    @test Hy ≈ expected_H * y
    @test LRO.unsafe_jtprod(buffer, y, i) ≈ sum(y[j] * full[j] for j in 1:n)
end

@testset "Burer-Monteiro with sparse plus low-rank components" begin
    model, reference, _, _, _, y = _fixture(Float64, :mixed)
    ranks = Int64[1, 2, 1]
    mixed = LRO.BurerMonteiro.Model{false}(model, ranks)
    dense = LRO.BurerMonteiro.Model{false}(reference, ranks)
    rng = MersenneTwister(3)
    x, v = randn(rng, mixed.meta.nvar), randn(rng, mixed.meta.nvar)
    @test NLPModels.obj(mixed, x) ≈ NLPModels.obj(dense, x)
    @test NLPModels.cons(mixed, x) ≈ NLPModels.cons(dense, x)
    @test NLPModels.grad(mixed, x) ≈ NLPModels.grad(dense, x)
    @test NLPModels.jprod(mixed, x, v) ≈ NLPModels.jprod(dense, x, v)
    @test NLPModels.jtprod(mixed, x, y) ≈ NLPModels.jtprod(dense, x, y)
    @test NLPModels.hprod(mixed, x, y, v) ≈ NLPModels.hprod(dense, x, y, v)
end

end
