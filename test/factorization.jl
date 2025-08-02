# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestSets

using Test
using LinearAlgebra
using SparseArrays
import LowRankOpt as LRO
using FillArrays

struct DummyFactorization <: LRO.AbstractFactorization{Float64,Matrix{Float64}} end
struct DummySparse <: SparseArrays.AbstractSparseVector{Float64,Int} end
Base.size(::DummySparse) = (2,)
Base.getindex(::DummySparse, ::Integer) = 1.0

function test_dot_error()
    F = DummyFactorization()
    err = ErrorException(
        "`dot` is not implemented yet between `Main.TestSets.DummyFactorization` and `Main.TestSets.DummyFactorization`",
    )
    @test_throws err dot(F, F)
    err = ErrorException(
        "`dot` is not implemented yet between `Main.TestSets.DummyFactorization` and `Matrix{Float64}`",
    )
    @test_throws err dot(F, ones(2, 2))
    return
end

function test_mul_error()
    F = LRO.positive_semidefinite_factorization(DummySparse())
    err = ErrorException(
        "Missing `_add_mul!` between `Vector{Float64}`, `Main.TestSets.DummySparse` and `Float64`",
    )
    _add = LinearAlgebra.MulAddMul(true, true)
    @test_throws err LRO.buffered_mul!(rand(2), F, rand(2), _add, nothing)
    return
end

function _test_dot(A, B)
    exp = dot(Matrix(A), Matrix(B))
    if !(A isa LRO.AsymmetricFactorization)
        @test dot(A, B) ≈ exp
        @test dot(Matrix(A), B) ≈ exp
    end
end

function _sample_matrices()
    matrices = Any[]
    x = collect(1:3)
    y = collect(4:6)
    for x in [collect(1:3), sparsevec([1, 3], [4, 5])]
        append!(
            matrices,
            [
                LRO.positive_semidefinite_factorization(x),
                LRO.Factorization(x, FillArrays.Fill(5, tuple())),
                LRO.positive_semidefinite_factorization([x y]),
                LRO.Factorization([x y], FillArrays.Fill(-3, 2)),
                LRO.Factorization([x y], [-2, 3]),
                LRO.AsymmetricFactorization(x, y, FillArrays.Fill(-3, tuple())),
            ],
        )
    end
    return matrices
end

function test_dot()
    matrices = _sample_matrices()
    for A in matrices
        for B in matrices
            _test_dot(A, B)
        end
    end
end

function _test_mul(A, B, α, β)
    if B isa AbstractVector
        res = ones(length(B))
    else
        res = ones(size(A, 1), size(B, 2))
    end
    err = ErrorException("This is inefficient, call `buffered_mul!` instead")
    if A isa LRO.AbstractFactorization
        if LRO.left_factor(A) isa AbstractMatrix
            if B isa AbstractVector
                buffer = zeros(LRO.max_rank(A))'
            else
                buffer = zeros(size(B, 2), LRO.max_rank(A))
            end
        else
            if B isa AbstractVector
                buffer = nothing
            else
                buffer = zeros(size(B, 2))
            end
        end
        @test_throws err LinearAlgebra.mul!(res, A, B, α, β)
        _add = LinearAlgebra.MulAddMul(α, β)
        LRO.buffered_mul!(res, A, B, _add, buffer)
    else
        LinearAlgebra.mul!(res, A, B, α, β)
    end
    expected = ones(size(res))
    LinearAlgebra.mul!(expected, Array(A), Array(B), α, β)
    @test res ≈ expected
end

function _test_mul(A, B)
    v = (false, true, 2.0)
    @testset "α=$α" for α in v
        @testset "β=$β" for β in v[1:2]
            _test_mul(A, B, α, β)
        end
    end
end

function test_mul()
    matrices = _sample_matrices()
    @testset "$(nameof(typeof(A)))" for A in matrices
        x = reverse(collect(axes(A, 2)))
        @testset "$(typeof(x))" begin
            _test_mul(A, x)
        end
        X = reshape(reverse(1:2size(A, 2)), size(A, 2), 2)
        @testset "$(typeof(X))" begin
            _test_mul(A, X)
        end
    end
end

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

TestSets.runtests()
