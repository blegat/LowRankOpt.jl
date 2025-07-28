# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestSets

using Test
using LinearAlgebra
import LowRankOpt as LRO
using FillArrays

struct DummyFactorization <: LRO.AbstractFactorization{Float64,Matrix{Float64}} end

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

function _test_dot(A, B)
    exp = dot(Matrix(A), Matrix(B))
    if !(A isa LRO.AsymmetricFactorization)
        @test dot(A, B) ≈ exp
        @test dot(Matrix(A), B) ≈ exp
    end
end

function _sample_matrices()
    x = collect(1:3)
    y = collect(4:6)
    return [
        LRO.positive_semidefinite_factorization(x),
        LRO.Factorization(x, FillArrays.Fill(5, tuple())),
        LRO.positive_semidefinite_factorization([x y]),
        LRO.Factorization([x y], FillArrays.Fill(-3, 2)),
        LRO.Factorization([x y], [-2, 3]),
        LRO.AsymmetricFactorization(x, y, FillArrays.Fill(-3, tuple())),
    ]
end

function test_dot()
    matrices = _sample_matrices()
    for A in matrices
        for B in matrices
            _test_dot(A, B)
        end
    end
end

function _test_mul(A, B)
    res = ones(size(A, 1), size(B, 2))
    LinearAlgebra.mul!(res, A, B)
    expected = ones(size(res))
    LinearAlgebra.mul!(expected, Array(A), Array(B))
    @test res ≈ expected
end

function test_mul()
    matrices = _sample_matrices()
    for A in matrices
        x = reverse(collect(axes(A, 2)))
        _test_mul(A, x)
        X = reshape(reverse(1:2size(A, 2)), size(A, 2), 2)
        _test_mul(A, X)
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
