# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestSets

using Test
using LinearAlgebra
import LowRankOpt as LRO
using FillArrays

function _test_dot(A, B)
    exp = dot(Matrix(A), Matrix(B))
    if !(A isa LRO.AsymmetricFactorization)
        @test dot(A, B) ≈ exp
        @test dot(Matrix(A), B) ≈ exp
    end
end

function test_dot()
    x = collect(1:3)
    y = collect(4:6)
    matrices = [
        LRO.positive_semidefinite_factorization(x),
        LRO.Factorization(x, FillArrays.Fill(5, tuple())),
        LRO.positive_semidefinite_factorization([x y]),
        LRO.Factorization([x y], FillArrays.Fill(-3, 2)),
        LRO.Factorization([x y], [-2, 3]),
        LRO.AsymmetricFactorization(x, y, FillArrays.Fill(-3, tuple())),
    ]
    for A in matrices
        for B in matrices
            _test_dot(A, B)
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
