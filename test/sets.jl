# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestSets

using Test
import LinearAlgebra
import MathOptInterface as MOI
import LowRankOpt as LRO

function _test_factorization(A, B)
    @test size(A) == size(B)
    @test A ≈ B
    d = LinearAlgebra.checksquare(A)
    n = div(d * (d + 1), 2)
    vA = LRO.TriangleVectorization(A)
    @test length(vA) == n
    @test eachindex(vA) == Base.OneTo(n)
    vB = LRO.TriangleVectorization(B)
    @test length(vB) == n
    @test eachindex(vA) == Base.OneTo(n)
    k = 0
    for j in 1:d
        for i in 1:j
            k += 1
            @test vA[k] == vB[k]
            @test vA[k] == A[i, j]
        end
    end
    for W in [LRO.WITH_SET, LRO.WITHOUT_SET]
        set = MOI.PositiveSemidefiniteConeTriangle(d)
        primal = LRO.SetDotProducts{W}(set, [vA])
        dual = LRO.LinearCombinationInSet{W}(set, [vA])
        @test MOI.dual_set(primal) == dual
        @test MOI.dual_set_type(typeof(primal)) == typeof(dual)
        @test MOI.dual_set(dual) == primal
        @test MOI.dual_set_type(typeof(dual)) == typeof(primal)
    end
    return
end

function test_inconsistent_length()
    err = ErrorException(
        "Length `1` of diagonal does not match number of columns `2` of factor",
    )
    @test_throws err LRO.Factorization(ones(1, 2), [1.0])
    err = ErrorException(
        "Size `(2, 1)` of left factor does not match size `(2, 2)` of right factor",
    )
    @test_throws err LRO.AsymmetricFactorization(ones(2, 1), ones(2, 2), [1.0])
    err = ErrorException(
        "Length `1` of diagonal does not match number of columns `2` of factor",
    )
    @test_throws err LRO.AsymmetricFactorization(ones(2, 2), ones(2, 2), [1.0])
    err = ErrorException(
        "Length `2` of left factor does not match the length `1` of right factor",
    )
    @test_throws err LRO.AsymmetricFactorization(
        ones(2),
        ones(1),
        ones(tuple()),
    )
end

function test_factorizations()
    f = [1, 2]
    g = [3, 4]
    _test_factorization(f * f', LRO.positive_semidefinite_factorization(f))
    _test_factorization(
        5 * f * g',
        LRO.AsymmetricFactorization(f, g, 5 * ones(Int, tuple())),
    )
    _test_factorization(2 * f * f', LRO.Factorization(f, 2))
    F = [1 2; 3 4; 5 6]
    d = [7, 8]
    _test_factorization(F * F', LRO.positive_semidefinite_factorization(F))
    _test_factorization(
        F * LinearAlgebra.Diagonal(d) * F',
        LRO.Factorization(F, d),
    )
    return
end

function _test_convert(a, b)
    @test MOI.Bridges.Constraint.conversion_cost(typeof(a), typeof(b)) == 1.0
    c = convert(typeof(a), b)
    @test typeof(c) == typeof(a)
    @test c == a
end

_test_convert(f, a, b) = _test_convert(f(a), f(b))

function test_promotion()
    F = ones(2, 2)
    f = ones(2)
    a = LRO.Factorization(F, ones(2))
    b = LRO.Factorization(f, 1.0)
    @test Base.promote_typeof(a, b) == typeof(a)
    @test Base.promote_typeof(b, a) == typeof(a)
    @test eltype([a, b]) == typeof(a)
    @test eltype([b, a]) == typeof(a)
end

function test_conversion()
    F = reshape([1, 2], 2, 1)
    lowrank = LRO.Factorization(F, [1])
    rankone = LRO.Factorization([1, 2], fill(1, tuple()))
    psd_rankone = LRO.positive_semidefinite_factorization([1, 2])
    for (a, b) in [(lowrank, rankone), (lowrank, psd_rankone)]
        _test_convert(a, b)
        _test_convert(a, b) do f
            return LRO.TriangleVectorization(f)
        end
        _test_convert(a, b) do f
            return LRO.SetDotProducts{LRO.WITH_SET}(
                MOI.PositiveSemidefiniteConeTriangle(2),
                [LRO.TriangleVectorization(f)],
            )
        end
    end
end

function test_symmetrize()
    for use_krylov in [false, true]
        f = LRO.symmetrize_factorization([0, 0, 1], [2, 0, 0]; use_krylov)
        σ = sortperm(f.scaling)
        @test f.scaling[σ] ≈ [-1, 1]
        F = f.factor[:, σ]
        if F[1, 1] < 0
            @test F ≈ [-1 1; 0 0; 1 1] / √2
        else
            @test F ≈ [1 1; 0 0; -1 1] / √2
        end
    end
end

function test_set_dot()
    psd = MOI.PositiveSemidefiniteConeTriangle(2)
    primal = LRO.SetDotProducts{LRO.WITH_SET}(psd, [zeros(3), ones(3)])
    dual = MOI.dual_set(primal)
    x = [2, -3, -1, 4, -2]
    y = [3, -1, -2, 1, -3]
    for set in [primal, dual]
        @test MOI.Utilities.set_dot(x, y, set) ==
              LinearAlgebra.dot(x, y) + x[4] * y[4]
        c = copy(x)
        c[4] /= 2
        @test MOI.Utilities.dot_coefficients(x, set) == c
    end
    @test MOI.Utilities.distance_to_set(y, primal) ≈ 5.291502622
    @test MOI.Utilities.distance_to_set(y, dual) ≈ 5
    primal = LRO.SetDotProducts{LRO.WITHOUT_SET}(psd, [zeros(3), ones(3)])
    dual = MOI.dual_set(primal)
    y = y[1:2]
    @test MOI.Utilities.distance_to_set(y, dual) ≈ 2
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
