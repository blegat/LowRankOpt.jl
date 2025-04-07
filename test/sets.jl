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
end

function test_factorizations()
    f = [1, 2]
    _test_factorization(f * f', LRO.positive_semidefinite_factorization(f))
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

function test_conversion()
    lowrank = LRO.Factorization(reshape([1, 2], 2, 1), [-1])
    rankone = LRO.Factorization([1, 2], fill(-1, tuple()))
    _test_convert(lowrank, rankone)
    _test_convert(lowrank, rankone) do f
        return LRO.TriangleVectorization(f)
    end
    _test_convert(lowrank, rankone) do f
        return LRO.SetDotProducts{LRO.WITH_SET}(
            MOI.PositiveSemidefiniteConeTriangle(2),
            [LRO.TriangleVectorization(f)],
        )
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
