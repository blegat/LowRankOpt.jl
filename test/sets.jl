module TestSets

using Test
import LinearAlgebra
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
    return
end

function test_factorizations()
    f = [1, 2]
    _test_factorization(f * f', LRO.PositiveSemidefiniteFactorization(f))
    _test_factorization(2 * f * f', LRO.Factorization(f, 2))
    F = [1 2; 3 4; 5 6]
    d = [7, 8]
    _test_factorization(F * F', LRO.PositiveSemidefiniteFactorization(F))
    _test_factorization(
        F * LinearAlgebra.Diagonal(d) * F',
        LRO.Factorization(F, d),
    )
    return
end

function test_symmetrize()
    for use_krylov in [false, true]
        f = LRO.symmetrize_factorization([0, 0, 1], [2, 0, 0]; use_krylov)
        σ = sortperm(f.scaling)
        @test f.scaling[σ] ≈ [-1, 1]
        @test f.factor[:, σ] ≈ [-1 1; 0 0; 1 1] / √2
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
