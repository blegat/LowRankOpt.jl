module TestEigenWorkspace

using Test, LinearAlgebra
# Register generic symmetric eigensolvers for BigFloat.
import GenericLinearAlgebra
import LowRankOpt as LRO

function check_workspace(::Type{T}) where {T}
    A = T[2 1 0; 1 2 0; 0 0 -1]
    expected = T[-1, 1, 3]
    if T == BigFloat
        # Detect accidental Float64 conversion in the generic fallback.
        perturbation = ldexp(one(T), -100)
        A[3, 3] += perturbation
        expected[1] += perturbation
    end
    scratch = similar(A)
    workspace = LRO.EigenWorkspace(scratch)
    @test workspace.scratch === scratch
    @test isnothing(workspace.lapack) == (T == BigFloat)

    for shift in (zero(T), T(4))
        input = A + shift * I
        original = copy(input)
        spectrum = expected .+ shift
        values, vectors =
            LRO.eigen!(workspace, copyto!(scratch, input); vectors = true)
        @test eltype(values) == T
        @test eltype(vectors) == T
        @test values ≈ spectrum
        @test input * vectors ≈ vectors * Diagonal(values)
        @test vectors' * vectors ≈ Matrix{T}(I, 3, 3)

        values, vectors = LRO.eigen!(workspace, copyto!(scratch, input))
        @test eltype(values) == T
        @test values ≈ spectrum
        if T == BigFloat
            @test vectors === nothing
        end
        @test LRO.eigmin!(workspace, input) ≈ first(spectrum)
        @test input == original
        @test workspace.scratch === scratch
    end
    return
end

function eigenvectors_allocations(workspace, A)
    return @allocated LRO.eigen!(workspace, A; vectors = true)
end

# Above 25, `syevd` uses divide and conquer, which checks the size of the work
# arrays.
function check_syevd(::Type{T}) where {T}
    n = 40
    B = randn(T, n, n)
    input = B + B'
    workspace = LRO.EigenWorkspace(zeros(T, n, n))
    for _ in 1:2
        A = copy(input)
        values, vectors = LRO.eigen!(workspace, A; vectors = true)
        @test vectors === A
        @test values ≈ eigvals(Symmetric(input))
        @test input * vectors ≈ vectors * Diagonal(values) rtol = 100eps(T)
    end
    @test eigenvectors_allocations(workspace, copy(input)) == 0
    return
end

@testset "Symmetric eigenvalue workspace" begin
    setprecision(BigFloat, 256) do
        for T in (Float32, Float64, BigFloat)
            @testset "$T" begin
                check_workspace(T)
                if T != BigFloat
                    check_syevd(T)
                end
            end
        end
    end
end

end
