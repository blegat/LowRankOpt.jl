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
        # The Krylov subspace is the whole space after 3 iterations.
        @test LRO.lanczos_eigmin!(workspace, input) ≈ first(spectrum)
        @test input == original
        @test workspace.scratch === scratch
    end
    return
end

function check_lanczos(::Type{T}) where {T}
    n = 100
    Q = Matrix(qr(sin.(1:n) .* cos.((1:n)' ./ 3)).Q)
    matrix(spectrum) = Matrix(Symmetric(Q * Diagonal(spectrum) * Q'))
    workspace = LRO.EigenWorkspace(zeros(T, n, n))
    # The isolated minimum eigenvalue converges quickly.
    A = matrix([-T(1); range(T(1), T(2); length = n - 1)])
    original = copy(A)
    estimate = LRO.lanczos_eigmin!(workspace, A)
    # The estimate is `θ - r` with `r ≤ 0.02 * abs(θ)` and `θ ≥ -1`.
    @test -1.03 <= estimate <= -1
    @test A == original
    # Without tolerance, the evenly spread spectrum does not converge within
    # the iterations so the eigenvalues are computed.
    A = matrix(range(-T(1), T(1); length = n))
    original = copy(A)
    @test LRO.lanczos_eigmin!(workspace, A; rtol = 0) ≈ -1
    @test A == original
    return
end

@testset "Lanczos minimum eigenvalue" begin
    for T in (Float32, Float64)
        @testset "$T" begin
            check_lanczos(T)
        end
    end
end

@testset "Symmetric eigenvalue workspace" begin
    setprecision(BigFloat, 256) do
        for T in (Float32, Float64, BigFloat)
            @testset "$T" begin
                check_workspace(T)
            end
        end
    end
end

end
