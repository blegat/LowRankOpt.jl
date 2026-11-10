module TestEigenWorkspace

using Test, LinearAlgebra
# Register generic symmetric eigensolvers for BigFloat.
import GenericLinearAlgebra
import LowRankOpt as LRO

# Symmetric operator given by its products only
struct Operator{M}
    matrix::M
end

function LinearAlgebra.mul!(w, op::Operator, v)
    return LinearAlgebra.mul!(w, op.matrix, v)
end

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
        @test LRO.lanczos_eigmin!(workspace, Operator(input)) ≈ first(spectrum)
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
        @test length(workspace.syevd_work) == 1 + 6n + 2n^2
        @test length(workspace.syevd_iwork) == 3 + 5n
    end
    @test eigenvectors_allocations(workspace, copy(input)) == 0
    return
end

function lanczos_allocations(workspace, A)
    LRO.lanczos_eigmin!(workspace, A)
    return @allocated LRO.lanczos_eigmin!(workspace, A)
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
    # The estimate is `θ - r` with `r ≤ 1e-3 * abs(θ)` and `θ ≥ -1`.
    @test -1.002 <= estimate <= -1
    @test A == original
    # Julia v1.10 allocates in `lanczos_eigmin!` for `Float32`
    if VERSION >= v"1.11"
        @test lanczos_allocations(workspace, A) == 0
    end
    # The evenly spread spectrum needs a restart to converge to `rtol = 1e-4`
    A = matrix(range(-T(1), T(1); length = n))
    original = copy(A)
    @test LRO.lanczos_eigmin!(workspace, A; rtol = T(1e-4)) ≈ -1 rtol = 1e-3
    @test A == original
    # Without tolerance, it does not converge within the restarts so the
    # eigenvalues are computed.
    @test LRO.lanczos_eigmin!(workspace, A; rtol = 0) ≈ -1
    @test A == original
    # The same with products only, the fallback forms the matrix.
    op = Operator(A)
    @test LRO.lanczos_eigmin!(workspace, op; rtol = T(1e-4)) ≈ -1 rtol = 1e-3
    @test LRO.lanczos_eigmin!(workspace, op; rtol = 0) ≈ -1
    if VERSION >= v"1.11"
        @test lanczos_allocations(workspace, op) == 0
    end
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
                if T != BigFloat
                    check_syevd(T)
                end
            end
        end
    end
end

end
