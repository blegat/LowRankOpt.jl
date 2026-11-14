module TestOptionalSchurBuffer

using Test, LinearAlgebra, SparseArrays
import LowRankOpt as LRO
import NLPModels

function example()
    A = [sparse([1.0 0; 0 0]), sparse([0.0 1; 1 0]), sparse([0.0 0; 0 1])]
    model = LRO.Model(
        [sparse([2.0 0; 0 3])],
        reshape(A, 1, 3),
        [1.0, 2.0, 3.0],
        SparseVector{Float64,Int64}(sparse([1.0, 2.0])),
        SparseMatrixCSC{Float64,Int64}(sparse([1.0 0; 0 2; 3 1])),
        Int64[2],
    )
    return model, A
end

function product_allocations(model, X, y, result, idx)
    return @allocated begin
        fill!(result, 0.0)
        LRO.add_jprod!(model, X, result, idx)
        LRO.unsafe_jtprod(model, y, idx)
    end
end

@testset "Optional Schur workspaces" begin
    model, A = example()
    full = LRO.BufferedModelForSchur(model, 8)
    lean = LRO.BufferedModelForSchur(model, 8; schur = false)
    @test !isnothing(full.schur_buffer)
    @test isnothing(lean.schur_buffer)
    @test lean.model === full.model === model
    @test lean.meta === model.meta
    idx = LRO.MatrixIndex(1)
    X = [4.0 0.5; 0.5 5.0]
    y = [1.0, -2.0, 3.0]
    result = zeros(3)
    for buffered in (full, lean)
        fill!(result, 0.0)
        LRO.add_jprod!(buffered, X, result, idx)
        @test result ≈ [dot(a, X) for a in A]
        @test LRO.unsafe_jtprod(buffered, y, idx) ≈
              sum(y[j] * A[j] for j in 1:3)
        @test LRO.unsafe_dual_cons(buffered, y, idx) ≈
              model.C[1] - sum(y[j] * A[j] for j in 1:3)
        product_allocations(buffered, X, y, result, idx)
        @test product_allocations(buffered, X, y, result, idx) == 0
        shaped = LRO.ShapedSolution([1.0, 2.0], [X])
        NLPModels.jprod!(buffered, shaped, shaped, result)
        @test result ≈ [dot(a, X) for a in A] + model.C_lin * shaped.scalars
        @test NLPModels.obj(buffered, shaped) ≈ dot(model.C[1], X) + 5.0
        @test LRO.grad(buffered, idx) === model.C[1]
        @test LRO.cons_constant(buffered) === model.b
    end

    # Rewrapping preserves both Jacobian caches, including their sparse patterns.
    @test LRO.BufferedModelForSchur(full, 0) === full
    @test LRO.BufferedModelForSchur(lean, 0; schur = false) === lean
    dropped = LRO.BufferedModelForSchur(full, 8; schur = false)
    restored = LRO.BufferedModelForSchur(lean, 8)
    for (source, wrapped) in ((full, dropped), (lean, restored))
        @test wrapped !== source
        @test wrapped.model === source.model
        @test wrapped.jprod_buffer === source.jprod_buffer
        @test wrapped.jtprod_buffer === source.jtprod_buffer
    end
    @test isnothing(dropped.schur_buffer)
    @test !isnothing(restored.schur_buffer)
    L = LRO.ShapedSolution([1.0, 2.0], [[2.0 0.3; 0.3 1.0]])
    R = LRO.ShapedSolution([3.0, 4.0], [[1.0 -0.1; -0.1 3.0]])
    expected = [tr(A[j] * L[idx] * A[k] * R[idx]) for j in 1:3, k in 1:3]
    expected += model.C_lin * Diagonal(L.scalars .* R.scalars) * model.C_lin'
    for buffered in (full, restored)
        H = zeros(3, 3)
        LRO.schur_complement!(buffered, L, R, H)
        @test H ≈ expected
        LRO.eval_schur_complement!(buffered, L, R, y, result)
        @test result ≈ expected * y
    end
    for buffered in (lean, dropped)
        @test_throws ArgumentError LRO.schur_complement!(
            buffered,
            L,
            R,
            zeros(3, 3),
        )
        @test_throws ArgumentError LRO.eval_schur_complement!(
            buffered,
            L,
            R,
            y,
            result,
        )
        @test_throws ArgumentError LRO.add_schur_complement!(
            buffered,
            idx,
            L[idx],
            R[idx],
            zeros(3, 3),
        )
    end
end

end
