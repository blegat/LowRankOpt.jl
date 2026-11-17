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
        LRO.add_jtprod!(model, X, result, idx)
        LRO.unsafe_jprod(model, y, idx)
    end
end

@testset "Optional Schur workspaces" begin
    model, A = example()
    full = LRO.ModelWorkspace(model)
    lean = LRO.ModelWorkspace(model; schur = false)
    @test !isnothing(full.schur_buffer)
    @test isnothing(lean.schur_buffer)
    @test lean.model === full.model === model
    @test !(lean isa NLPModels.AbstractNLPModel)
    idx = LRO.MatrixIndex(1)
    X = [4.0 0.5; 0.5 5.0]
    y = [1.0, -2.0, 3.0]
    result = zeros(3)
    for buffered in (full, lean)
        fill!(result, 0.0)
        LRO.add_jtprod!(buffered, X, result, idx)
        @test result ≈ [-dot(a, X) for a in A]
        @test LRO.unsafe_jprod(buffered, y, idx) ≈
              -sum(y[j] * A[j] for j in 1:3)
        @test LRO.unsafe_cons(buffered, y, idx) ≈
              model.C[1] - sum(y[j] * A[j] for j in 1:3)
        product_allocations(buffered, X, y, result, idx)
        @test product_allocations(buffered, X, y, result, idx) == 0
        shaped = LRO.ShapedConeVector([1.0, 2.0], [X])
        LRO.jtprod!(buffered, shaped, result)
        @test result ≈ [-dot(a, X) for a in A] - model.C_lin * shaped.scalars
        @test result ≈ model.jacobian' * [shaped.scalars; vec(X)]
        native_product =
            LRO.VectorizedConeVector(zeros(model.meta.ncon), model.dim)
        LRO.jprod!(buffered, y, native_product)
        @test Vector(native_product) ≈ model.jacobian * y
        LRO.cons!(buffered, y, native_product)
        @test Vector(native_product) ≈ NLPModels.cons(model, y)
    end

    # Rewrapping preserves both Jacobian caches and their sparse patterns,
    # and ignores `dense_condition` if the Schur workspaces are kept.
    condition = LRO.MinNonzeros(0)
    @test LRO.ModelWorkspace(full; dense_condition = condition) === full
    @test LRO.ModelWorkspace(lean; schur = false) === lean
    dropped = LRO.ModelWorkspace(full; schur = false)
    restored = LRO.ModelWorkspace(lean)
    for (source, wrapped) in ((full, dropped), (lean, restored))
        @test wrapped !== source
        @test wrapped.model === source.model
        @test wrapped.coefficient_buffer === source.coefficient_buffer
        @test wrapped.jprod_buffer === source.jprod_buffer
    end
    @test isnothing(dropped.schur_buffer)
    @test !isnothing(restored.schur_buffer)
    L = LRO.ShapedConeVector([1.0, 2.0], [[2.0 0.3; 0.3 1.0]])
    R = LRO.ShapedConeVector([3.0, 4.0], [[1.0 -0.1; -0.1 3.0]])
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

@testset "Scalar-only workspace storage" begin
    model, _ = example()
    @test_throws ArgumentError LRO.ModelWorkspace(model; jacobian = false)
    full = LRO.ModelWorkspace(model)
    disabled = LRO.ModelWorkspace(model; schur = false, jacobian = false)
    dropped = LRO.ModelWorkspace(full; schur = false, jacobian = false)
    @test_throws ArgumentError LRO.ModelWorkspace(full; jacobian = false)
    @test dropped.scalar_jacobian_transpose === full.scalar_jacobian_transpose
    for workspace in (disabled, dropped)
        @test workspace.model === model
        @test isnothing(workspace.coefficient_buffer)
        @test isnothing(workspace.jprod_buffer)
        @test isnothing(workspace.schur_buffer)
        @test LRO.ModelWorkspace(workspace; schur = false, jacobian = false) ===
              workspace
        direction = [1.0, -2.0, 3.0]
        scalar = zeros(2)
        result = zeros(3)
        LRO.jprod!(workspace, direction, scalar, LRO.ScalarIndex)
        @test scalar ≈ model.jacobian[1:2, :] * direction
        LRO.jtprod!(workspace, [2.0, -1.0], result, LRO.ScalarIndex)
        @test result ≈ model.jacobian[1:2, :]' * [2.0, -1.0]
        LRO.cons!(workspace, direction, scalar, LRO.ScalarIndex)
        @test scalar ≈ NLPModels.cons(model, direction)[1:2]
        idx = LRO.MatrixIndex(1)
        @test_throws ArgumentError LRO.unsafe_jprod(workspace, direction, idx)
        @test_throws ArgumentError LRO.add_jtprod!(
            workspace,
            ones(2, 2),
            result,
            idx,
        )
        restored = LRO.ModelWorkspace(workspace)
        @test restored.model === model
        @test restored.scalar_jacobian_transpose ===
              workspace.scalar_jacobian_transpose
        @test !isnothing(restored.coefficient_buffer)
        @test !isnothing(restored.jprod_buffer)
        @test !isnothing(restored.schur_buffer)
        product = LRO.VectorizedConeVector(zeros(model.meta.ncon), model.dim)
        LRO.jprod!(restored, direction, product)
        @test Vector(product) ≈ model.jacobian * direction
        products_only = LRO.ModelWorkspace(workspace; schur = false)
        @test isnothing(products_only.schur_buffer)
        @test !isnothing(products_only.coefficient_buffer)
        @test !isnothing(products_only.jprod_buffer)
        LRO.jprod!(products_only, direction, product)
        @test Vector(product) ≈ model.jacobian * direction
    end
end

@testset "Native products preserve strided vector views" begin
    model, _ = example()
    workspace = LRO.ModelWorkspace(model)
    function guarded(values)
        storage = fill(123.0, 2length(values) + 1)
        slice = view(storage, 2:2:(2length(values)))
        slice .= values
        return storage, slice
    end
    direction_storage, direction = guarded([1.0, -2.0, 3.0])
    multiplier_storage, multipliers = guarded(collect(1.0:model.meta.ncon))
    output_storage, output = guarded(zeros(model.meta.ncon))
    adjoint_storage, adjoint = guarded(zeros(model.meta.nvar))
    original_direction = copy(direction_storage)
    original_multipliers = copy(multiplier_storage)
    for operation in (LRO.jprod!, LRO.cons!)
        operation(workspace, direction, output)
        expected = model.jacobian * direction
        if operation === LRO.cons!
            expected += model.constant
        end
        @test output ≈ expected
        @test all(==(123.0), output_storage[1:2:end])
    end
    LRO.jtprod!(workspace, multipliers, adjoint)
    @test adjoint ≈ model.jacobian' * multipliers
    @test all(==(123.0), adjoint_storage[1:2:end])
    @test direction_storage == original_direction
    @test multiplier_storage == original_multipliers
end

end
