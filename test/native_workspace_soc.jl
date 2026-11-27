module TestNativeWorkspaceSOC

using Test, LinearAlgebra, SparseArrays
import LowRankOpt as LRO
import NLPModels

@testset "Mixed native workspace products" begin
    model = LRO.Model(
        [spdiagm([3.0, 4.0])],
        reshape([spdiagm([1.0, 2.0]), sparse([0.0 1; 1 0])], 1, 2),
        [-1.0, 2.0],
        sparse([2.0]),
        sparse(reshape([1.0, -2.0], 2, 1)),
        [2];
        d_eq = sparsevec([-1.0]),
        C_eq = sparse(reshape([-1.0, -1.0], 2, 1)),
    )
    LRO.append_second_order_constraints!(
        model,
        sparse([1.0 0; 0 1; 1 -1]),
        [3.0, 0, 0],
        [3],
    )
    LRO.append_second_order_constraints!(
        model,
        sparse([2.0 -1; 1 2]),
        [4.0, 1.0],
        [2],
    )
    workspace = LRO.ModelWorkspace(model; schur = false)
    lean = LRO.ModelWorkspace(model; schur = false, jacobian = false)
    @test isempty(lean.soc_jacobian_transpose)
    @test_throws ArgumentError LRO.ModelWorkspace(model)
    @test_throws ArgumentError LRO.ModelWorkspace(workspace)
    @test LRO.ModelWorkspace(workspace; schur = false) === workspace
    restored = LRO.ModelWorkspace(lean; schur = false)
    @test length(restored.soc_jacobian_transpose) == 2
    dropped = LRO.ModelWorkspace(workspace; schur = false, jacobian = false)
    @test isempty(dropped.soc_jacobian_transpose)
    x, v = [0.2, 0.3], [-0.4, 0.7]
    multipliers = collect(range(-1.0, 2.0; length = model.meta.ncon))
    result, adjoint = zeros(model.meta.ncon), zeros(model.meta.nvar)
    for w in (workspace, restored)
        LRO.jprod!(w, v, result)
        @test result ≈ model.jacobian * v
        LRO.jtprod!(w, multipliers, adjoint)
        @test adjoint ≈ model.jacobian' * multipliers
        @test dot(result, multipliers) ≈ dot(v, adjoint)
        LRO.cons!(w, x, result)
        @test result ≈ NLPModels.cons(model, x)
    end
    cone = LRO.VectorizedConeVector(multipliers, model.dim)
    shaped = LRO.ShapedConeVector(
        copy(cone[LRO.ScalarIndex]),
        [copy(cone[i]) for i in LRO.second_order_indices(model)],
        [copy(cone[LRO.MatrixIndex(1)])],
    )
    LRO.jprod!(workspace, v, shaped)
    @test collect(shaped) ≈ model.jacobian * v
    LRO.jtprod!(workspace, shaped, adjoint)
    @test adjoint ≈ model.jacobian' * collect(shaped)
    LRO.cons!(workspace, x, shaped)
    @test collect(shaped) ≈ NLPModels.cons(model, x)
    @test_throws ArgumentError LRO.jprod!(lean, v, zeros(3), LRO.SOCIndex(1))
    @test_throws ArgumentError LRO.add_jtprod!(
        lean,
        zeros(3),
        adjoint,
        LRO.SOCIndex(1),
    )
    @test_throws ArgumentError LRO.cons!(lean, x, zeros(3), LRO.SOCIndex(1))
    function product_allocations(w, x, v, multipliers, result, adjoint)
        return @allocated begin
            LRO.jprod!(w, v, result)
            LRO.jtprod!(w, multipliers, adjoint)
            LRO.cons!(w, x, result)
        end
    end
    product_allocations(workspace, x, v, multipliers, result, adjoint)
    @test product_allocations(workspace, x, v, multipliers, result, adjoint) ==
          0
end

@testset "Native SOC residuals" begin
    model = LRO.Model(
        Matrix{Float64}[],
        (),
        [-1.0, 2.0],
        spzeros(0),
        spzeros(2, 0),
        Int[];
        objective_constant = 7,
    )
    LRO.append_second_order_constraints!(
        model,
        sparse([1.0 0; 0 1; 1 -1]),
        [2.0, 0, 0],
        [3],
    )
    x, slack, multipliers = [1.0, 2.0], [1.0, 2.0, 0.0], [1.0, 3.0, 0.0]
    expected =
        [sqrt(5)/3, 1/3, sqrt(5)/(1+sqrt(5)), 4/(1+sqrt(5)), -5/14, -7/14]
    for shaped in (false, true)
        s =
            shaped ?
            LRO.ShapedConeVector(Float64[], [slack], Matrix{Float64}[]) : slack
        y =
            shaped ?
            LRO.ShapedConeVector(Float64[], [multipliers], Matrix{Float64}[]) :
            multipliers
        @test collect(LRO.errors(model, x; slack = s, multipliers = y)) ≈
              expected
    end
    feasible = LRO.errors(model, x; multipliers = [-3.0, 1.0, 1.0])
    @test feasible[1] == feasible[2] == feasible[4] == 0
end

end
