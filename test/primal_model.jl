module TestPrimalModel

using Test, LinearAlgebra, SparseArrays
import LowRankOpt as LRO
import NLPModels

@testset "Free-variable NLPModels contract" begin
    C = sparse([2.0 0.5; 0.5 3.0])
    A = reshape([sparse([1.0 0; 0 2]), sparse([0.0 1; 1 0])], 1, 2)
    model = LRO.Model(
        [C],
        A,
        [-1.0, 2.0],
        sparse([3.0, 4.0]),
        sparse([1.0 0; -1 2]),
        [2];
        objective_constant = 7,
    )
    x, v = [0.2, 0.4], [0.3, -0.7]
    J = [-1.0 1; 0 -2; -1 0; 0 -1; 0 -1; -2 0]
    expected = [3.2, 3.2, 1.8, 0.1, 0.1, 2.6]
    @test model.meta.nvar == 2
    @test model.meta.ncon == 6
    @test model.meta.lcon == [0.0, 0.0, -Inf, -Inf, -Inf, -Inf]
    @test all(isinf, model.meta.ucon)
    @test all(==(-Inf), model.meta.lvar)
    @test all(==(Inf), model.meta.uvar)
    @test NLPModels.obj(model, x) ≈ 6.4
    @test NLPModels.grad(model, x) ≈ [1.0, -2.0]
    @test NLPModels.cons(model, x) ≈ expected
    @test NLPModels.jac(model, x) ≈ J
    @test model.meta.lin_nnzj == model.meta.nnzj
    @test model.meta.nln_nnzj == 0
    @test NLPModels.cons_lin(model, x) ≈ expected
    @test isempty(NLPModels.cons_nln(model, x))
    @test NLPModels.jac_lin(model, x) ≈ J
    @test size(NLPModels.jac_nln(model, x)) == (0, 2)
    @test NLPModels.jprod_lin(model, x, v) ≈ J * v
    @test isempty(NLPModels.jprod_nln(model, x, v))
    @test NLPModels.jtprod_lin(model, x, ones(6)) ≈ J' * ones(6)
    @test iszero(NLPModels.jtprod_nln(model, x, Float64[]))
    @test iszero(NLPModels.hess(model, x))
    @test iszero(NLPModels.hprod(model, x, v))
    @test NLPModels.jprod(model, x, v) ≈ J * v
    multipliers = [0.3, -0.4, 0.5, -0.6, 0.7, 0.8]
    @test NLPModels.jtprod(model, x, multipliers) ≈ J' * multipliers
    @test dot(NLPModels.jprod(model, x, v), multipliers) ≈
          dot(v, NLPModels.jtprod(model, x, multipliers))
    h = 1e-6
    @test (
        NLPModels.cons(model, x + h * v) - NLPModels.cons(model, x - h * v)
    ) / (2h) ≈ J * v atol = 1e-8
    @test iszero(NLPModels.hess(model, x, multipliers))
    @test iszero(NLPModels.hprod(model, x, multipliers, v))
    @test model.C[1] === C
    @test model.b == [-1.0, 2.0]
    @test LRO.primal_model(model) === model
    for dual in (LRO.DualModel(model), LRO.dual_model(model))
        @test dual.model === model
        @test LRO.primal_model(dual) === model
        @test dual.C === model.C
        @test dual.A === model.A
        @test dual.b === model.b
        @test dual.d_lin === model.d_lin
        @test dual.C_lin === model.C_lin
        @test dual.msizes === model.msizes
        @test dual.meta.nvar == model.meta.ncon
        @test dual.meta.ncon == model.meta.nvar
        @test NLPModels.obj(LRO.primal_model(dual), x) ≈ 6.4
        @test LRO.primal_model(dual).objective_constant == 7
        buffered = LRO.BufferedDualModel(dual, 8)
        @test LRO.primal_model(buffered) === model
        @test NLPModels.obj(LRO.primal_model(buffered), x) ≈ 6.4
    end
    @test LRO.num_scalars(model) == 2
    @test LRO.num_matrices(model) == 1
    @test LRO.side_dimension(model, LRO.MatrixIndex(1)) == 2
    @test !NLPModels.unconstrained(model)
    @test !NLPModels.has_bounds(model)
end

@testset "Primal metadata preserves arithmetic type" begin
    for T in (Float32, BigFloat)
        one = sparse(ones(T, 1, 1))
        model = LRO.Model(
            [one],
            reshape([one], 1, 1),
            T[-1],
            sparse(T[2]),
            one,
            [1],
        )
        @test eltype(model.meta.x0) == T
        @test eltype(model.meta.lcon) == T
        @test NLPModels.obj(model, T[0.5]) ≈ T(0.5)
        @test NLPModels.cons(model, T[0.5]) ≈ T[1.5, 0.5]
    end
end

@testset "Model accepts 32-bit sparse indices" begin
    C = sparse(Int32[1, 2], Int32[1, 2], [2.0, 3.0], 2, 2)
    A = reshape([sparse(Int32[1], Int32[1], [1.0], 2, 2)], 1, 1)
    model = LRO.Model(
        [C],
        A,
        [1.0],
        sparsevec(Int32[1], [4.0], 1),
        sparse(Int32[1], Int32[1], [2.0], 1, 1),
        Int32[2],
    )
    @test model.meta.nvar == 1
    @test model.meta.ncon == 5
    @test NLPModels.obj(model, [1.5]) == -1.5
    @test NLPModels.cons(model, [1.5]) == [1.0, 0.5, 0.0, 0.0, 3.0]
    @test NLPModels.jac(model, [1.5]) == reshape([-2.0, -1.0, 0, 0, 0], 5, 1)
    @test LRO.primal_model(LRO.DualModel(model)) === model
end

@testset "Constant and unconstrained conic models" begin
    C = sparse(Matrix{Float64}(I, 2, 2))
    model = LRO.Model(
        [C],
        Matrix{typeof(C)}(undef, 1, 0),
        Float64[],
        spzeros(0),
        spzeros(0, 0),
        [2];
        objective_constant = 3,
    )
    @test model.meta.nvar == 0
    @test model.meta.ncon == 4
    @test NLPModels.obj(model, Float64[]) == 3
    @test NLPModels.cons(model, Float64[]) == vec(Matrix(C))
    @test size(NLPModels.jac(model, Float64[])) == (4, 0)
    @test isempty(NLPModels.grad(model, Float64[]))
    model = LRO.Model(
        SparseMatrixCSC{Float64,Int}[],
        Matrix{SparseMatrixCSC{Float64,Int}}(undef, 0, 1),
        [-1.0],
        spzeros(0),
        spzeros(1, 0),
        Int[],
    )
    @test model.meta.nvar == 1
    @test model.meta.ncon == 0
    @test NLPModels.unconstrained(model)
    @test isempty(NLPModels.cons(model, [1.0]))
end

end
