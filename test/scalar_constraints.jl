module TestScalarConstraints
using Test, LinearAlgebra, SparseArrays
import LowRankOpt as LRO
import NLPModels

@testset "Bounded rows and explicit dualization" begin
    model = LRO.Model(
        Matrix{Float64}[],
        (),
        [-1.0, 2.0],
        sparsevec(Float64[]),
        spzeros(2, 0),
        Int[];
        objective_constant = 7,
    )
    J = sparse([1.0 2; 3 4; 5 6; 7 8])
    c = [1.0, 2, 3, 4]
    LRO.append_scalar_constraints!(
        model,
        J,
        c,
        [0.0, -Inf, 3, -1],
        [Inf, 5, 3, 2],
    )
    @test model.meta.nvar == 2
    @test model.meta.ncon == 4
    @test model.meta.jfix == [3]
    @test model.meta.jrng == [4]
    @test NLPModels.cons(model, [2.0, 3]) ≈ J*[2.0, 3] + c
    dual = LRO.dual_model(model)
    @test LRO.primal_model(dual) === model
    @test LRO.primal_model(dual).objective_constant == 7
    @test LRO.primal_model(dual).meta.jfix == [3]
    @test LRO.primal_model(dual).meta.jrng == [4]
    reduced = LRO.dual_model(model; equalities = false)
    @test LRO.primal_model(reduced) === model
    @test size(reduced.C_lin) == (2, 4)
    @test size(dual.C_lin) == (2, 6)
    @test Matrix(dual.C_lin) ≈
          hcat(-J[1, :], J[2, :], -J[3, :], J[3, :], -J[4, :], J[4, :])
    @test dual.d_lin ≈ [1, 3, 0, 0, 5, -2]
    cone = LRO.VectorizedConeVector(collect(1.0:6), dual.dim)
    multipliers = LRO.cone_multipliers(model, cone)
    @test multipliers[LRO.ScalarIndex] ≈ [1, -2, -1, -1]
    C, d, rows, signs = LRO.scalar_bounds(model; equalities = false)
    @test size(C) == (2, 4)
    @test rows == [1, 2, 4, 4]
    @test signs == [-1, 1, -1, 1]
end
end
