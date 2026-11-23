module TestScalarConstraints
using Test, LinearAlgebra, SparseArrays
using JuMP
import LowRankOpt as LRO
import NLPModels
import Percival

@testset "MOI bounds and duals for mixed constraint types" begin
    model = Model(LRO.Optimizer)
    set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
    set_attribute(model, "sub_solver", Percival.PercivalSolver)
    set_attribute(model, "ranks", Int[])
    set_silent(model)
    @variable(model, x[1:8])
    @constraint(model, positive, [x[1] - 1, x[2] - 2] in Nonnegatives())
    @constraint(model, upper, 2x[3] <= 6)
    @constraint(model, interval, 8 <= 2x[4] <= 12)
    @constraint(model, negative, [x[5] - 5, x[6] - 6] in Nonpositives())
    @constraint(model, fixed, [x[7] - 7, x[8] - 8] in Zeros())
    @objective(model, Min, dot([1, 2, -6, 8, -5, -6, 7, -8], x))
    optimize!(model)
    opt = unsafe_backend(model)
    F = MOI.ScalarAffineFunction{Float64}
    for S in (
        MOI.EqualTo{Float64},
        MOI.GreaterThan{Float64},
        MOI.LessThan{Float64},
        MOI.Interval{Float64},
    )
        @test !MOI.supports_constraint(opt, F, S)
    end
    @test all(iszero, opt.model.meta.lcon)
    @test all(==(Inf), opt.model.meta.ucon)
    bridges = sprint(print_active_bridges, model)
    @test occursin("SplitIntervalBridge", bridges)
    @test occursin("LessToGreaterBridge", bridges) ||
          occursin("NonposToNonnegBridge", bridges)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test value.(x) ≈ 1:8 atol = 1e-5
    @test dual(positive) ≈ [1, 2] atol = 1e-5
    @test dual(upper) ≈ -3 atol = 1e-5
    @test dual(interval) ≈ 4 atol = 1e-5
    @test dual(negative) ≈ [-5, -6] atol = 1e-5
    @test dual(fixed) ≈ [7, -8] atol = 1e-5
end

@testset "Linear block dimensions" begin
    b = [1.0]
    c = spzeros(1)
    C = spzeros(1, 1)
    @test_throws DimensionMismatch LRO.Model(
        Matrix{Float64}[],
        (),
        b,
        c,
        spzeros(1, 0),
        Int[],
    )
    @test_throws DimensionMismatch LRO.Model(
        Matrix{Float64}[],
        (),
        b,
        c,
        C,
        Int[];
        d_eq = c,
        C_eq = spzeros(2, 1),
    )
end

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
    J = sparse([1.0 2; 3 4; 7 8])
    c = [1.0, 2, 4]
    LRO.append_scalar_constraints!(model, J, c, [0.0, -Inf, -1], [Inf, 5, 2])
    @test model.meta.jrng == [3]
    @test NLPModels.cons(model, [2.0, 3]) ≈ J*[2.0, 3] + c
    dual = LRO.dual_model(model)
    @test LRO.primal_model(dual) === model
    @test LRO.primal_model(dual).objective_constant == 7
    @test Matrix(dual.C_lin) ≈ hcat(-J[1, :], J[2, :], -J[3, :], J[3, :])
    @test dual.d_lin ≈ [1, 3, 5, -2]
    cone = LRO.VectorizedConeVector(collect(1.0:4), dual.dim)
    @test LRO.cone_multipliers(model, cone)[LRO.ScalarIndex] ≈ [1, -2, -1]

    LRO.append_scalar_constraints!(model, sparse([5.0 6]), [3.0], [3.0], [3.0])
    @test model.meta.jfix == [4]
    C, d, rows, signs = LRO.scalar_bounds(model, 1:3)
    @test C == dual.C_lin
    @test d == dual.d_lin
    @test rows == [1, 2, 3, 3]
    @test signs == [-1, 1, -1, 1]
    @test_throws ArgumentError LRO.scalar_bounds(model)
    @test_throws ArgumentError LRO.dual_model(model)
    @test_throws ArgumentError LRO.cone_multipliers(model, cone)
    @test_throws ArgumentError LRO.BurerMonteiro.Solver(
        model;
        sub_solver = Percival.PercivalSolver,
        ranks = Int[],
    )
end

@testset "An original inequality changed to an equality" begin
    model = LRO.Model(
        Matrix{Float64}[],
        (),
        [1.0],
        spzeros(1),
        sparse(ones(1, 1)),
        Int[],
    )
    cone = LRO.VectorizedConeVector([1.0], model.dim)
    model.meta = NLPModels.NLPModelMeta(model.meta; lcon = [2.0], ucon = [2.0])
    @test_throws ArgumentError LRO.dual_model(model)
    @test_throws ArgumentError LRO.cone_multipliers(model, cone)
end
end
