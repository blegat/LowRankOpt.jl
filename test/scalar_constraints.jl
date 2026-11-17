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
@testset "Original scalar rows with changed bounds" begin
    for (lower, upper, coefficients, constants, weights, multiplier) in (
        (0.0, Inf, [-2.0], [3.0], [3.0], 3.0),
        (1.0, Inf, [-2.0], [2.0], [3.0], 3.0),
        (-Inf, 7.0, [2.0], [4.0], [3.0], -3.0),
        (0.0, 7.0, [-2.0, 2.0], [3.0, 4.0], [3.0, 2.0], 1.0),
        (1.0, 7.0, [-2.0, 2.0], [2.0, 4.0], [3.0, 2.0], 1.0),
        (5.0, 5.0, [-2.0, 2.0], [-2.0, 2.0], [2.0, 3.0], -1.0),
    )
        # The original scalar row is c(x) = 2x + 3, before changing bounds.
        model = LRO.Model(
            Matrix{Float64}[],
            (),
            [-2multiplier],
            sparsevec([3.0]),
            sparse(reshape([-2.0], 1, 1)),
            Int[],
        )
        model.meta =
            NLPModels.NLPModelMeta(model.meta; lcon = [lower], ucon = [upper])
        @test LRO.num_scalars(model) == length(model.d_lin) == 1
        @test NLPModels.cons(model, [0.4]) ≈ [3.8]
        C, d, rows, signs = LRO.scalar_bounds(model)
        @test Matrix(C) == reshape(coefficients, 1, :)
        @test d == constants
        @test rows == ones(Int, length(weights))
        @test signs == coefficients ./ 2
        dual = LRO.dual_model(model)
        @test LRO.primal_model(dual) === model
        @test dual.meta.nvar == length(weights)
        @test dual.C_lin == C
        @test dual.d_lin == d
        cone = LRO.VectorizedConeVector(weights, dual.dim)
        mapped = LRO.cone_multipliers(model, cone)
        @test length(mapped) == model.meta.ncon == 1
        @test mapped[LRO.ScalarIndex] == [multiplier]
        # Cone/MOI multipliers have the opposite sign to NLPModels multipliers.
        nlp_multipliers = -Vector(mapped)
        @test nlp_multipliers == [-multiplier]
        adjoint = NLPModels.jtprod(model, [0.4], nlp_multipliers)
        @test adjoint == C * weights
        @test NLPModels.grad(model, [0.4]) + adjoint == [0.0]
        @test NLPModels.cons(dual, cone) == [0.0]
        if iszero(lower) && upper == Inf
            @test dual.C_lin === model.C_lin
            @test dual.d_lin === model.d_lin
            @test mapped === cone
        end
        C_reduced, d_reduced, rows_reduced, _ =
            LRO.scalar_bounds(model; equalities = false)
        if lower == upper
            @test size(C_reduced) == (1, 0)
            @test isempty(d_reduced)
            @test isempty(rows_reduced)
        else
            reduced = LRO.dual_model(model; equalities = false)
            @test reduced.C_lin == C_reduced == C
            @test reduced.d_lin == d_reduced == d
        end
    end
end

end
