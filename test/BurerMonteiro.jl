using Test
using LinearAlgebra
using JuMP
import LowRankOpt as LRO
import Percival

function test_vecprod(f, len, J; tol = 1e-6)
    v = ones(len)
    @test f(v) ≈ J * v rtol = tol atol = tol
    v = -ones(len)
    @test f(v) ≈ J * v rtol = tol atol = tol
    v = 2ones(len)
    @test f(v) ≈ J * v rtol = tol atol = tol
    v = -2ones(len)
    @test f(v) ≈ J * v rtol = tol atol = tol
    v = rand(len)
    @test f(v) ≈ J * v rtol = tol atol = tol
end

import NLPModels, FiniteDiff

function grad_check(model, x; tol = 1e-6)
    f(x) = NLPModels.obj(model, x)
    g = FiniteDiff.finite_difference_gradient(f, x)
    @test NLPModels.grad(model, x) ≈ g rtol = tol atol = tol
end

function jac_check(model, x; kws...)
    f(x) = NLPModels.cons(model, x)
    J = FiniteDiff.finite_difference_jacobian(f, x)
    @testset "jprod" begin
        test_vecprod(v -> NLPModels.jprod(model, x, v), model.meta.nvar, J; kws...)
    end
    @testset "jtprod" begin
        test_vecprod(v -> NLPModels.jtprod(model, x, v), model.meta.ncon, J'; kws...)
    end
end

function hess_check(model, x; kws...)
    obj_weight = rand()
    y = rand(model.meta.ncon)
    f(x) = obj_weight * NLPModels.obj(model, x) - dot(y, NLPModels.cons(model, x))
    J = FiniteDiff.finite_difference_hessian(f, x)
    test_vecprod(v -> NLPModels.hprod(model, x, y, v; obj_weight), model.meta.nvar, J; kws...)
end

using NLPModelsTest
function diff_check(model)
    b = unsafe_backend(model)
    bm = b.solver.model
    x = rand(bm.meta.nvar)
    @testset "Gradient" begin
        grad_check(bm, x)
        @test isempty(NLPModelsTest.gradient_check(bm; x))
    end
    @testset "Jacobian" begin
        jac_check(bm, x)
    end
    @testset "Hessian" begin
        hess_check(bm, x)
    end
end

@testset "Simple LP" begin
    model = Model(LRO.Optimizer)
    @variable(model, x)
    @constraint(model, con_ref, 1 - x in Nonnegatives())
    @objective(model, Max, x)
    set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
    set_attribute(model, "sub_solver", Percival.PercivalSolver)
    set_attribute(model, "verbose", 1)
    set_attribute(model, "ranks", Int[])

    set_attribute(model, "max_iter", 0)
    optimize!(model)
    @test termination_status(model) == MOI.ITERATION_LIMIT
    diff_check(model)

    set_attribute(model, "max_iter", 10)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test value(x) ≈ 1
    @test dual(con_ref) ≈ 1
    @test objective_value(model) ≈ 1
    @test dual_objective_value(model) ≈ 1
    diff_check(model)
end

@testset "Simple SDP" begin
    model = Model(LRO.Optimizer)
    @variable(model, x)
    @constraint(model, con_ref, Symmetric((1 - x) * ones(1, 1)) in PSDCone())
    @objective(model, Max, x)
    set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
    set_attribute(model, "sub_solver", Percival.PercivalSolver)
    set_attribute(model, "ranks", [1])
    set_attribute(model, "verbose", 2)
    @test solver_name(model) == "LowRankOpt with no solver loaded yet"

    set_attribute(model, "max_iter", 0)
    optimize!(model)
    solution_summary(model)
    @test solver_name(model) == "BurerMonteiro with Percival"
    @test termination_status(model) == MOI.ITERATION_LIMIT
    diff_check(model)

    set_attribute(model, "max_iter", 10)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test dual_status(model) == MOI.FEASIBLE_POINT
    @test value(x) ≈ 1
    t = MOI.get(model, MOI.ConstraintDual(), con_ref)
    @test t isa LRO.TriangleVectorization
    @test t.matrix isa LRO.Factorization
    @test t.matrix ≈ ones(1, 1)
    @test only(dual(con_ref)) ≈ 1
    solution_summary(model)
    @test objective_value(model) ≈ 1
    @test dual_objective_value(model) ≈ 1
    diff_check(model)
end

@testset "Simple SDP" begin
    include(joinpath(dirname(@__DIR__), "examples", "maxcut.jl"))
    weights = [0 5 7 6; 5 0 0 1; 7 0 0 1; 6 1 1 0];
    model = maxcut(weights, LRO.Optimizer)
    set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
    set_attribute(model, "sub_solver", Percival.PercivalSolver)
    set_attribute(model, "ranks", [1])
    set_attribute(model, "verbose", 2)
    optimize!(model)
    solution_summary(model)
    diff_check(model)
end
