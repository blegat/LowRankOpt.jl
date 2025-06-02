using Test
using LowRankOpt
import Percival

set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
set_attribute(model, "sub_solver", Percival.PercivalSolver)
set_attribute(model, "ranks", [1])
optimize!(model)
solution_summary(model)

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
        @test isempty(NLPModelsTest.gradient_check(bm; x))
    end
    @testset "Jacobian" begin
        jac_check(bm, x)
    end
    @testset "Hessian" begin
        hess_check(bm, x)
    end
end

function full_check(model)
    set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
    set_attribute(model, "sub_solver", Percival.PercivalSolver)
    set_attribute(model, "max_iter", 0)
    set_attribute(model, "max_eval", 0)
    set_attribute(model, "ranks", [1])
    optimize!(model)
    diff_check(model)
end

@testset "Simple LP" begin
    model = Model(LowRankOpt.Optimizer)
    @variable(model, x)
    @constraint(model, x + 1 >= 0)
    @objective(model, Min, x)
    full_check(model)
end

@testset "Simple SDP" begin
    model = Model(LowRankOpt.Optimizer)
    @variable(model, x)
    @constraint(model, x * ones(1, 1) in PSDCone())
    @objective(model, Min, x)
    full_check(model)
end

@testset "Simple SDP" begin
    include(joinpath(dirname(@__DIR__), "examples", "maxcut.jl"))
    weights = [0 5 7 6; 5 0 0 1; 7 0 0 1; 6 1 1 0];
    model = maxcut(weights, LowRankOpt.Optimizer)
    set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
    set_attribute(model, "sub_solver", Percival.PercivalSolver)
    set_attribute(model, "ranks", [1])
    set_attribute(model, "max_iter", 200)
    set_attribute(model, "max_eval", 200)
    set_attribute(model, "verbose", 2)
    optimize!(model)
    solution_summary(model)
    diff_check(model)
end
