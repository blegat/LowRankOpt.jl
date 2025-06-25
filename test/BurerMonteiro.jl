using Test
using LinearAlgebra
using SparseArrays
using JuMP
import LowRankOpt as LRO
using Dualization
import SolverCore
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
        test_vecprod(
            v -> NLPModels.jprod(model, x, v),
            model.meta.nvar,
            J;
            kws...,
        )
    end
    @testset "jtprod" begin
        test_vecprod(
            v -> NLPModels.jtprod(model, x, v),
            model.meta.ncon,
            J';
            kws...,
        )
    end
end

function hess_check(model, x; kws...)
    obj_weight = rand()
    y = rand(model.meta.ncon)
    f(x) =
        obj_weight * NLPModels.obj(model, x) - dot(y, NLPModels.cons(model, x))
    J = FiniteDiff.finite_difference_hessian(f, x)
    return test_vecprod(
        v -> NLPModels.hprod(model, x, y, v; obj_weight),
        model.meta.nvar,
        J;
        kws...,
    )
end

using NLPModelsTest
function diff_check(model)
    b = unsafe_backend(model)
    if b isa DualOptimizer
        b = b.dual_problem.dual_model.model.optimizer
    end
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

@testset "Simple LP $opt" for opt in
                              [LRO.Optimizer, dual_optimizer(LRO.Optimizer)]
    model = Model(opt)
    @variable(model, x)
    @constraint(model, con_ref, 1 - x in Nonnegatives())
    @objective(model, Max, x)
    set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
    set_attribute(model, "sub_solver", Percival.PercivalSolver)
    set_silent(model)
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
    b = unsafe_backend(model)
    if !(b isa DualOptimizer)
        @test isnothing(LRO.buffer_for_jtprod(b.model))
    end
    raw_sol = MOI.get(model, LRO.RawStatus(:solution))
    sol = MOI.get(model, LRO.Solution())
    @test raw_sol isa Vector{Float64}
    @test sol isa LRO.BurerMonteiro.Solution{Float64,Vector{Float64}}
    outer = LRO.BurerMonteiro._OuterProduct(sol, sol)
    @test length(outer) == length(sol)
    @test sprint(show, outer) == "_OuterProduct($sol, $sol)"
    @test sol == raw_sol
    if b isa DualOptimizer
        @test abs(sol[1]) < 1e-6
    end
    diff_check(model)
end;

@testset "Simple SDP $opt" for (is_dual, opt) in [
    (false, LRO.Optimizer),
    (true, dual_optimizer(LRO.Optimizer)),
]
    model = Model(opt)
    @variable(model, x)
    @constraint(model, con_ref, Symmetric((1 - x) * ones(1, 1)) in PSDCone())
    @objective(model, Max, x)
    set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
    set_attribute(model, "sub_solver", Percival.PercivalSolver)
    set_attribute(model, "ranks", [1])
    set_attribute(model, "verbose", 2)
    if is_dual
        @test solver_name(model) ==
              "Dual model with LowRankOpt with no solver loaded yet attached"
    else
        @test solver_name(model) == "LowRankOpt with no solver loaded yet"
    end

    set_attribute(model, "max_iter", 0)
    optimize!(model)
    solution_summary(model)
    if is_dual
        @test solver_name(model) ==
              "Dual model with BurerMonteiro with Percival attached"
    else
        @test solver_name(model) == "BurerMonteiro with Percival"
    end
    @test termination_status(model) == MOI.ITERATION_LIMIT
    diff_check(model)
    if !is_dual # See https://github.com/jump-dev/Dualization.jl/issues/195
        @test MOI.supports(
            unsafe_backend(model),
            MOI.RawOptimizerAttribute("max_iter"),
        )
    end
    @test MOI.get(
        unsafe_backend(model),
        MOI.RawOptimizerAttribute("max_iter"),
    ) == 0

    set_attribute(model, "max_iter", 10)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test dual_status(model) == MOI.FEASIBLE_POINT
    @test value(x) ≈ 1
    t = MOI.get(model, MOI.ConstraintDual(), con_ref)
    if !is_dual
        @test t isa LRO.TriangleVectorization
        @test t.matrix isa LRO.Factorization
        @test t.matrix ≈ ones(1, 1)
    end
    @test only(dual(con_ref)) ≈ 1
    solution_summary(model)
    @test objective_value(model) ≈ 1
    @test dual_objective_value(model) ≈ 1
    diff_check(model)
end;

include(joinpath(dirname(@__DIR__), "examples", "maxcut.jl"))
weights = [0 5 7 6; 5 0 0 1; 7 0 0 1; 6 1 1 0];
@testset "Max-CUT $opt" for (is_dual, opt) in [
    (false, LRO.Optimizer),
    (true, dual_optimizer(LRO.Optimizer)),
]
    model = maxcut(weights, opt)
    set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
    set_attribute(model, "sub_solver", Percival.PercivalSolver)
    set_attribute(model, "ranks", [is_dual ? 2 : 3])
    set_attribute(model, "verbose", 2)

    set_attribute(model, "max_iter", 0)
    optimize!(model)
    @test termination_status(model) == MOI.ITERATION_LIMIT
    diff_check(model)

    set_attribute(model, "max_iter", 20)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 18 rtol = 1e-6
    diff_check(model)
end;

@testset "ResultCount" begin
    model = LRO.Optimizer()
    @test MOI.get(model, MOI.ResultCount()) == 0
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION
end

@testset "MOI runtests" begin
    model = LRO.Optimizer()
    MOI.set(
        model,
        MOI.RawOptimizerAttribute("solver"),
        LRO.BurerMonteiro.Solver,
    )
    MOI.set(
        model,
        MOI.RawOptimizerAttribute("sub_solver"),
        Percival.PercivalSolver,
    )
    config = MOI.Test.Config()
    MOI.Test.runtests(model, config; include = ["Silent"])
end;

@testset "No constraints" begin
    model = LRO.Model(
        [spzeros(1, 1)],
        [ones(1, 1) for _ in 1:1, _ in 1:0],
        zeros(0),
        sparsevec(Int[], Float64[], 0),
        sparse(Int[], Int[], Float64[], 0, 0),
        [1],
    )
    @test model.meta.ncon == 0
    @test LRO.norm_jac(model, LRO.MatrixIndex(1)) == 0
    @test isnothing(LRO.buffer_for_jtprod(model, LRO.MatrixIndex(1)))
end;

struct ConvexSolver{T} <: SolverCore.AbstractOptimizationSolver
    model::LRO.Model{T}
    stats::SolverCore.GenericExecutionStats{T,Vector{T},Vector{T},Any}
end

function ConvexSolver(model::LRO.Model)
    stats = SolverCore.GenericExecutionStats(model)
    return ConvexSolver(model, stats)
end

function SolverCore.solve!(::ConvexSolver, ::LRO.Model)
    return
end

function _alloc_schur_complement(model, i, Wi, H, schur_buffer)
    if VERSION < v"1.11"
        return
    end
    LRO.add_schur_complement!(model, i, Wi, H, schur_buffer)
    @test 0 ==
          @allocated LRO.add_schur_complement!(model, i, Wi, H, schur_buffer)
end

function schur_test(model, w, κ)
    schur_buffer = LRO.buffer_for_schur_complement(model, κ)
    jtprod_buffer = LRO.buffer_for_jtprod(model)
    n = model.meta.ncon
    y = rand(n)

    Jv = similar(y)
    vJ = similar(w)
    NLPModels.jprod!(model, w, w, Jv, schur_buffer[1])
    NLPModels.jtprod!(model, w, y, vJ, jtprod_buffer)
    @test dot(Jv, y) ≈ dot(vJ, w)

    H = zeros(n, n)
    H = LRO.schur_complement!(model, w, H, schur_buffer)
    Hy = similar(y)
    LRO.eval_schur_complement!(Hy, model, w, y, schur_buffer[1], jtprod_buffer)
    @test Hy ≈ H * y
    for i in LRO.matrix_indices(model)
        Wi = @inferred w[i]
        _alloc_schur_complement(model, i, Wi, H, schur_buffer)
    end
    for i in LRO.matrix_indices(model)
        ret = LRO.dual_cons!(jtprod_buffer, model, i, y)
        @test ret isa SparseMatrixCSC
    end
    @test LRO.dual_cons(model, LRO.ScalarIndex, y) isa SparseArrays.SparseVector
end

function schur_test(model, κ)
    w = rand(model.meta.nvar)
    W = LRO.VectorizedSolution(w, model.dim)
    for i in LRO.matrix_indices(model)
        W[i] .= W[i] .+ W[i]'
    end
    return schur_test(model, W, κ)
end

@testset "ConvexSolver" begin
    model = maxcut(weights, LRO.Optimizer)
    set_attribute(model, "solver", ConvexSolver)
    b = unsafe_backend(model)
    optimize!(model)
    b.solver.stats.status = :first_order
    @test MOI.get(model, LRO.ConvexTerminationStatus()) == MOI.OPTIMAL
    @test termination_status(model) == MOI.OPTIMAL
    @test MOI.get(model, LRO.Solution()) isa LRO.VectorizedSolution{Float64}
    b.solver.stats.status = :infeasible
    @test termination_status(model) == MOI.DUAL_INFEASIBLE
    @test dual_status(model) == MOI.INFEASIBLE_POINT
    b.solver.stats.status = :unbounded
    @test termination_status(model) == MOI.INFEASIBLE
    @test primal_status(model) == MOI.INFEASIBLE_POINT
    x = LRO.VectorizedSolution(collect(1:b.model.meta.nvar), b.model.dim)
    sim = similar(x)
    @test sim isa typeof(x)
    sim .= x
    @test sim == x
    X = LRO.ShapedSolution(
        Vector(x[LRO.ScalarIndex]),
        [Matrix(x[i]) for i in LRO.matrix_indices(b.model)],
    )
    y = collect(1:b.model.meta.ncon)
    @test NLPModels.jac(b.model, 1, LRO.MatrixIndex(1)) ==
          sparse([1], [1], [-1], 4, 4)
    @test NLPModels.jac(b.model, 1, LRO.ScalarIndex) ==
          sparsevec([1, 2], [-1, 1], 8)
    @test LRO.norm_jac(b.model, LRO.MatrixIndex(1)) == 4
    grad = similar(x)
    NLPModels.grad!(b.model, X, grad)
    @test Vector(grad) == [
        Vector(NLPModels.grad(b.model, LRO.ScalarIndex));
        NLPModels.grad(b.model, LRO.MatrixIndex(1))[:]
    ]
    for xx in [x, X]
        err = LRO.errors(b.solver.model, xx; y, dual_slack = xx, dual_err = xx)
        @test length(err) == 6
        @test err[1] ≈ 4.155017729878046
        @test err[2] ≈ 0.318237296391563
        @test err[3] ≈ 70/9
        @test err[4] ≈ 70/9
        @test err[5] ≈ 0.92
        @test err[6] ≈ 392.0
    end
    schur_test(b.model, 0)
    schur_test(b.model, 1)
    schur_test(b.model, 2)
end
