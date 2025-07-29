# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using Test
using LinearAlgebra
using SparseArrays
using JuMP
import LowRankOpt as LRO
using Dualization
import SolverCore
import Percival

include("diff_check.jl")

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

@testset "Trace" begin
    T = Float64
    model = Model(LRO.Optimizer)
    cone = MOI.PositiveSemidefiniteConeTriangle(2)
    factors = LRO.TriangleVectorization.(
        LRO.positive_semidefinite_factorization.([T[1, 0], T[0, 1]]),
    )
    set = LRO.LinearCombinationInSet{LRO.WITH_SET}(cone, factors)
    set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
    set_attribute(model, "sub_solver", Percival.PercivalSolver)
    set_attribute(model, "ranks", [2])
    @variable(model, x)
    @variable(model, y)
    @constraint(model, [-x, -x, 3, 1, 4] in set)
    @constraint(model, x >= y)

    set_attribute(model, "max_iter", 0)
    optimize!(model)
    @test termination_status(model) == MOI.ITERATION_LIMIT
    nlp = unsafe_backend(model).model;
    @test nlp.C isa Vector{SparseMatrixCSC{T,Int}}
    @test nlp.C[1] == [3 1; 1 4]
    @test nlp.A isa Matrix{
        Union{
            LRO.FillArrays.Zeros{T,2,Tuple{Base.OneTo{Int},Base.OneTo{Int}}},
            LRO.Factorization{T,Matrix{T},Vector{T}},
        },
    }
    @test nlp.A[1].factor == Matrix(I, 2, 2)
    @test nlp.A[1].scaling == [1, 1]
    @test nlp.A[2] === LRO.FillArrays.Zeros{T}(2, 2)
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

@testset "Fallback" begin
    @test LRO.Optimizer() isa LRO.Optimizer{Float64}
end
