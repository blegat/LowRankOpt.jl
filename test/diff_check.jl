# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using Test
using LinearAlgebra
using FillArrays
using JuMP
import SolverCore
using Dualization
import LowRankOpt as LRO

function _test_vecprod(f, len, J; tol = 1e-6)
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
    gg = NLPModels.grad(model, x)
    @test gg ≈ g rtol = tol atol = tol
    v = rand(length(x))
    @test dot(g, v) ≈ LRO.BurerMonteiro.gprod(model, x, v) rtol = tol atol = tol
end

function jac_check(model, x; kws...)
    f(x) = NLPModels.cons(model, x)
    J = FiniteDiff.finite_difference_jacobian(f, x)
    @testset "jprod" begin
        _test_vecprod(
            v -> NLPModels.jprod(model, x, v),
            model.meta.nvar,
            J;
            kws...,
        )
    end
    @testset "jtprod" begin
        _test_vecprod(
            v -> NLPModels.jtprod(model, x, v),
            model.meta.ncon,
            J';
            kws...,
        )
        v = rand(length(x))
        y = rand(model.meta.ncon)
        # Warmup
        NLPModels.jtprod!(model, x, y, v)
        @test 0 == @allocated NLPModels.jtprod!(model, x, y, v)
    end
end

function hess_check(model, x; kws...)
    obj_weight = rand()
    y = rand(model.meta.ncon)
    f(x) =
        obj_weight * NLPModels.obj(model, x) - dot(y, NLPModels.cons(model, x))
    J = FiniteDiff.finite_difference_hessian(f, x)
    return _test_vecprod(
        v -> NLPModels.hprod(model, x, y, v; obj_weight),
        model.meta.nvar,
        J;
        kws...,
    )
end

using NLPModelsTest

function _backend(model)
    b = unsafe_backend(model)
    if b isa DualOptimizer
        b = b.dual_problem.dual_model.model.optimizer
    end
    return b
end

function diff_check(model::NLPModels.AbstractNLPModel)
    x = rand(model.meta.nvar)
    @testset "Gradient" begin
        grad_check(model, x)
        @test isempty(NLPModelsTest.gradient_check(model; x))
    end
    @testset "Jacobian" begin
        jac_check(model, x)
    end
    @testset "Hessian" begin
        hess_check(model, x)
    end
end

function diff_check(model::JuMP.AbstractModel)
    b = _backend(model)
    return diff_check(b.solver.model)
end

struct ConvexSolver{T} <: SolverCore.AbstractOptimizationSolver
    model::LRO.DualModel{T}
    stats::SolverCore.GenericExecutionStats{T,Vector{T},Vector{T},Any}
end

function ConvexSolver(src::Union{LRO.Model,LRO.DualModel})
    model = LRO.dual_model(src)
    stats = SolverCore.GenericExecutionStats(model)
    return ConvexSolver(model, stats)
end

function LRO.MOI.get(solver::ConvexSolver, ::LRO.DualSolution)
    return LRO.VectorizedConeVector(solver.stats.solution, solver.model.dim)
end

function SolverCore.solve!(::ConvexSolver, ::LRO.Model)
    return
end

function _alloc_schur_complement(model, i, Wi, H)
    if VERSION < v"1.11"
        return
    end
    LRO.add_schur_complement!(model, i, Wi, H)
    @test 0 == @allocated LRO.add_schur_complement!(model, i, Wi, H)
end

function schur_test(workspace::LRO.ModelWorkspace, w)
    model = workspace.model
    T = eltype(model.b)
    n = model.meta.nvar
    y = rand(T, n)
    idx = @inferred LRO.matrix_indices(model)
    if !isempty(idx)
        @inferred first(idx)
        @inferred idx[1]
    end

    # Native J maps free variables to cone entries; J' maps them back.
    Jv, vJ = similar(w), similar(y)
    LRO.jprod!(workspace, y, Jv)
    LRO.jtprod!(workspace, w, vJ)
    @test Vector(Jv) ≈ model.jacobian * y
    @test vJ ≈ model.jacobian' * Vector(w)
    @test dot(Jv, w) ≈ dot(y, vJ)
    @test 0 == @allocated LRO.jprod!(workspace, y, Jv)
    @test 0 == @allocated LRO.jtprod!(workspace, w, vJ)
    LRO.cons!(workspace, y, Jv)
    @test Vector(Jv) ≈ NLPModels.cons(model, y)

    H = zeros(T, n, n)
    LRO.schur_complement!(workspace, w, H)
    Hy = similar(y)
    LRO.eval_schur_complement!(workspace, w, y, Hy)
    @test Hy ≈ H * y
    if T == Float64
        @test 0 == @allocated LRO.eval_schur_complement!(workspace, w, y, Hy)
    end
    # Reuse the workspaces for another right-hand side, then assemble again.
    LRO.eval_schur_complement!(workspace, w, -y, Hy)
    @test Hy ≈ -(H * y)
    H_again = similar(H)
    LRO.schur_complement!(workspace, w, H_again)
    @test H_again ≈ H
    for i in idx
        Wi = @inferred w[i]
        _alloc_schur_complement(workspace, i, Wi, H)
    end
    for i in idx
        @test workspace.jprod_buffer[i.value] isa
              Union{FillArrays.Zeros,Matrix,SparseArrays.SparseMatrixCSC}
        ret = LRO.unsafe_jprod(workspace, y, i)
        @test ret === workspace.jprod_buffer[i.value]
        rows = (model.dim.offsets[i.value]+1):model.dim.offsets[i.value+1]
        expected = reshape(model.jacobian[rows, :] * y, size(model.C[i.value]))
        @test ret ≈ expected
        ret = LRO.unsafe_cons(workspace, y, i)
        @test ret isa
              Union{Matrix,SparseArrays.SparseMatrixCSC,FillArrays.Zeros}
        @test ret ≈ model.C[i.value] + expected
        if workspace.jprod_buffer[i.value] isa FillArrays.Zeros
            @test ret === model.C[i.value]
        elseif model.C[i.value] isa FillArrays.Zeros
            @test ret === workspace.jprod_buffer[i.value]
        end
    end
    # Preconditioners also call the block product with sparse directions.
    y_sparse = SparseArrays.sparsevec(y)
    for i in idx
        dense = copy(LRO.unsafe_jprod(workspace, y, i))
        @test LRO.unsafe_jprod(workspace, y_sparse, i) ≈ dense
    end
    ns = LRO.num_scalars(model)
    scalar = ones(T, ns)
    LRO.jprod!(workspace, y, scalar, LRO.ScalarIndex)
    @test scalar ≈ model.jacobian[1:ns, :] * y
    LRO.jtprod!(workspace, w[LRO.ScalarIndex], vJ, LRO.ScalarIndex)
    @test vJ ≈ model.jacobian[1:ns, :]' * w[LRO.ScalarIndex]
    LRO.cons!(workspace, y, scalar, LRO.ScalarIndex)
    @test scalar ≈ model.constant[1:ns] + model.jacobian[1:ns, :] * y
    return
end

function schur_test(workspace::LRO.ModelWorkspace)
    model = workspace.model
    w = rand(eltype(model.b), model.meta.ncon)
    W = LRO.VectorizedConeVector(w, model.dim)
    for i in LRO.matrix_indices(model)
        W[i] .= W[i] .+ W[i]'
    end
    return schur_test(workspace, W)
end

function schur_test(model::LRO.Model, κ)
    return schur_test(
        LRO.ModelWorkspace(
            model;
            dense_condition = LRO.And(LRO.MinNonzeros(κ), LRO.DenseCost(100)),
        ),
    )
end

schur_test(model::LRO.DualModel, κ) = schur_test(LRO.primal_model(model), κ)

function schur_test(model::JuMP.AbstractModel, κ)
    return schur_test(_backend(model).model, κ)
end

function LRO.MOI.get(solver::ConvexSolver, ::LRO.PrimalSolution)
    return solver.stats.multipliers
end
LRO.MOI.get(solver::ConvexSolver, ::LRO.DualObjective) = -solver.stats.objective
