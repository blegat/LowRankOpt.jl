# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using Test
using LinearAlgebra
using FillArrays
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
    model::LRO.Model{T}
    stats::SolverCore.GenericExecutionStats{T,Vector{T},Vector{T},Any}
end

function ConvexSolver(model::LRO.Model)
    stats = SolverCore.GenericExecutionStats(model)
    return ConvexSolver(model, stats)
end

function LRO.MOI.get(solver::ConvexSolver, ::LRO.Solution)
    return LRO.VectorizedSolution(solver.stats.solution, solver.model.dim)
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

function schur_test(model::LRO.BufferedModelForSchur{T}, w) where {T}
    n = model.meta.ncon
    y = rand(T, n)

    idx = LRO.matrix_indices(model)
    @test NLPModels.obj(model, w) ≈
          dot(LRO.grad(model, LRO.ScalarIndex), w[LRO.ScalarIndex]) +
          dot(LRO.grad.(model, idx), getindex.(Ref(w), idx))
    @test LRO.dual_obj(model, y) ≈ dot(LRO.cons_constant(model), y)

    @test LRO.jac(model, 1, LRO.ScalarIndex) ==
          LRO.jac(model.model, 1, LRO.ScalarIndex)
    @test LRO.norm_jac.(model, idx) == LRO.norm_jac.(model.model, idx)
    @test LRO.side_dimension.(model, idx) ==
          LRO.side_dimension.(model.model, idx)
    if !isempty(idx)
        @test LRO.side_dimension.([model], idx[1]) ==
              [LRO.side_dimension(model, idx[1])]
    end

    Jv = similar(y)
    vJ = similar(w)
    NLPModels.jprod!(model, w, w, Jv)
    NLPModels.jtprod!(model, w, y, vJ)
    @test dot(Jv, y) ≈ dot(vJ, w)

    H = zeros(n, n)
    H = LRO.schur_complement!(model, w, H)
    Hy = similar(y)
    LRO.eval_schur_complement!(model, w, y, Hy)
    @test Hy ≈ H * y
    for i in LRO.matrix_indices(model)
        Wi = @inferred w[i]
        _alloc_schur_complement(model, i, Wi, H)
    end
    for i in LRO.matrix_indices(model)
        @test model.jtprod_buffer[i.value] isa
              Union{FillArrays.Zeros,SparseArrays.SparseMatrixCSC}
        ret = LRO.unsafe_jtprod(model, y, i)
        @test ret === model.jtprod_buffer[i.value]
        ret = LRO.unsafe_dual_cons(model, y, i)
        @test ret isa Union{SparseArrays.SparseMatrixCSC,FillArrays.Zeros}
        Aty = sum(model.model.A[i.value, j] * y[j] for j in 1:model.meta.ncon)
        @test ret ≈ model.model.C[i.value] - Aty
        if Aty isa FillArrays.Zeros
            @test ret === model.model.C[i.value]
        elseif model.model.C[i.value] isa FillArrays.Zeros
            @test ret === model.jtprod_buffer[i.value]
        end
    end
    dcons = ones(LRO.num_scalars(model))
    LRO.dual_cons!(model, y, dcons, LRO.ScalarIndex)
    @test dcons ≈ model.model.d_lin - model.model.C_lin' * y
end

function schur_test(model::LRO.BufferedModelForSchur{T}) where {T}
    w = rand(T, model.meta.nvar)
    W = LRO.VectorizedSolution(w, model.model.dim)
    for i in LRO.matrix_indices(model)
        W[i] .= W[i] .+ W[i]'
    end
    return schur_test(model, W)
end

function schur_test(model::LRO.Model, κ)
    return schur_test(LRO.BufferedModelForSchur(model, κ))
end

function schur_test(model::JuMP.AbstractModel, κ)
    b = _backend(model)
    return schur_test(b.solver.model, κ)
end
