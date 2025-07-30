# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

include("diff_check.jl")
include(joinpath(dirname(@__DIR__), "examples", "maxcut.jl"))

weights = [0 5 7 6; 5 0 0 1; 7 0 0 1; 6 1 1 0];

function test_maxcut(; is_dual, sparse, vector)
    opt = LRO.Optimizer
    if is_dual
        opt = dual_optimizer(opt)
    end
    model = maxcut(weights, opt; sparse, vector)
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
    T = Float64
    if is_dual
        unsafe_model = _backend(model)
        lro_model = unsafe_model.model
        solver = unsafe_model.solver
        @test lro_model.C isa Vector{SparseMatrixCSC{T,Int64}}
        F = if sparse
            if vector
                SparseVector{T,Int}
            else
                SparseMatrixCSC{T,Int}
            end
        else
            if vector
                Vector{T}
            else
                Matrix{T}
            end
        end
        D = vector ? LRO.One{Float64} : LRO.Ones{Float64}
        @test lro_model.A isa Matrix{LowRankOpt.Factorization{Float64,F,D}}
    else
        lro_model = unsafe_backend(model).model
        @test lro_model.C isa Vector{SparseMatrixCSC{T,Int64}}
        @test lro_model.A isa Matrix{SparseMatrixCSC{T,Int64}}
        solver = unsafe_backend(model).solver
        LRO.BurerMonteiro.set_rank!(solver.model, LRO.MatrixIndex(1), 4)
        @test solver.model.dim.ranks == [4]
        @test solver.model.dim.offsets == [8, 24]
        @test length(solver.model.dim) == 24
        @test length(solver.model.meta.x0) == 24
    end
    i = LRO.MatrixIndex(1)
    LRO.BurerMonteiro.set_rank!(solver.model, LRO.MatrixIndex(1), 1)
    diff_check(model)
    X = LRO.positive_semidefinite_factorization(rand(4))
    JtV = LRO.positive_semidefinite_factorization(ones(4))
    NLPModels.grad!(solver.model, X, JtV, i)
    @test JtV.factor ≈ 2lro_model.C[] * X.factor
    fill!(JtV.factor, 0.0)
    y = rand(lro_model.meta.ncon)
    LRO.BurerMonteiro.add_jtprod!(solver.model, X, y, JtV, i)
    if is_dual
        @test JtV.factor ≈ 2y .* X.factor
    end
end

@testset "Max-CUT dual ? $is_dual" for is_dual in [false, true]
    @testset "Sparse ? $sparse" for sparse in [false, true]
        @testset "Vector ? $vector" for vector in [false, true]
            test_maxcut(; is_dual, sparse, vector)
        end
    end
end;

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
        ret = LRO.dual_cons!(model, y, i)
        @test ret isa SparseMatrixCSC
    end
    dcons = ones(LRO.num_scalars(model))
    LRO.dual_cons!(model, y, dcons, LRO.ScalarIndex)
    @test dcons ≈ model.model.d_lin - model.model.C_lin' * y
end

function schur_test(model::LRO.Model{T}, κ) where {T}
    w = rand(T, model.meta.nvar)
    W = LRO.VectorizedSolution(w, model.dim)
    for i in LRO.matrix_indices(model)
        W[i] .= W[i] .+ W[i]'
    end
    return schur_test(LRO.BufferedModelForSchur(model, κ), W)
end

@testset "ConvexSolver $T" for T in [Float32, Float64]
    model = maxcut(T.(weights), LRO.Optimizer{T})
    set_attribute(model, "solver", ConvexSolver)
    b = unsafe_backend(model)
    optimize!(model)
    b.solver.stats.status = :first_order
    @test MOI.get(model, LRO.ConvexTerminationStatus()) == MOI.OPTIMAL
    @test termination_status(model) == MOI.OPTIMAL
    @test MOI.get(model, LRO.Solution()) isa LRO.VectorizedSolution{T}
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
end;
