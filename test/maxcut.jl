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
