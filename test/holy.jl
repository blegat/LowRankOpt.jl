# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

include("diff_check.jl")
include(joinpath(dirname(@__DIR__), "examples", "holy_model.jl"))

import Random
using Dualization
import Percival

function test_holy(; is_dual, low_rank, square_scalars, n = 10)
    opt = LRO.Optimizer
    if is_dual
        opt = dual_optimizer(opt)
    end
    Random.seed!(0)
    A = data(n)
    if low_rank
        model = holy_lowrank(A)
    else
        model = holy_classic(A)
    end
    set_optimizer(model, opt)
    set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
    set_attribute(model, "sub_solver", Percival.PercivalSolver)
    set_attribute(model, "ranks", [is_dual ? 2 : 3])
    set_attribute(model, "verbose", 2)
    set_attribute(model, "square_scalars", square_scalars)

    set_attribute(model, "max_iter", 0)
    optimize!(model)
    @test termination_status(model) == MOI.ITERATION_LIMIT
    diff_check(model)
    b = _backend(model)
    T = Float64
    F = if low_rank
        LRO.Factorization{T,SparseVector{T,Int},LRO.One{T}}
    else
        SparseMatrixCSC{T,Int}
    end
    @test b.model.C isa Vector{F}
    @test b.model.A isa Matrix{F}
end

@testset "Holy low-rank ? $low_rank" for low_rank in [false, true]
    @testset "Square scalars ? $square_scalars" for square_scalars in [false, true]
        test_holy(; is_dual = true, low_rank, square_scalars)
    end
end;
