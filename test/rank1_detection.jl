# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.
#
# Tests for `rank_one`/`detect_rank_one` and the `detect_rank_one` Optimizer
# option that converts sparse rank-one constraint matrices to rank-one
# `Factorization`s (so they hit the rank-1 Schur specialization in `schur.jl`).

module TestRank1Detection

using Test
using LinearAlgebra
using SparseArrays
import FillArrays
import MathOptInterface as MOI
using JuMP
import LowRankOpt as LRO

include("diff_check.jl")

function test_rank_one_unit_vector()
    A = sparse([3], [3], [1.0], 4, 4) # e₃e₃'
    f = LRO.rank_one(A)
    @test f isa LRO.Factorization
    @test Matrix(f) == Matrix(A)
    @test only(f.scaling) == 1.0
    return
end

function test_rank_one_general()
    b = [1.0, 2.0, 0.0, 3.0]
    M = sparse(2.0 .* (b * b'))
    f = LRO.rank_one(M)
    @test f isa LRO.Factorization
    @test Matrix(f) ≈ Matrix(M)
    return
end

function test_rank_one_indefinite()
    b = [1.0, 2.0, 0.0, 3.0]
    M = sparse(-1.5 .* (b * b'))
    f = LRO.rank_one(M)
    @test f isa LRO.Factorization
    @test Matrix(f) ≈ Matrix(M)
    @test only(f.scaling) < 0 # negative scaling propagates
    return
end

function test_rank_one_zero()
    @testset "$(typeof(Z))" for Z in
                                [spzeros(3, 3), FillArrays.Zeros{Float64}(3, 3)]
        f = LRO.rank_one(Z)
        @test f isa LRO.Factorization
        @test iszero(Matrix(f))
    end
    return
end

function test_rank_one_rejects_higher_rank()
    @test isnothing(LRO.rank_one(sparse([2.0 1.0; 1.0 2.0]))) # rank 2
    # Symmetric, nonzero, but zero diagonal ⇒ not rank one.
    @test isnothing(LRO.rank_one(sparse([1, 2], [2, 1], [1.0, 1.0], 2, 2)))
    return
end

function test_rank_one_rejects_extra_nonzero()
    # `e₁e₁'` plus a stray entry is not rank one.
    A = sparse([1, 1], [1, 2], [1.0, 1.0], 2, 2)
    @test isnothing(LRO.rank_one(A))
    return
end

function test_rank_one_from_factorization()
    g = LRO.positive_semidefinite_factorization([1.0, 2.0, 3.0])
    f = LRO.rank_one(g)
    @test f isa LRO.Factorization{Float64,<:SparseVector,<:AbstractArray{Float64,0}}
    @test Matrix(f) ≈ Matrix(g)
    return
end

function test_detect_rank_one_all()
    arr = [sparse([j], [j], [1.0], 4, 4) for i in 1:1, j in 1:4]
    R = LRO.detect_rank_one(arr)
    @test !isnothing(R)
    @test isconcretetype(eltype(R))
    @test eltype(R) <: LRO.Factorization
    @test all(i -> Matrix(R[i]) == Matrix(arr[i]), eachindex(arr))
    return
end

function test_detect_rank_one_mixed()
    arr = Matrix{SparseMatrixCSC{Float64,Int}}(undef, 1, 2)
    arr[1, 1] = sparse([1], [1], [1.0], 2, 2)
    arr[1, 2] = sparse([2.0 1.0; 1.0 2.0]) # rank 2
    @test isnothing(LRO.detect_rank_one(arr))
    return
end

function test_detect_rank_one_with_zeros()
    Z = FillArrays.Zeros{Float64}(2, 2)
    S = sparse([1], [1], [1.0], 2, 2)
    arr = Matrix{Union{typeof(Z),typeof(S)}}(undef, 1, 2)
    arr[1, 1] = Z
    arr[1, 2] = S
    R = LRO.detect_rank_one(arr)
    @test !isnothing(R)
    @test iszero(Matrix(R[1, 1]))
    @test Matrix(R[1, 2]) == Matrix(S)
    return
end

function test_detect_rank_one_empty()
    @test isnothing(LRO.detect_rank_one(Matrix{SparseMatrixCSC{Float64,Int}}(
        undef,
        0,
        0,
    )))
    return
end

# A tiny max-cut-like SDP in SDPA format whose data matrices `Fⱼ = eⱼeⱼ'` are
# rank one, written to a temporary file.
const _SDPA = """
2
1
2
1.0 1.0
0 1 1 1 2.0
0 1 2 2 2.0
0 1 1 2 1.0
1 1 1 1 1.0
2 1 2 2 1.0
"""

function _build(detect)
    path = tempname() * ".dat-s"
    write(path, _SDPA)
    src = MOI.FileFormats.SDPA.Model{Float64}()
    MOI.read_from_file(src, path)
    rm(path)
    model = GenericModel{Float64}(LRO.Optimizer{Float64})
    LRO.Bridges.add_all_bridges(backend(model).optimizer, Float64)
    MOI.copy_to(backend(model), src)
    set_attribute(model, "solver", ConvexSolver)
    set_attribute(model, "detect_rank_one", detect)
    optimize!(model) # `ConvexSolver` is a no-op: this just triggers `copy_to`
    return unsafe_backend(model).model
end

function test_option_default_keeps_sparse()
    model = _build(false)
    @test eltype(model.A) <: SparseMatrixCSC
    return
end

function test_option_converts_to_rank_one()
    sparse_model = _build(false)
    rank_one_model = _build(true)
    @test eltype(rank_one_model.A) <: LRO.Factorization
    # `C` (the objective block) is not rank one, so it stays as is.
    @test eltype(rank_one_model.C) <: SparseMatrixCSC
    # The conversion is exact: same matrices, different representation.
    @test size(rank_one_model.A) == size(sparse_model.A)
    @test all(
        i -> Matrix(rank_one_model.A[i]) == Matrix(sparse_model.A[i]),
        eachindex(sparse_model.A),
    )
    return
end

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
end

end

TestRank1Detection.runtests()
