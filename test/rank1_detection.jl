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
    @test f isa
          LRO.Factorization{Float64,<:SparseVector,<:AbstractArray{Float64,0}}
    @test Matrix(f) ≈ Matrix(g)
    return
end

function test_detect_rank_one_all()
    arr = [sparse([j], [j], [1.0], 4, 4) for i in 1:1, j in 1:4]
    R = LRO.detect_rank_one(arr)
    @test !isnothing(R)
    @test isconcretetype(eltype(only(R).values))
    @test eltype(only(R).values) <: LRO.Factorization
    @test all(
        j -> Matrix(only(R).values[j, 1]) == Matrix(arr[1, j]),
        axes(arr, 2),
    )
    return
end

function test_detect_rank_one_mixed()
    arr = Matrix{SparseMatrixCSC{Float64,Int}}(undef, 1, 2)
    arr[1, 1] = sparse([1], [1], [1.0], 2, 2)
    arr[1, 2] = sparse([2.0 1.0; 1.0 2.0]) # rank 2
    components = LRO.detect_rank_one(arr)
    @test length(components) == 2
    @test all(
        j ->
            sum(component.values[j, 1] for component in components) ≈ arr[1, j],
        1:2,
    )
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
    @test iszero(Matrix(only(R).values[1, 1]))
    @test Matrix(only(R).values[2, 1]) == Matrix(S)
    return
end

function test_rank_one_unsupported_type()
    @test isnothing(LRO.rank_one(Diagonal([1.0, 2.0])))
    return
end

# Detected rank-one matrices have sparse factors, they are used both by the
# rank-one Schur kernel and by the sparse Jacobian buffer.
function test_detected_schur_complement()
    d, n = 4, 3
    us = [sparsevec([j, j + 1], [1.0, j], d) for j in 1:n]
    arr = [sparse(u * u') for _ in 1:1, u in us]
    C = [spzeros(d, d)]
    b = zeros(n)
    d_lin = sparsevec(Int64[], Float64[], 0)
    C_lin = spzeros(Float64, Int64, n, 0)
    model = LRO.Model(C, LRO.detect_rank_one(arr), b, d_lin, C_lin, Int64[d])
    @test eltype(only(model.A).values) <: LRO.Factorization
    reference = LRO.Model(C, arr, b, d_lin, C_lin, Int64[d])
    buffer = LRO.BufferedModelForSchur(model, 0)
    reference_buffer = LRO.BufferedModelForSchur(reference, 0)
    W = Matrix(Symmetric(ones(d, d) + d * I))
    i = LRO.MatrixIndex(1)
    H = zeros(n, n)
    LRO.add_schur_complement!(buffer, i, W, H)
    expected = zeros(n, n)
    LRO.add_schur_complement!(reference_buffer, i, W, expected)
    @test H ≈ expected
    y = [1.0, -2.0, 3.0]
    @test LRO.unsafe_jtprod(buffer, y, i) ≈
          LRO.unsafe_jtprod(reference_buffer, y, i)
    return
end

function test_detect_rank_one_empty()
    @test isnothing(
        LRO.detect_rank_one(Matrix{SparseMatrixCSC{Float64,Int}}(undef, 0, 0)),
    )
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
    @test eltype(only(model.A).values) <: SparseMatrixCSC
    return
end

function test_option_converts_to_rank_one()
    sparse_model = _build(false)
    rank_one_model = _build(true)
    @test eltype(only(rank_one_model.A).values) <: LRO.Factorization
    # `C` (the objective block) is not rank one, so it stays as is.
    @test eltype(rank_one_model.C) <: SparseMatrixCSC
    # The conversion is exact: same matrices, different representation.
    @test size(only(rank_one_model.A)) == size(only(sparse_model.A))
    @test all(
        j ->
            Matrix(LRO.jac(rank_one_model, j, LRO.MatrixIndex(1))) ==
            Matrix(LRO.jac(sparse_model, j, LRO.MatrixIndex(1))),
        1:sparse_model.meta.ncon,
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
