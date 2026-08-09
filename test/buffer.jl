module TestBuffer

import FillArrays, SparseArrays
using JuMP, Dualization
include("diff_check.jl")

# Test with zero Ai matrices
function _test_zero_Ai(all_zero::Bool, matrix_in_objective::Bool)
    model = Model(dual_optimizer(LRO.Optimizer))
    @variable(model, x[1:2] in MOI.Nonnegatives(2))
    @variable(model, X[1:2, 1:2] in PSDCone())
    @constraint(model, sum(x) == 1)
    @constraint(model, 2sum(x) == 2)
    if !all_zero
        @constraint(model, sum(X) == 2)
    end
    @constraint(model, x[1] - x[2] == 1)
    if matrix_in_objective
        @objective(model, Max, x[1] + X[1, 2] - X[1, 1])
    else
        @objective(model, Max, x[1])
    end
    set_attribute(model, "solver", ConvexSolver)
    optimize!(model)
    b = _backend(model)
    T = Float64
    Z = FillArrays.Zeros{T,2,Tuple{Base.OneTo{Int},Base.OneTo{Int}}}
    S = SparseArrays.SparseMatrixCSC{T,Int}
    @test b.model.C isa Vector{matrix_in_objective ? S : Z}
    if all_zero
        @test b.model.A isa Matrix{Z}
    else
        @test b.model.A isa Matrix{Union{Z,S}}
    end
    @test b.model.A[1] isa Z
    @test b.model.A[2] isa Z
    if all_zero
        @test b.model.A[3] isa Z
    else
        @test b.model.A[3] isa S
        @test b.model.A[4] isa Z
    end
    buf = LRO.BufferedModelForSchur(b.model, 1)
    for A in b.model.A
        if all_zero
            @test buf.jtprod_buffer[] isa LRO.FillArrays.Zeros
        else
            @test buf.jtprod_buffer[] !== A
        end
    end
    for κ in 0:5
        schur_test(model, κ)
    end
end

function test_zero_Ai()
    @testset "all_zero=$all_zero" for all_zero in [false, true]
        @testset "matrix_in_objective=$matrix_in_objective" for matrix_in_objective in
                                                                [false, true]
            _test_zero_Ai(all_zero, matrix_in_objective)
        end
    end
    return
end

# `buffer_for_jtprod` keeps `∑ⱼ Aᵢⱼ yⱼ` sparse when the block is wide
# compared to the number of constraints and the merged pattern is sparse, see
# `LRO.DENSE_JTPROD_DENSITY`. No problem in the rest of this suite reaches
# that path -- they all have `ncon >= side_dimension` -- so build one here.
function test_sparse_jtprod_buffer()
    T = Float64
    d, ncon = 20, 2
    @assert ncon < d
    A = [SparseArrays.sparse([j], [j], [T(j)], d, d) for _ in 1:1, j in 1:ncon]
    model = LRO.Model(
        [SparseArrays.spzeros(T, d, d)],
        A,
        zeros(T, ncon),
        # `schur_test` exercises the scalar block too, so give it one variable.
        SparseArrays.sparsevec([1], T[1], 1),
        SparseArrays.sparse([1, 2], [1, 1], T[1, 1], ncon, 1),
        [d],
    )
    i = LRO.MatrixIndex(1)
    buffer = LRO.buffer_for_jtprod(model, i)
    @test buffer isa SparseArrays.SparseMatrixCSC
    @test SparseArrays.nnz(buffer) <= LRO.DENSE_JTPROD_DENSITY * d^2

    buf = LRO.BufferedModelForSchur(model, 1)
    @test buf.jtprod_buffer[i.value] isa SparseArrays.SparseMatrixCSC
    y = T[2, -3]
    expected = sum(A[1, j] * y[j] for j in 1:ncon)
    @test LRO.unsafe_jtprod(buf, y, i) ≈ expected
    # Loraine's `H_alpha` preconditioner passes a sparse `y`, see `diff_check`.
    @test LRO.unsafe_jtprod(buf, SparseArrays.sparsevec(y), i) ≈ expected

    for κ in 0:2
        schur_test(model, κ)
    end
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

TestBuffer.runtests()
