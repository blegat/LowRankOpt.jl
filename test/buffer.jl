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

# Exercise both sides of `ncon == d`, including overlapping constraint
# patterns. The buffer choice must account for the Schur product, not just
# the cost of filling A(y).
function test_sparse_jtprod_buffer()
    T = Float64
    @testset "d=$d, ncon=$ncon" for (d, ncon) in [(20, 2), (40, 40), (80, 160)]
        A = [
            SparseArrays.sparse([mod1(j, d)], [mod1(j, d)], [T(j)], d, d)
            for _ in 1:1, j in 1:ncon
        ]
        model = LRO.Model(
            [SparseArrays.spzeros(T, d, d)],
            A,
            zeros(T, ncon),
            # `schur_test` exercises the scalar block too.
            SparseArrays.sparsevec([1], T[1], 1),
            SparseArrays.sparse([1, 2], [1, 1], T[1, 1], ncon, 1),
            [d],
        )
        i = LRO.MatrixIndex(1)
        buffer = LRO.buffer_for_jtprod(model, i)
        @test buffer isa SparseArrays.SparseMatrixCSC
        @test SparseArrays.nnz(buffer) == min(d, ncon)
        @test LRO.buffer_for_jtprod(model, i; density = 0) isa Matrix
        @test LRO.buffer_for_jtprod(model, i; work_ratio = 0) isa Matrix
        # Both cutoffs include equality, independently of the other cutoff.
        density = SparseArrays.nnz(buffer) / d^2
        work_ratio = (ncon * d + ncon * SparseArrays.nnz(buffer)) / d^3
        @test LRO.buffer_for_jtprod(model, i; density, work_ratio = Inf) isa
              SparseArrays.SparseMatrixCSC
        @test LRO.buffer_for_jtprod(model, i; density = Inf, work_ratio) isa
              SparseArrays.SparseMatrixCSC
        @test LRO.buffer_for_jtprod(
            model,
            i;
            density = Inf,
            work_ratio = work_ratio / 2,
        ) isa Matrix

        buf = LRO.BufferedModelForSchur(model, 1)
        @test buf.jtprod_buffer[i.value] isa SparseArrays.SparseMatrixCSC
        # A lazy zero matrix contributes nothing, including when only a
        # subset of constraints is requested. It must not use the generic
        # unsafe view, which cannot flatten FillArrays.Zeros matrices.
        V = FillArrays.Zeros{T}(d, d)
        Jv = T.(1:ncon)
        expected_Jv = copy(Jv)
        LRO.add_jprod!(buf, V, Jv, i)
        @test Jv == expected_Jv
        I = [ncon, 1]
        sub_Jv = expected_Jv[I]
        LRO.add_sub_jprod!(buf, i, V, sub_Jv, I)
        @test sub_Jv == expected_Jv[I]
        y = T[isodd(j) ? j : -j for j in 1:ncon]
        expected = sum(A[1, j] * y[j] for j in 1:ncon)
        @test LRO.unsafe_jtprod(buf, y, i) ≈ expected
        @test LRO.unsafe_jtprod(buf, SparseArrays.sparsevec(y), i) ≈ expected
        for κ in 0:2
            schur_test(model, κ)
        end
    end
    return
end

function test_sparse_jtprod_mixed_patterns()
    d = 20
    Z = FillArrays.Zeros(d, d)
    P = SparseArrays.sparse(
        [1, 3, 4, 1, 1],
        [1, 1, 1, 3, 4],
        [1.0, 2.0, 3.0, 2.0, 3.0],
        d,
        d,
    )
    Q = SparseArrays.sparse(
        [3, 5, 1, 1],
        [1, 1, 3, 5],
        [-2.0, 4.0, -2.0, 4.0],
        d,
        d,
    )
    # A zero constraint after P must preserve the merged sparse pattern.
    # Q skips row 1, then row 4, in the merged pattern's first column.
    # Its overlap with P cancels numerically, but must remain in the pattern.
    A = Matrix{Union{typeof(Z),typeof(P)}}(undef, 1, 4)
    A[1, :] = [Z, P, Z, Q]
    original = deepcopy(A)
    model = LRO.Model(
        [SparseArrays.spzeros(d, d)],
        A,
        zeros(4),
        SparseArrays.sparsevec([1], [1.0], 1),
        SparseArrays.sparse([1, 2], [1, 1], [1.0, 1.0], 4, 1),
        [d],
    )
    i = LRO.MatrixIndex(1)
    buf = LRO.BufferedModelForSchur(model, 1)
    buffer = buf.jtprod_buffer[i.value]
    @test buffer isa SparseArrays.SparseMatrixCSC
    @test SparseArrays.nnz(buffer) == 7
    rows, cols = copy(buffer.rowval), copy(buffer.colptr)
    # Repeated calls check that accumulation clears values without losing
    # the merged pattern, including entries cancelled by a previous call.
    for y in ([1.0, 1.0, 1.0, 1.0], [2.0, -3.0, 4.0, 5.0], zeros(4))
        expected = y[2] * P + y[4] * Q
        for weights in (y, SparseArrays.sparsevec(y))
            @test LRO.unsafe_jtprod(buf, weights, i) ≈ expected
            @test buffer.rowval == rows
            @test buffer.colptr == cols
        end
    end
    @test A == original
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
