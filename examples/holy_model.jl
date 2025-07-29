# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# This is a model of a self-avoiding chain (e.g., a polymer) in arbitrary dimensions
# See https://github.com/jump-dev/JuMP.jl/issues/4011

using LinearAlgebra, SparseArrays
using JuMP
import LowRankOpt as LRO

function data(i, j, n)
    A = zeros(n, n)
    A[i, j] = A[j, i] = 1
    return A
end

# pick one random pair to be close
function data(n)
    i = rand(1:(n-1))
    j = rand((i+1):n)
    return data(i, j, n)
end

function holy_classic(A)
    n = LinearAlgebra.checksquare(A)
    model = Model()
    @variable(model, XX[1:n, 1:n], PSD)    # modeling the Gram matrix, XX[i, j] = X[:, i]'X[:, j]
    # Impose a penalty on distance between the paired nodes
    @objective(
        model,
        Min,
        sum(
            A[i, j] * (XX[i, i] + XX[j, j] - XX[i, j] - XX[j, i]) for
            i in 1:(n-1), j in (i+1):n
        )
    )
    # Link successive nodes together, with an edge length of 1
    @constraint(
        model,
        [i = 1:(n-1)],
        XX[i, i] + XX[i+1, i+1] - XX[i, i+1] - XX[i+1, i] == 1
    )
    # Avoid self-intersections
    @constraint(
        model,
        [i = 1:(n-2), j = (i+2):n],
        XX[i, i] + XX[j, j] - XX[i, j] - XX[j, i] >= 1
    )
    return model
end

# Our constraints are low-rank. The matrix `[1 -1; -1 1]` is equal
# to `[1, -1] * [1, -1]'` so it is rank-1.
# So it can also use a rank-1 formulation.

function _factor(i, j, n)
    F = sparsevec([i, j], Float64[1, -1], n)
    return LRO.positive_semidefinite_factorization(F)
end

function holy_lowrank(A)
    n = LinearAlgebra.checksquare(A)
    model = Model()
    set = LRO.SetDotProducts{LRO.WITHOUT_SET}(
        MOI.PositiveSemidefiniteConeTriangle(n),
        LRO.TriangleVectorization.([
            _factor(i, j, n) for i in 1:(n-1) for j in (i+1):n
        ]),
    )
    @variable(model, x[1:MOI.dimension(set)] in set)
    Δ = Matrix{VariableRef}(undef, n, n)
    k = 0
    for i in 1:(n-1)
        for j in (i+1):n
            k += 1
            Δ[i, j] = x[k]
        end
    end
    # Impose a penalty on distance between the paired nodes
    @objective(
        model,
        Min,
        sum(A[i, j] * Δ[i, j] for i in 1:(n-1), j in (i+1):n)
    )
    # Link successive nodes together, with an edge length of 1
    @constraint(model, [i = 1:(n-1)], Δ[i, i+1] == 1)
    # Avoid self-intersections
    @constraint(model, [i = 1:(n-2), j = (i+2):n], Δ[i, j] >= 1)
    return model
end
