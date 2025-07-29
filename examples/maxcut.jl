# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using LinearAlgebra, SparseArrays, JuMP, LowRankOpt
import LowRankOpt as LRO

function e_i(T, i, n; sparse, vector)
    if sparse
        if vector
            return sparsevec([i], [one(T)], n)
        else
            return SparseArrays.sparse([i], [1], [one(T)], n, 1)
        end
    else
        if vector
            ei = zeros(T, n)
        else
            ei = zeros(T, n, 1)
        end
        ei[i] = 1
        return ei
    end
end

function maxcut_objective(weights)
    L = Diagonal(dropdims(sum(weights, dims = 2), dims = 2)) - weights
    return L / 4
end

function maxcut_set(N; sparse = true, T = Float64, vector = true)
    cone = MOI.PositiveSemidefiniteConeTriangle(N)
    factors = LRO.TriangleVectorization.(
        LRO.positive_semidefinite_factorization.(
            e_i.(T, 1:N, N; sparse, vector),
        ),
    )
    return LRO.SetDotProducts{LRO.WITH_SET}(cone, factors)
end

function maxcut(weights, solver; sparse = true, vector = true)
    T = float(eltype(weights))
    N = LinearAlgebra.checksquare(weights)
    model = GenericModel{T}(solver)
    LRO.Bridges.add_all_bridges(backend(model).optimizer, T)
    set = maxcut_set(N; sparse, T, vector)
    @variable(model, dot_prod_set[1:MOI.dimension(set)] in set)
    dot_prod = dot_prod_set[1:length(set.vectors)]
    X = reshape_vector(
        dot_prod_set[length(set.vectors) .+ (1:MOI.dimension(set.set))],
        SymmetricMatrixShape(N),
    )
    @objective(model, Max, dot(maxcut_objective(weights), X))
    @constraint(model, dot_prod .== 1)
    return model
end
