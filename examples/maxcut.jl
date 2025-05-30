using LinearAlgebra, JuMP, LowRankOpt
import LowRankOpt as LRO

function e_i(i, n)
    ei = zeros(n)
    ei[i] = 1
    return ei
end

function maxcut_objective(weights)
    N = LinearAlgebra.checksquare(weights)
    L = Diagonal(weights * ones(N)) - weights
    return L / 4
end

function maxcut(weights, solver)
    N = LinearAlgebra.checksquare(weights)
    model = Model(solver)
    LRO.Bridges.add_all_bridges(backend(model).optimizer, Float64)
    cone = MOI.PositiveSemidefiniteConeTriangle(N)
    factors = LRO.TriangleVectorization.(
        LRO.positive_semidefinite_factorization.(e_i.(1:N, N)),
    )
    set = LRO.SetDotProducts{LRO.WITH_SET}(cone, factors)
    @variable(
        model,
        dot_prod_set[1:(length(factors)+MOI.dimension(cone))] in set
    )
    dot_prod = dot_prod_set[1:length(factors)]
    X = reshape_vector(
        dot_prod_set[length(factors) .+ (1:MOI.dimension(cone))],
        SymmetricMatrixShape(N),
    )
    @objective(model, Max, dot(maxcut_objective(weights), X))
    @constraint(model, dot_prod .== 1)
    return model
end
