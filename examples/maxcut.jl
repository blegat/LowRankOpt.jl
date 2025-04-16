using LinearAlgebra, JuMP, LowRankOpt

function maxcut(weights, solver)
    N = LinearAlgebra.checksquare(weights)
    L = Diagonal(weights * ones(N)) - weights
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
    @objective(model, Max, dot(L, X) / 4)
    @constraint(model, dot_prod .== 1)
    return model
end
