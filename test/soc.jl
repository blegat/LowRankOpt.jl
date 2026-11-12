module TestSOC
using Test, LinearAlgebra, SparseArrays
import LowRankOpt as LRO
import NLPModels

@testset "Native SOC model and explicit PSD adapter" begin
    model = LRO.Model(
        Matrix{Float64}[],
        (),
        [-1.0, 2.0],
        spzeros(0),
        spzeros(2, 0),
        Int[];
        objective_constant = 7,
    )
    J = sparse([1.0 0; 0 1; 1 -1])
    c = [2.0, 1, 0]
    LRO.append_second_order_constraints!(model, J, c, [3])
    @test model.meta.nvar == 2
    @test model.meta.ncon == 3
    @test LRO.num_second_order_cones(model) == 1
    @test LRO.num_matrices(model) == 0
    @test all(isinf, model.meta.lcon)
    @test NLPModels.cons(model, [1.0, 2]) ≈ c+J*[1, 2]
    dual = LRO.dual_model(model)
    @test LRO.primal_model(dual) === model
    @test LRO.primal_model(dual).objective_constant == 7
    @test LRO.num_second_order_cones(LRO.primal_model(dual)) == 1
    @test LRO.num_matrices(LRO.primal_model(dual)) == 0
    @test LRO.num_matrices(dual) == 1
    x = [0.2, 0.3]
    g = c+J*x
    arrow =
        dual.C[1] - sum(x[j]*LRO.jac(dual, j, LRO.MatrixIndex(1)) for j in 1:2)
    @test arrow ≈ [g[1] g[2] g[3]; g[2] g[1] 0; g[3] 0 g[1]]
    v = [1.0, 2, -1]
    Z = v*v'
    cone = LRO.ShapedConeVector(Float64[], [Z])
    native = LRO.cone_multipliers(model, cone)
    @test native[LRO.SOCIndex(1)] ≈ [6, 4, -2]
    @test dot(native[LRO.SOCIndex(1)], g) ≈ dot(Z, arrow)
    @test collect(native) ≈ [6, 4, -2]
    @test LRO.rotate_soc(LRO.rotate_soc(g)) ≈ g
    @test dot(LRO.rotate_soc(g), LRO.rotate_soc(g)) ≈ dot(g, g)
    LRO.append_scalar_constraints!(model, sparse([1.0 1]), [0.0], [1.0], [1.0])
    @test model.meta.ncon == 4
    @test model.dim.num_scalars == 1
    @test model.dim.second_order_dimensions == [3]
    @test NLPModels.cons(model, x) ≈ [sum(x); g]
end

@testset "SOC vector order" begin
    dim = LRO.ConeDimensions(1, [3], [2], [4, 8])
    flat = collect(1.0:8)
    vector = LRO.VectorizedConeVector(flat, dim)
    shaped = LRO.ShapedConeVector([1.0], [[2.0, 3, 4]], [[5.0 7; 6 8]])
    @test vector[LRO.SOCIndex(1)] == [2, 3, 4]
    @test vector[LRO.MatrixIndex(1)] == [5 7; 6 8]
    @test collect(shaped) == flat
    @test dot(shaped, shaped) ≈ dot(flat, flat)
    @test LRO.num_second_order_cones(shaped) == 1
    @test LRO.num_matrices(shaped) == 1
end
end
