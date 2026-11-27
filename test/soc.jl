module TestSOC
using Test, LinearAlgebra, SparseArrays
import LowRankOpt as LRO
import NLPModels

@testset "Native SOC model and explicit PSD adapter" begin
    model = LRO.Model(
        Matrix{Float64}[],
        (),
        [-1.0, 2.0],
        sparsevec([1.0]),
        sparse(ones(2, 1)),
        Int[];
        objective_constant = 7,
    )
    J = sparse([1.0 0; 0 1; 1 -1])
    c = [2.0, 1, 0]
    LRO.append_second_order_constraints!(model, J, c, [3])
    @test model.meta.nvar == 2
    @test model.meta.ncon == 4
    @test LRO.num_second_order_cones(model) == 1
    @test LRO.num_matrices(model) == 0
    @test model.meta.lcon == [0, -Inf, -Inf, -Inf]
    @test NLPModels.cons(model, [1.0, 2]) ≈ [-2; c+J*[1, 2]]
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
    cone = LRO.ShapedConeVector([2.0], [Z])
    native = LRO.cone_multipliers(model, cone)
    @test native[LRO.SOCIndex(1)] ≈ [6, 4, -2]
    @test dot(native[LRO.SOCIndex(1)], g) ≈ dot(Z, arrow)
    @test collect(native) ≈ [2, 6, 4, -2]
    @test LRO.rotate_soc(LRO.rotate_soc(g)) ≈ g
    @test dot(LRO.rotate_soc(g), LRO.rotate_soc(g)) ≈ dot(g, g)
end

@testset "SOC vector order" begin
    dim = LRO.ConeDimensions(1, [3], [2], [4, 8])
    flat = collect(1.0:8)
    vector = LRO.VectorizedConeVector(flat, dim)
    shaped = LRO.ShapedConeVector([1.0], [[2.0, 3, 4]], [[5.0 7; 6 8]])
    @test only(LRO.second_order_indices(vector)) == LRO.SOCIndex(1)
    @test getindex.((vector, shaped), LRO.SOCIndex(1)) == ([2, 3, 4], [2, 3, 4])
    @test vector[LRO.MatrixIndex(1)] == [5 7; 6 8]
    @test collect(shaped) == flat
    @test -shaped == -flat
    @test dot(shaped, shaped) ≈ dot(flat, flat)
    @test LRO.num_second_order_cones(shaped) == 1
    @test LRO.num_matrices(shaped) == 1
end

@testset "SOC lift preserves existing PSD blocks" begin
    model = LRO.Model(
        [fill(3.0, 1, 1)],
        reshape([fill(2.0, 1, 1)], 1, 1),
        [0.0],
        spzeros(0),
        spzeros(1, 0),
        [1],
    )
    psd = fill(4.0, 1, 1)
    cone = LRO.ShapedConeVector(Float64[], [psd])
    @test LRO.cone_multipliers(model, cone) === cone

    LRO.append_second_order_constraints!(
        model,
        sparse([1.0; -1;;]),
        [2.0, 0],
        [2],
    )
    dual = LRO.dual_model(model)
    @test dual.msizes == [1, 2]
    @test first(dual.A).values === only(model.A).values
    @test LRO.jac(dual, 1, LRO.MatrixIndex(1)) == fill(2.0, 1, 1)
    @test LRO.jac(dual, 1, LRO.MatrixIndex(2)) == [-1 1; 1 -1]
    cone = LRO.ShapedConeVector(Float64[], [psd, [3.0 1; 1 2]])
    recovered = LRO.cone_multipliers(model, cone)
    @test recovered[LRO.SOCIndex(1)] == [5, 2]
    @test recovered[LRO.MatrixIndex(1)] === psd
end

end
