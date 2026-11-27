module TestSparseMOICoefficients
using Test, LinearAlgebra, SparseArrays, Random
import LowRankOpt as LRO
import NLPModels
import MathOptInterface as MOI

include("workspace_utils.jl")

function captured(source; options...)
    optimizer = LRO.Optimizer()
    MOI.set(optimizer, MOI.RawOptimizerAttribute("solver"), _ -> nothing)
    for (key, value) in options
        MOI.set(optimizer, MOI.RawOptimizerAttribute(string(key)), value)
    end
    MOI.copy_to(optimizer, source)
    return optimizer.model
end

@testset "Sparse MOI coefficient construction" begin
    for rank_one in (false, true)
        source = MOI.Utilities.Model{Float64}()
        variables = MOI.add_variables(source, 1000)
        terms = MOI.VectorAffineTerm.(
            [1, 2, 3],
            MOI.ScalarAffineTerm.([2.0, 3, -4], variables[[1, 1000, 2]]),
        )
        MOI.add_constraint(
            source,
            MOI.VectorAffineFunction(terms, [1.0, 0.5, 2]),
            MOI.PositiveSemidefiniteConeTriangle(2),
        )
        MOI.add_constraint(
            source,
            MOI.VectorAffineFunction(
                [
                    MOI.VectorAffineTerm(
                        1,
                        MOI.ScalarAffineTerm(1.0, variables[1]),
                    ),
                ],
                [2.0],
            ),
            MOI.PositiveSemidefiniteConeTriangle(1),
        )
        optimizer = LRO.Optimizer()
        MOI.set(optimizer, MOI.RawOptimizerAttribute("solver"), _ -> nothing)
        MOI.set(
            optimizer,
            MOI.RawOptimizerAttribute("detect_rank_one"),
            rank_one,
        )
        MOI.copy_to(optimizer, source)
        model = optimizer.model
        dual = LRO.dual_model(model)
        x = zeros(1000)
        x[[1, 2, 1000]] .= [0.2, 0.3, 0.4]
        @test sum(sum(length, c.values.nzval) for c in model.A) == 4
        @test all(c -> all(allunique, c.values.rowval), model.A)
        @test all(c -> all(issorted, c.values.rowval), model.A)
        @test model.meta.nvar == 1000
        @test model.meta.ncon == 5
        @test model.meta.nnzj == 5
        @test NLPModels.cons(model, x) ≈ [1.4, 1.7, 1.7, 0.8, 2.2]
        @test LRO.jac(dual, 1, LRO.MatrixIndex(1)) ≈ [-2.0 0; 0 0]
        @test LRO.jac(dual, 1000, LRO.MatrixIndex(1)) ≈ [0.0 -3; -3 0]
        @test iszero(LRO.jac(dual, 500, LRO.MatrixIndex(1)))
        @test LRO.jac(dual, 1, LRO.MatrixIndex(2)) ≈ fill(-1.0, 1, 1)
    end
end

# Compares the products of the workspace, which only visit the stored
# coefficients, with the coefficients read from the native Jacobian.
function test_workspace(model; kws...)
    rng = MersenneTwister(0)
    workspace = LRO.ModelWorkspace(model; kws...)
    m = model.meta.nvar
    idx = LRO.matrix_indices(model)
    A = [Matrix(_coefficient(model, j, i)) for i in idx, j in 1:m]
    nscalars = model.meta.ncon - sum(abs2, model.msizes)
    W = LRO.ShapedConeVector(
        rand(rng, nscalars),
        [(R = randn(rng, d, d); R * R' + I) for d in model.msizes],
    )
    expected = [
        sum(tr(A[i.value, j] * W[i] * A[i.value, k] * W[i]) for i in idx)
        for j in 1:m, k in 1:m
    ]
    H = zeros(m, m)
    LRO.schur_complement!(workspace, W, H)
    @test H ≈ expected
    y = randn(rng, m)
    Hy = zeros(m)
    LRO.eval_schur_complement!(workspace, W, y, Hy)
    @test Hy ≈ expected * y
    for i in idx
        @test LRO.unsafe_jprod(workspace, y, i) ≈
              -sum(y[j] * A[i.value, j] for j in 1:m)
    end
    return workspace
end

@testset "Workspace products with absent variables" begin
    # Variable 5 is absent from the first block and variable 6 from both.
    source = MOI.Utilities.Model{Float64}()
    y = MOI.add_variables(source, 6)
    d = 20
    tri(i, j) = div(j * (j - 1), 2) + i
    terms = MOI.VectorAffineTerm.(
        tri.([1, 3, 4, 5, 5], [2, 3, 4, 6, 5]),
        MOI.ScalarAffineTerm.([1.0, 2, -1, 3, 1], y[[1, 2, 3, 4, 4]]),
    )
    MOI.add_constraint(
        source,
        MOI.VectorAffineFunction(terms, zeros(tri(d, d))),
        MOI.PositiveSemidefiniteConeTriangle(d),
    )
    MOI.add_constraint(
        source,
        MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, y[5]))],
            [1.0, 0, 1],
        ),
        MOI.PositiveSemidefiniteConeTriangle(2),
    )
    model = captured(source; detect_rank_one = false)
    values = only(model.A).values
    @test values isa LRO.SparseBlockCoefficients{SparseMatrixCSC{Float64,Int}}
    @test values.rowval == [[1, 2, 3, 4], [5]]
    for dense in (false, true)
        workspace = test_workspace(model; dense_condition = _ -> dense)
        @test workspace.jprod_buffer[1] isa SparseMatrixCSC
        component = only(workspace.schur_buffer[3])
        @test component.permutation[1] == [4, 1, 2, 3]
        @test component.permutation[2] == [5]
        @test component.last_dense == (dense ? [4, 1] : [0, 0])
    end
end

@testset "Workspace products of factored coefficients" begin
    # The rank-two factors form a general component which is not sparse and
    # the rank-one factor a rank-one component. Variable 1 is absent.
    factors = LRO.positive_semidefinite_factorization.([
        [1.0 0; 2 1; 0 1],
        [1.0, -1, 2],
        [0.0 1; 1 0; 1 1],
    ])
    set = LRO.LinearCombinationInSet{LRO.WITHOUT_SET}(
        MOI.PositiveSemidefiniteConeTriangle(3),
        LRO.TriangleVectorization.(factors),
    )
    source = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    x = MOI.add_variables(source, 4)
    MOI.add_constraint(
        source,
        MOI.Utilities.vectorize([-1.0 * x[2], -1.0 * x[3], -1.0 * x[4]]),
        set,
    )
    model = captured(source)
    @test length(model.A) == 2
    @test all(c -> c.values isa LRO.SparseBlockCoefficients, model.A)
    @test sort([only(c.values.rowval) for c in model.A]) == [[2, 4], [3]]
    test_workspace(model)
end

@testset "Dense coefficient block" begin
    source = MOI.Utilities.Model{Float64}()
    x = MOI.add_variable(source)
    MOI.add_constraint(
        source,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.([1, 3], MOI.ScalarAffineTerm.(1.0, [x, x])),
            zeros(3),
        ),
        MOI.PositiveSemidefiniteConeTriangle(2),
    )
    model = captured(source)
    @test only(model.A).values isa
          LRO.SparseBlockCoefficients{SparseMatrixCSC{Float64,Int}}
    @test NLPModels.cons(model, [2.0]) ≈ [2, 0, 0, 2]
end

@testset "Zero-variable models" begin
    if pkgversion(NLPModels) < v"0.21.13"
        @test_skip NLPModels.NLPModelMeta(0)
    else
        constant = MOI.Utilities.Model{Float64}()
        MOI.add_constraint(
            constant,
            MOI.VectorAffineFunction(
                MOI.VectorAffineTerm{Float64}[],
                [1.0, 0, 1],
            ),
            MOI.PositiveSemidefiniteConeTriangle(2),
        )
        model = captured(constant)
        @test isempty(model.A)
        @test model.meta.nvar == 0
        @test NLPModels.cons(model, Float64[]) ≈ [1, 0, 0, 1]
        empty = captured(MOI.Utilities.Model{Float64}())
        @test isempty(empty.C)
        @test isempty(empty.A)
        @test empty.meta.nvar == empty.meta.ncon == 0
    end
end

@testset "Zero contributions in a factored PSD constraint" begin
    factors = LRO.positive_semidefinite_factorization.([[1.0, 2], zeros(2)])
    set = LRO.LinearCombinationInSet{LRO.WITHOUT_SET}(
        MOI.PositiveSemidefiniteConeTriangle(2),
        LRO.TriangleVectorization.(factors),
    )
    source = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    x, y = MOI.add_variables(source, 2)
    MOI.add_constraint(
        source,
        MOI.Utilities.vectorize([-1.0 * x, -1.0 * y]),
        set,
    )
    model = captured(source)
    @test only(only(model.A).values.rowval) == [1]
    @test NLPModels.cons(model, [1.0, 3.0]) == -vec([1 2; 2 4])

    builder = LRO._MatrixBuilder{Float64}(2)
    LRO._add!(builder, 1, 0.0, set)
    empty_matrix = LRO._instantiate(builder)
    @test empty_matrix isa LRO.FillArrays.Zeros
    @test size(empty_matrix) == (2, 2)
end

end
