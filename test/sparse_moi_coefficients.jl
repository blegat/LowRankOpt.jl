module TestSparseMOICoefficients
using Test, LinearAlgebra, SparseArrays
import LowRankOpt as LRO
import NLPModels
import MathOptInterface as MOI

function captured(source)
    optimizer = LRO.Optimizer()
    MOI.set(optimizer, MOI.RawOptimizerAttribute("solver"), _ -> nothing)
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
        @test sum(sum(length, c.values.columns) for c in model.A) == 4
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
          LRO.SparseBlockCoefficients{SparseMatrixCSC{Float64,Int64}}
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
    @test collect(keys(only(only(model.A).values.columns))) == [1]
    @test NLPModels.cons(model, [1.0, 3.0]) == -vec([1 2; 2 4])

    builder = LRO._MatrixBuilder{Float64}(2)
    LRO._add!(builder, 1, 0.0, set)
    empty_matrix = LRO._instantiate(builder)
    @test empty_matrix isa LRO.FillArrays.Zeros
    @test size(empty_matrix) == (2, 2)
end

end
