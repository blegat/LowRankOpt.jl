module TestEqualityBasis
using Test, LinearAlgebra
import LowRankOpt as LRO
import MathOptInterface as MOI
function model(A, b)
    T=eltype(A)
    source=MOI.Utilities.Model{T}()
    x=MOI.add_variables(source, size(A, 2))
    for i in axes(A, 1)
        terms=[
            MOI.ScalarAffineTerm(A[i, j], x[j]) for
            j in axes(A, 2) if !iszero(A[i, j])
        ]
        MOI.add_constraint(
            source,
            MOI.ScalarAffineFunction(terms, zero(T)),
            MOI.EqualTo(T(b[i])),
        )
    end
    optimizer=LRO.Optimizer{T}()
    MOI.set(optimizer, MOI.RawOptimizerAttribute("solver"), _->nothing)
    MOI.copy_to(optimizer, source)
    return optimizer.model
end
@testset "Verified equality dependencies" begin
    m=model([1.0 1; 2 2], [1.0, 2])
    rows=LRO.equality_basis(m)
    @test length(rows)==1
    @test length(LRO.equality_basis(model(Float32[1 1; 2 2], Float32[1, 2])))==1
    @test_throws ArgumentError LRO.equality_basis(m; tolerance = -1.0)
    @test_throws ArgumentError LRO.equality_basis(m; tolerance = Inf)
    @test rank(Matrix(m.jacobian[rows, :]))==1
    @test length(LRO.equality_basis(model([1.0 0; 1 1], [1.0, 2])))==2
    @test length(LRO.equality_basis(model([1.0 0 1; 0 1 1], [1.0, 2])))==2
    @test isempty(LRO.equality_basis(model(zeros(2, 1), zeros(2))))
    @test length(LRO.equality_basis(model(zeros(1, 1), [1.0])))==1
    @test isempty(LRO.equality_basis(model(zeros(0, 1), Float64[])))
    @test length(LRO.equality_basis(model([1.0 1; 1 -1], [1.0, 2])))==2
    @test LRO.equality_basis(model(reshape([NaN], 1, 1), [1.0]))==[1]
    @test length(LRO.equality_basis(model(Float16[1 1; 2 2], Float16[1, 2])))==2
end
@testset "Do not discard independent or inconsistent constraints" begin
    @test length(LRO.equality_basis(model([1.0 1; 2 2], [1.0, 3])))==2
    @test length(LRO.equality_basis(model([1.0 1; 1 1+1e-14], [1.0, 1])))==2
    rounded=model([1.0 1; 2 2], [1.0, 2+1e-12])
    @test length(LRO.equality_basis(rounded))==2
    @test length(LRO.equality_basis(rounded; tolerance = 1e-10))==1
    @test length(LRO.equality_basis(rounded; tolerance = 1e-14))==2
end
end
