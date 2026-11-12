module TestCertificates
using Test, LinearAlgebra, SparseArrays
import LowRankOpt as LRO

@testset "Conic Farkas certificates" begin
    model = LRO.Model(
        Matrix{Float64}[],
        (),
        zeros(2),
        sparsevec([1.0, -2.0]),
        sparse([1.0 0; 0 -1]),
        Int[],
    )
    LRO.append_second_order_constraints!(
        model,
        sparse(Matrix{Float64}(I, 2, 2)),
        zeros(2),
        [2],
    )
    lambda = [1.0, 1.0, 1.0, -1.0]
    @test LRO.infeasibility_certificate(model, lambda) ≈ 1
    @test LRO.infeasibility_certificate(model, 2lambda) ≈ 2
    @test isnothing(LRO.infeasibility_certificate(model, -lambda))
    @test isnothing(LRO.infeasibility_certificate(model, [0.0, 1, 1, -1]))
    @test isnothing(LRO.infeasibility_certificate(model, fill(Inf, 4)))
    @test isnothing(LRO.infeasibility_certificate(model, zeros(4)))
    @test LRO.constraint_violation(model, [1.0, 2.0]) ≈ 1
end
@testset "Recession certificates" begin
    bare = LRO.Model(
        Matrix{Float64}[],
        (),
        [1.0, 0],
        spzeros(0),
        spzeros(2, 0),
        Int[],
    )
    model = LRO.Model(
        bare.C,
        bare.A,
        bare.b,
        spzeros(1),
        sparse(reshape([-1.0, 0.0], 2, 1)),
        Int[];
        d_eq = spzeros(1),
        C_eq = sparse(reshape([0.0, -1.0], 2, 1)),
    )
    LRO.append_second_order_constraints!(
        model,
        sparse(Matrix{Float64}(I, 2, 2)),
        zeros(2),
        [2],
    )
    @test LRO.recession_certificate(model, [1.0, 0]) ≈ 1
    @test LRO.recession_certificate(model, [3.0, 0]) ≈ 3
    @test isnothing(LRO.recession_certificate(model, [-1.0, 0]))
    @test isnothing(LRO.recession_certificate(model, [1.0, 2]))
    @test isnothing(LRO.recession_certificate(model, [1.0, -2]))
    LRO.append_second_order_constraints!(
        bare,
        sparse(Matrix{Float64}(I, 2, 2)),
        zeros(2),
        [2],
    )
    @test isnothing(LRO.recession_certificate(bare, [1.0, 2]))
    @test isnothing(LRO.recession_certificate(model, [Inf, 0]))
end

@testset "PSD certificate cone membership" begin
    model = LRO.Model(
        [-Matrix{Float64}(I, 2, 2)],
        reshape([zeros(2, 2)], 1, 1),
        [0.0],
        spzeros(0),
        spzeros(1, 0),
        [2],
    )
    @test LRO.infeasibility_certificate(model, [1.0, 0, 0, 1]) ≈ 2
    @test isnothing(LRO.infeasibility_certificate(model, [3.0, 0, 0, -1]))
    @test isnothing(LRO.infeasibility_certificate(model, [1.0, 10, -10, 1]))
    @test LRO.constraint_violation(model, [0.0]) ≈ 1
end

@testset "PSD recession membership" begin
    for sign in (-1.0, 1.0)
        model = LRO.Model(
            [Matrix{Float64}(I, 2, 2)],
            reshape([sign*Matrix{Float64}(I, 2, 2)], 1, 1),
            [1.0],
            spzeros(0),
            spzeros(1, 0),
            [2],
        )
        certificate = LRO.recession_certificate(model, [1.0])
        @test sign < 0 ? certificate ≈ 1 : isnothing(certificate)
    end
end

end
