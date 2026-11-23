module TestScalarConstraints
using Test, LinearAlgebra, SparseArrays
using JuMP
import LowRankOpt as LRO
import Percival

@testset "MOI bounds and duals for mixed constraint types" begin
    model = Model(LRO.Optimizer)
    set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
    set_attribute(model, "sub_solver", Percival.PercivalSolver)
    set_attribute(model, "ranks", Int[])
    set_silent(model)
    @variable(model, x[1:8])
    @constraint(model, positive, [x[1] - 1, x[2] - 2] in Nonnegatives())
    @constraint(model, upper, 2x[3] <= 6)
    @constraint(model, interval, 8 <= 2x[4] <= 12)
    @constraint(model, negative, [x[5] - 5, x[6] - 6] in Nonpositives())
    @constraint(model, fixed, [x[7] - 7, x[8] - 8] in Zeros())
    @objective(model, Min, dot([1, 2, -6, 8, -5, -6, 7, -8], x))
    optimize!(model)
    opt = unsafe_backend(model)
    F = MOI.ScalarAffineFunction{Float64}
    for S in (
        MOI.EqualTo{Float64},
        MOI.GreaterThan{Float64},
        MOI.LessThan{Float64},
        MOI.Interval{Float64},
    )
        @test !MOI.supports_constraint(opt, F, S)
    end
    @test all(iszero, opt.model.meta.lcon)
    @test all(==(Inf), opt.model.meta.ucon)
    bridges = sprint(print_active_bridges, model)
    @test occursin("SplitIntervalBridge", bridges)
    @test occursin("LessToGreaterBridge", bridges) ||
          occursin("NonposToNonnegBridge", bridges)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test value.(x) ≈ 1:8 atol = 1e-5
    @test dual(positive) ≈ [1, 2] atol = 1e-5
    @test dual(upper) ≈ -3 atol = 1e-5
    @test dual(interval) ≈ 4 atol = 1e-5
    @test dual(negative) ≈ [-5, -6] atol = 1e-5
    @test dual(fixed) ≈ [7, -8] atol = 1e-5
end

@testset "Linear block dimensions" begin
    b = [1.0]
    c = spzeros(1)
    C = spzeros(1, 1)
    @test_throws DimensionMismatch LRO.Model(
        Matrix{Float64}[],
        (),
        b,
        c,
        spzeros(1, 0),
        Int[],
    )
    @test_throws DimensionMismatch LRO.Model(
        Matrix{Float64}[],
        (),
        b,
        c,
        C,
        Int[];
        d_eq = c,
        C_eq = spzeros(2, 1),
    )
end
end
