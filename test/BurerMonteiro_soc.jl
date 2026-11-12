module TestBurerMonteiroSOC
using Test, LinearAlgebra
import JuMP
import Percival
import LowRankOpt as LRO
import NLPModels
import MathOptInterface as MOI

@testset "BurerMonteiro explicitly dualizes native SOC blocks" begin
    m = JuMP.Model(LRO.Optimizer)
    JuMP.set_silent(m)
    JuMP.set_attribute(m, "solver", LRO.BurerMonteiro.Solver)
    JuMP.set_attribute(m, "sub_solver", Percival.PercivalSolver)
    JuMP.set_attribute(m, "ranks", Int[])
    JuMP.@variable(m, t)
    c = JuMP.@constraint(m, [t, 3, 4] in MOI.SecondOrderCone(3))
    JuMP.@objective(m, Min, t+2)
    JuMP.optimize!(m)
    o = JuMP.unsafe_backend(m)
    @test JuMP.termination_status(m) == MOI.LOCALLY_SOLVED
    @test JuMP.value(t) ≈ 5 atol=1e-4
    @test JuMP.objective_value(m) ≈ 7 atol=1e-4
    @test o.solver.model.dim.ranks == [3]
    @test JuMP.dual(c) ≈ [1, -0.6, -0.8] atol=1e-4
    @test MOI.get(o, MOI.DualObjectiveValue()) ≈ 7 atol=1e-4
    cone = MOI.get(o, LRO.DualSolution())
    @test cone isa LRO.ShapedConeVector
    @test length(cone) == o.model.meta.ncon
    @test norm(
        NLPModels.grad(o.model, o.solver.stats.multipliers) +
        NLPModels.jtprod(o.model, o.solver.stats.multipliers, -collect(cone)),
        Inf,
    ) < 1e-4
end
end
