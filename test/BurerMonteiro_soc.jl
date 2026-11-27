module TestBurerMonteiroSOC
using Test
import JuMP
import Percival
import LowRankOpt as LRO
import MathOptInterface as MOI

@testset "MOI bridges SOC constraints for BurerMonteiro" begin
    m = JuMP.Model(LRO.Optimizer)
    JuMP.set_silent(m)
    JuMP.set_attribute(m, "solver", LRO.BurerMonteiro.Solver)
    JuMP.set_attribute(m, "sub_solver", Percival.PercivalSolver)
    JuMP.set_attribute(m, "ranks", [2])
    JuMP.@variable(m, t)
    c = JuMP.@constraint(m, [t, 3, 4] in MOI.SecondOrderCone(3))
    JuMP.@objective(m, Min, t+2)
    JuMP.optimize!(m)
    o = JuMP.unsafe_backend(m)
    F = MOI.VectorAffineFunction{Float64}
    @test !MOI.supports_constraint(o, F, MOI.SecondOrderCone)
    @test !MOI.supports_constraint(o, F, MOI.RotatedSecondOrderCone)
    bridges = sprint(JuMP.print_active_bridges, m)
    @test occursin("SOCtoRSOCBridge", bridges)
    @test occursin("RSOCtoPSDBridge", bridges)
    @test LRO.num_second_order_cones(o.model) == 0
    @test o.model.msizes == [2]
    @test JuMP.termination_status(m) == MOI.LOCALLY_SOLVED
    @test JuMP.value(t) ≈ 5 atol=1e-4
    @test JuMP.objective_value(m) ≈ 7 atol=1e-4
    @test JuMP.dual(c) ≈ [1, -0.6, -0.8] atol=1e-4
    @test MOI.get(o, MOI.DualObjectiveValue()) ≈ 7 atol=1e-4
end

end
