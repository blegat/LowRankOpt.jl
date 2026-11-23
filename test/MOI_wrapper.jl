# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestMOIWrapper

using Test
using JuMP
import LowRankOpt as LRO
import SolverCore

# A native solver uses the default result accessors, unlike Burer-Monteiro.
struct NativeSolver{S} <: SolverCore.AbstractOptimizationSolver
    stats::S
end

LRO.supports_scalar_constraint(::Type{NativeSolver}, S, options) = true

function MOI.get(solver::NativeSolver, ::LRO.DualSolution)
    return LRO.ShapedConeVector(-solver.stats.multipliers, Matrix{Float64}[])
end

function NativeSolver(model::LRO.Model)
    return NativeSolver(SolverCore.GenericExecutionStats(model))
end

function SolverCore.solve!(solver::NativeSolver, ::LRO.Model)
    SolverCore.set_solution!(solver.stats, [2.0, 3.0])
    SolverCore.set_status!(solver.stats, :first_order)
    return
end

@testset "Native scalar affine rows" begin
    for (set, lower, upper) in (
        (MOI.EqualTo(4.0), 4.0, 4.0),
        (MOI.GreaterThan(3.0), 3.0, Inf),
        (MOI.LessThan(5.0), -Inf, 5.0),
        (MOI.Interval(3.0, 5.0), 3.0, 5.0),
    )
        model = Model(
            optimizer_with_attributes(LRO.Optimizer, "solver" => NativeSolver),
        )
        @variable(model, x)
        @variable(model, y)
        @constraint(model, [y - 3] in Nonnegatives())
        row = @constraint(model, 2x in set)
        optimize!(model)
        opt = unsafe_backend(model)
        @test MOI.supports_constraint(
            opt,
            MOI.ScalarAffineFunction{Float64},
            typeof(set),
        )
        @test opt.model.jacobian == [0 1; 2 0]
        @test opt.model.meta.lcon == [0, lower]
        @test opt.model.meta.ucon == [Inf, upper]
        SolverCore.set_constraint_multipliers!(opt.solver.stats, [-1.0, -2.0])
        @test dual(row) == 2
        attr = MOI.RawOptimizerAttribute("solver")
        @test_throws MOI.SetAttributeNotAllowed MOI.set(
            opt,
            attr,
            LRO.BurerMonteiro.Solver,
        )
        @test MOI.get(opt, attr) === NativeSolver
        MOI.set(opt, attr, NativeSolver)
        @test MOI.get(opt, attr) === NativeSolver
    end
end

@testset "Unbridged copy rejects unsupported scalar rows" begin
    src = MOI.Utilities.Model{Float64}()
    x = MOI.add_variable(src)
    F = MOI.ScalarAffineFunction{Float64}
    MOI.add_constraint(src, convert(F, x), MOI.LessThan(2.0))
    @test_throws MOI.UnsupportedConstraint{F,MOI.LessThan{Float64}} MOI.copy_to(
        LRO.Optimizer(),
        src,
    )
end

@testset "Native solver result attributes" begin
    for attr in MOI.AbstractModelAttribute[
        LRO.PrimalSolution(),
        LRO.DualSolution(),
        LRO.DualObjective(),
    ]
        @test MOI.is_set_by_optimize(attr)
    end
    for sense in (MOI.MIN_SENSE, MOI.MAX_SENSE)
        model = Model(LRO.Optimizer)
        set_attribute(model, "solver", NativeSolver)
        @variable(model, x)
        @variable(model, y)
        @constraint(model, [x - 2, 2 - x, y - 3, 3 - y] in Nonnegatives())
        @objective(model, Min, 2x - 3y + 7)
        set_objective_sense(model, sense)
        optimize!(model)
        solver = unsafe_backend(model).solver
        @test termination_status(model) == MOI.OPTIMAL
        @test MOI.get(model, LRO.PrimalSolution()) === solver.stats.solution
        @test [value(x), value(y)] == [2.0, 3.0]
        @test objective_value(model) == 2.0
        @test isnan(MOI.get(model, LRO.DualObjective()))
        @test isnan(dual_objective_value(model))
        # Native solvers report a bound on the minimization objective,
        # without its constant. The wrapper restores the constant and sense.
        bound = sense == MOI.MIN_SENSE ? -6.0 : 4.0
        SolverCore.set_solver_specific!(solver.stats, :dual_objective, bound)
        @test MOI.get(model, LRO.DualObjective()) == bound
        @test dual_objective_value(model) ==
              (sense == MOI.MIN_SENSE ? 1.0 : 3.0)
    end
end

end
