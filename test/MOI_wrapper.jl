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

function NativeSolver(model::LRO.Model)
    return NativeSolver(SolverCore.GenericExecutionStats(model))
end

function SolverCore.solve!(solver::NativeSolver, ::LRO.Model)
    SolverCore.set_solution!(solver.stats, [2.0, 3.0])
    SolverCore.set_status!(solver.stats, :first_order)
    return
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
