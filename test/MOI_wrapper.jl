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
    dim::LRO.ConeDimensions
end

LRO.support_cone(::Type{NativeSolver}, ::Type{MOI.Zeros}) = true
LRO.support_cone(::Type{NativeSolver}, ::Type{MOI.SecondOrderCone}) = true

function MOI.get(solver::NativeSolver, ::LRO.DualSolution)
    return LRO.VectorizedConeVector(-solver.stats.multipliers, solver.dim)
end

function NativeSolver(model::LRO.Model)
    return NativeSolver(SolverCore.GenericExecutionStats(model), model.dim)
end

function SolverCore.solve!(solver::NativeSolver, ::LRO.Model)
    SolverCore.set_solution!(solver.stats, [2.0, 3.0])
    SolverCore.set_status!(solver.stats, :first_order)
    return
end

@testset "Scalar and vector inputs give the solver the same two blocks" begin
    @test LRO.support_cone(NativeSolver, MOI.Zeros)
    @test !LRO.support_cone(LRO.BurerMonteiro.Solver, MOI.Zeros)
    @test LRO.support_cone(LRO.BurerMonteiro.Solver, MOI.Nonnegatives)
    for grouped in (true, false)
        factory =
            optimizer_with_attributes(LRO.Optimizer, "solver" => NativeSolver)
        model =
            direct_model(MOI.instantiate(factory; with_bridge_type = Float64))
        @variable(model, x)
        @variable(model, y)
        if grouped
            eq = (@constraint(model, [2x - 4, 3y - 9, x + y - 5] in Zeros()),)
            ge = (
                @constraint(model, [x - 1, y - 2, x + y - 3] in Nonnegatives()),
            )
        else
            e = @constraint(model, 2x == 4)
            g = @constraint(model, 1x >= 1)
            eq = (e, @constraint(model, [3y - 9, x + y - 5] in Zeros()))
            ge = (g, @constraint(model, [y - 2, x + y - 3] in Nonnegatives()))
        end
        optimize!(model)
        opt = unsafe_backend(model)
        @test MOI.supports_constraint(
            opt,
            MOI.VectorAffineFunction{Float64},
            MOI.Zeros,
        )
        @test !MOI.supports_constraint(
            opt,
            MOI.ScalarAffineFunction{Float64},
            MOI.EqualTo{Float64},
        )
        @test opt.model.C_lin == [-1 0 -1; 0 -1 -1]
        @test opt.model.d_lin == [-1, -2, -3]
        @test opt.model.C_eq == [-2 0 -1; 0 -3 -1]
        @test opt.model.d_eq == [-4, -9, -5]
        @test opt.model.meta.jfix == [4, 5, 6]
        @test opt.model.jacobian == -hcat(opt.model.C_lin, opt.model.C_eq)'
        @test LRO.NLPModels.cons(opt.model, [2.0, 3.0]) == [1, 1, 2, 0, 0, 0]
        @test_throws ArgumentError LRO.dual_model(opt.model)
        @test_throws ArgumentError LRO.BurerMonteiro.Solver(
            opt.model;
            sub_solver = nothing,
            ranks = Int[],
        )
        SolverCore.set_constraint_multipliers!(
            opt.solver.stats,
            [-1.0, -2, -3, 4, -5, 6],
        )
        @test vcat(dual.(ge)...) == [1, 2, 3]
        @test vcat(dual.(eq)...) == [-4, 5, -6]
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

@testset "A solver can opt into native SOC blocks" begin
    model = Model(
        optimizer_with_attributes(LRO.Optimizer, "solver" => NativeSolver),
    )
    @variable(model, x)
    @variable(model, y)
    c = @constraint(model, [x + y, 3, 4] in SecondOrderCone())
    optimize!(model)
    opt = unsafe_backend(model)
    @test opt.model.dim.second_order_dimensions == [3]
    @test isempty(opt.model.msizes)
    @test LRO.NLPModels.cons(opt.model, [2.0, 3.0]) == [5, 3, 4]
    SolverCore.set_constraint_multipliers!(opt.solver.stats, [-1.0, 0.6, 0.8])
    @test dual(c) ≈ [1, -0.6, -0.8]
    @test_throws MOI.SetAttributeNotAllowed MOI.set(
        opt,
        MOI.RawOptimizerAttribute("solver"),
        LRO.BurerMonteiro.Solver,
    )
    @test_throws ArgumentError LRO.BurerMonteiro.Solver(
        opt.model;
        sub_solver = nothing,
        ranks = Int[],
    )
end

@testset "Unbridged copy requires equality support" begin
    src = MOI.Utilities.Model{Float64}()
    x = MOI.add_variable(src)
    F = MOI.VectorAffineFunction{Float64}
    MOI.add_constraint(src, MOI.Utilities.vectorize([2.0 * x]), MOI.Zeros(1))
    @test !MOI.supports_constraint(LRO.Optimizer(), F, MOI.Zeros)
    @test_throws MOI.UnsupportedConstraint{F,MOI.Zeros} MOI.copy_to(
        LRO.Optimizer(),
        src,
    )
end

@testset "Copied constraint duals survive reuse of the MOI cache" begin
    opt = LRO.Optimizer()
    MOI.set(opt, MOI.RawOptimizerAttribute("solver"), NativeSolver)
    src = MOI.Utilities.Model{Float64}()
    x, y = MOI.add_variables(src, 2)
    eq = MOI.add_constraint(
        src,
        MOI.Utilities.vectorize([2.0 * x, 3.0 * y]),
        MOI.Zeros(2),
    )
    cache = MOI.default_cache(opt, Float64)
    index_map = MOI.copy_to(cache, src)
    ci = MOI.copy_to(opt, cache)[index_map[eq]]
    MOI.empty!(cache)
    MOI.optimize!(opt)
    SolverCore.set_constraint_multipliers!(opt.solver.stats, [1.0, -2.0])
    @test MOI.get(opt, MOI.ConstraintDual(), ci) == [-1, 2]
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
