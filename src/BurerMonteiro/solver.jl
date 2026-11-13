struct Solver{S,T,D,JTB,ST} <: SolverCore.AbstractOptimizationSolver
    model::Model{S,T,D,JTB}
    solver::ST
    stats::SolverCore.GenericExecutionStats{T,Vector{T},Vector{T},Any}
end

function Solver(
    src::Union{LRO.Model,LRO.DualModel};
    sub_solver,
    ranks,
    square_scalars = false,
    kws...,
)
    model = Model{square_scalars}(LRO.dual_model(src), ranks)
    solver = sub_solver(model; kws...)
    stats = SolverCore.GenericExecutionStats(model)
    return Solver(model, solver, stats)
end

function SolverCore.solve!(
    solver::Solver,
    model::NLPModels.AbstractNLPModel; # Same as `solver.model.model`
    kws...,
)
    return SolverCore.solve!(solver.solver, solver.model, solver.stats; kws...)
end

function MOI.get(solver::Solver, attr::MOI.SolverName)
    return "BurerMonteiro with " * MOI.get(solver.solver, attr)
end

function MOI.get(solver::Solver, ::LRO.ConvexTerminationStatus)
    return NLPModelsJuMP.TERMINATION_STATUS[solver.stats.status]
    # TODO if the dual is feasible, we can still claim that we found the optimal
    #      and turn `LOCALLY_SOLVED` into `OPTIMAL`
end

function MOI.get(solver::Solver, ::LRO.DualSolution)
    return Solution(solver.stats.solution, solver.model.dim)
end

# Burer-Monteiro solves the explicitly dualized model.
MOI.get(solver::Solver, ::LRO.PrimalSolution) = solver.stats.multipliers
MOI.get(solver::Solver, ::LRO.DualObjective) = -solver.stats.objective
