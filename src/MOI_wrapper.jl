import SolverCore
import NLPModelsJuMP

const VAF{T} = MOI.VectorAffineFunction{T}
const NNG = MOI.Nonnegatives
const PSD = MOI.PositiveSemidefiniteConeTriangle
const LOW{W} = LinearCombinationInSet{W,PSD}

MOI.Utilities.@product_of_sets(NNGCones, NNG)

const OptimizerCache{T} = MOI.Utilities.GenericModel{
    T,
    MOI.Utilities.ObjectiveContainer{T},
    MOI.Utilities.VariablesContainer{T},
    MOI.Utilities.MatrixOfConstraints{
        T,
        MOI.Utilities.MutableSparseMatrixCSC{
            T,
            Int64,
            MOI.Utilities.OneBasedIndexing,
        },
        Vector{T},
        NNGCones{T},
    }
}

mutable struct Optimizer{T} <: MOI.AbstractOptimizer
    solver::Union{Nothing,SolverCore.AbstractOptimizationSolver}
    model::Union{Nothing,Model{T}}
    lmi_id::Dict{MOI.ConstraintIndex{VAF{T}},Int64}
    lin_cones::Union{Nothing,NNGCones{T}}
    max_sense::Bool
    objective_constant::T
    silent::Bool
    options::Dict{String,Any}

    function Optimizer{T}() where {T}
        return new{T}(
            nothing,
            nothing,
            Dict{MOI.ConstraintIndex{VAF{T}},Int64}(),
            nothing,
            false,
            0.0,
            false,
            Dict{String,Any}(),
        )
    end
end

Optimizer() = Optimizer{Float64}()

function MOI.default_cache(::Optimizer, ::Type{T}) where {T}
    return MOI.Utilities.UniversalFallback(OptimizerCache{T}())
end

MOI.is_empty(optimizer::Optimizer) = isnothing(optimizer.model)

function MOI.empty!(optimizer::Optimizer)
    optimizer.solver = nothing
    optimizer.model = nothing
    optimizer.lin_cones = nothing
    empty!(optimizer.lmi_id)
    optimizer.objective_constant = NaN
    return
end

# /!\ FIXME type piracy
function MOI.get(
    solver::SolverCore.AbstractOptimizationSolver,
    ::MOI.SolverName,
)
    return string(parentmodule(typeof(solver)))
end

function MOI.get(optimizer::Optimizer, attr::MOI.SolverName)
    if isnothing(optimizer.solver)
        return "LowRankOpt with no solver loaded yet"
    else
        return MOI.get(optimizer.solver, attr)
    end
end

# MOI.RawOptimizerAttribute

function MOI.supports(::Optimizer, param::MOI.RawOptimizerAttribute)
    return true
end

function MOI.set(optimizer::Optimizer, param::MOI.RawOptimizerAttribute, value)
    optimizer.options[param.name] = value
    return
end

function MOI.get(optimizer::Optimizer, param::MOI.RawOptimizerAttribute)
    return optimizer.options[param.name]
end

# MOI.Silent

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
    return
end

MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

# MOI.supports

function MOI.supports(
    ::Optimizer,
    ::Union{
        MOI.ObjectiveSense,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}},
    },
) where {T}
    return true
end

const SUPPORTED_CONES = Union{NNG,PSD,LOW}

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{VAF{T}},
    ::Type{<:SUPPORTED_CONES},
) where {T}
    return true
end

SOLVER_OPTIONS = ["solver", "sub_solver", "ranks"]

function MOI.optimize!(model::Optimizer)
    options = Dict{Symbol,Any}(
        Symbol(key) => model.options[key] for
        key in keys(model.options) if !(key in SOLVER_OPTIONS)
    )
    if model.silent
        options[:verbose] = 0
    end
    SolverCore.solve!(model.solver, model.model; options...)
    return
end

function __add(A, lmi_id, k, i, j, v)
    I, J, V, _, _ = A[lmi_id, k]
    push!(I, i)
    push!(J, j)
    push!(V, v)
    return
end

function _add(A, lmi_id, k, i, j, coef)
    __add(A, lmi_id, k, i, j, coef)
    if i != j
        __add(A, lmi_id, k, j, i, coef)
    end
    return
end

function _add_constraints(::Optimizer{T}, _, _, _, ::Type{VAF{T}}, ::Type{NNG}) where {T}
    return
end

function _add_constraints(dest::Optimizer{T}, src, A, msizes, ::Type{VAF{T}}, ::Type{S}) where {T,S}
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{VAF{T},S}())
        # No need to map with `index_map` since we have the same indices thanks to
        # the `MatrixOfConstraints`
        func = MOI.get(src, MOI.CanonicalConstraintFunction(), ci)
        set = MOI.get(src, MOI.ConstraintSet(), ci)
        d = MOI.side_dimension(set)
        push!(msizes, d)
        lmi_id = length(msizes)
        dest.lmi_id[ci] = lmi_id
        for k in axes(A, 2)
            A[lmi_id, k] = (Int64[], Int64[], T[], d, d)
        end
        row = 0
        for j in 1:d
            for i in 1:j
                row += 1
                _add(A, lmi_id, 1, i, j, func.constants[row])
            end
        end
        for term in func.terms
            i, j = MOI.Utilities.inverse_trimap(term.output_index)
            scalar = term.scalar_term
            col = 1 + scalar.variable.value
            _add(A, lmi_id, col, i, j, -scalar.coefficient)
        end
    end
    return
end

function _nlmi(src, ::Type{F}, ::Type{S}) where {F,S}
    if S == NNG
        return 0
    else
        return MOI.get(src, MOI.NumberOfConstraints{F,S}())
    end
end

function MOI.copy_to(dest::Optimizer{T}, src::MOI.Utilities.UniversalFallback{OptimizerCache{T}}) where {T}
    MOI.empty!(dest)
    Cd_lin = src.model.constraints
    SM = SparseArrays.SparseMatrixCSC{T,Int64}
    C_lin = convert(SM, Cd_lin.coefficients)
    C_lin = -convert(SM, C_lin')
    n = MOI.get(src, MOI.NumberOfVariables())
    constraint_types = MOI.get(src, MOI.ListOfConstraintTypesPresent())
    nlmi = sum(constraint_types) do (F, S)
        _nlmi(src, F, S)
    end
    A = Matrix{Tuple{Vector{Int64},Vector{Int64},Vector{T},Int64,Int64}}(
        undef,
        nlmi,
        n + 1,
    )
    empty!(dest.lmi_id)
    msizes = Int64[]
    for (F, S) in constraint_types
        _add_constraints(dest, src, A, msizes, F, S)
    end
    dest.max_sense = MOI.get(src, MOI.ObjectiveSense()) == MOI.MAX_SENSE
    obj = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}())
    dest.objective_constant = MOI.constant(obj)
    b0 = zeros(T, n)
    for term in obj.terms
        b0[term.variable.value] += term.coefficient
    end
    b = dest.max_sense ? b0 : -b0

    AA = SparseArrays.SparseMatrixCSC{T,Int}[
        SparseArrays.sparse(IJV...) for IJV in A
    ]
    dest.model = Model(
        AA[:, 1],
        AA[:, 2:end],
        b,
        convert(
            SparseArrays.SparseVector{T,Int64},
            SparseArrays.sparsevec(Cd_lin.constants),
        ),
        C_lin,
        msizes,
    )
    # FIXME this does not work if an option is changed between `MOI.copy_to` and `MOI.optimize!`
    options = copy(dest.options)
    dest.lin_cones = Cd_lin.sets
    options = Dict{Symbol,Any}(
        Symbol(key) => dest.options[key] for key in keys(dest.options) if
        key in SOLVER_OPTIONS && key != "solver"
    )
    dest.solver = dest.options["solver"](dest.model; options...)
    return MOI.Utilities.identity_index_map(src)
end

function MOI.copy_to(dest::Optimizer{T}, src::MOI.ModelLike) where {T}
    cache = MOI.Utilities.UniversalFallback(OptimizerCache{T}())
    index_map = MOI.copy_to(cache, src)
    MOI.copy_to(dest, cache)
    return index_map
end

function MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec)
    return optimizer.solver.stats.elapsed_time
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return SolverCore.STATUSES[optimizer.solver.stats.status]
end

struct RawStatus <: MOI.AbstractModelAttribute
    name::Symbol
end

MOI.is_set_by_optimize(::RawStatus) = true

function MOI.get(optimizer::Optimizer, attr::RawStatus)
    return getfield(optimizer.solver.stats, attr.name)
end

# From the point of view of the solver, only a local solution is found.
# However, this this is a convex problem, this is actually a global minimum!
# We define this function instead of hard-coding `MOI.OPTIMAL` so that
# `BurerMonteiro` can override it since it is solving a non-convex formulation.
struct ConvexTerminationStatus <: MOI.AbstractModelAttribute end
MOI.is_set_by_optimize(::ConvexTerminationStatus) = true

function MOI.get(
    solver::SolverCore.AbstractOptimizationSolver,
    ::ConvexTerminationStatus,
)
    status = NLPModelsJuMP.TERMINATION_STATUS[solver.stats.status]
    if status == MOI.LOCALLY_SOLVED
        status = MOI.OPTIMAL
    elseif status == MOI.LOCALLY_INFEASIBLE
        # Since we solve the dual, we need to dualize the status
        status = MOI.DUAL_INFEASIBLE
    elseif status == MOI.NORM_LIMIT
        # Since we solve the dual, we need to dualize the status
        status = MOI.INFEASIBLE
    end
    return status
end

function MOI.get(optimizer::Optimizer, attr::ConvexTerminationStatus)
    if isnothing(optimizer.solver)
        return MOI.OPTIMIZE_NOT_CALLED
    end
    return MOI.get(optimizer.solver, attr)
end

function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    return MOI.get(optimizer, ConvexTerminationStatus())
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    if MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        return 0
    else
        return 1
    end
end

function MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    elseif MOI.get(optimizer, MOI.TerminationStatus()) in
           [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        return MOI.FEASIBLE_POINT
    elseif MOI.get(optimizer, MOI.TerminationStatus()) in
           [MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE]
        return MOI.INFEASIBLE_POINT
    else
        # TODO
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(optimizer::Optimizer{T}, attr::MOI.ObjectiveValue) where {T}
    MOI.check_result_index_bounds(optimizer, attr)
    val = dual_obj(optimizer.model, optimizer.solver.stats.multipliers)
    return optimizer.objective_constant + (optimizer.max_sense ? val : -val)
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.solver.stats.multipliers[vi.value]
end

function MOI.get(
    optimizer::Optimizer{T},
    attr::MOI.DualObjectiveValue,
) where {T}
    MOI.check_result_index_bounds(optimizer, attr)
    val = optimizer.solver.stats.objective
    return optimizer.objective_constant + (optimizer.max_sense ? val : -val)
end

function MOI.get(optimizer::Optimizer, attr::MOI.DualStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    elseif MOI.get(optimizer, MOI.TerminationStatus()) in
           [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        return MOI.FEASIBLE_POINT
    elseif MOI.get(optimizer, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
        return MOI.INFEASIBLE_POINT
    else
        # TODO
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

struct Solution <: MOI.AbstractModelAttribute end
MOI.is_set_by_optimize(::Solution) = true

function MOI.get(solver::SolverCore.AbstractOptimizationSolver, ::Solution)
    return VectorizedSolution(solver.stats.solution, solver.model.dim)
end
MOI.get(optimizer::Optimizer, attr::Solution) = MOI.get(optimizer.solver, attr)

function MOI.get(
    optimizer::Optimizer{T},
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{VAF{T},PSD},
) where {T}
    MOI.check_result_index_bounds(optimizer, attr)
    lmi_id = optimizer.lmi_id[ci]
    sol = MOI.get(optimizer, Solution())
    return TriangleVectorization(sol[MatrixIndex(lmi_id)])
end

function MOI.get(
    optimizer::Optimizer{T},
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{VAF{T},NNG},
) where {T}
    MOI.check_result_index_bounds(optimizer, attr)
    rows = MOI.Utilities.rows(optimizer.lin_cones, ci)
    sol = MOI.get(optimizer, Solution())
    return sol[ScalarIndex][rows]
end
