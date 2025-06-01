import SolverCore
import NLPModelsJuMP

const VAF{T} = MOI.VectorAffineFunction{T}
const PSD = MOI.PositiveSemidefiniteConeTriangle
const NNG = MOI.Nonnegatives

MOI.Utilities.@product_of_sets(NNGCones, NNG)

MOI.Utilities.@product_of_sets(PSDCones, PSD)

MOI.Utilities.@struct_of_constraints_by_set_types(PSDOrNot, PSD, NNG)

const OptimizerCache{T} = MOI.Utilities.GenericModel{
    T,
    MOI.Utilities.ObjectiveContainer{T},
    MOI.Utilities.VariablesContainer{T},
    PSDOrNot{T}{
        MOI.Utilities.MatrixOfConstraints{
            T,
            MOI.Utilities.MutableSparseMatrixCSC{
                T,
                Int64,
                MOI.Utilities.OneBasedIndexing,
            },
            Vector{T},
            PSDCones{T},
        },
        MOI.Utilities.MatrixOfConstraints{
            T,
            MOI.Utilities.MutableSparseMatrixCSC{
                T,
                Int64,
                MOI.Utilities.OneBasedIndexing,
            },
            Vector{T},
            NNGCones{T},
        },
    },
}

mutable struct Optimizer{T} <: MOI.AbstractOptimizer
    solver::Union{Nothing,SolverCore.AbstractOptimizationSolver}
    model::Union{Nothing,Model{T}}
    stats::Union{
        Nothing,
        SolverCore.GenericExecutionStats{T,Vector{T},Vector{T},Any},
    }
    lmi_id::Dict{MOI.ConstraintIndex{VAF{T},PSD},Int64}
    lin_cones::Union{Nothing,NNGCones{T}}
    max_sense::Bool
    objective_constant::T
    silent::Bool
    options::Dict{String,Any}

    function Optimizer{T}() where {T}
        return new{T}(
            nothing,
            nothing,
            nothing,
            Dict{MOI.ConstraintIndex{VAF{T},PSD},Int64}(),
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
    optimizer.model = nothing
    optimizer.lin_cones = nothing
    return
end

MOI.get(::Optimizer, ::MOI.SolverName) = "Loraine"

# MOI.RawOptimizerAttribute

function MOI.supports(::Optimizer, param::MOI.RawOptimizerAttribute)
    return true
end

function MOI.set(optimizer::Optimizer, param::MOI.RawOptimizerAttribute, value)
    if !MOI.supports(optimizer, param)
        throw(MOI.UnsupportedAttribute(param))
    end
    optimizer.options[param.name] = value
    if !isnothing(optimizer.solver)
        setproperty!(optimizer.solver, Symbol(param.name), value)
    end
    return
end

function MOI.get(optimizer::Optimizer, param::MOI.RawOptimizerAttribute)
    if !MOI.supports(optimizer, param)
        throw(MOI.UnsupportedAttribute(param))
    end
    return optimizer.options[param.name]
end

# MOI.Silent

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
    return
end

MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

function MOI.set(optimizer::Optimizer, ::MOI.ObjectiveSense, value::Bool)
    optimizer.max_sense = value
    return
end

# MOI.supports

function MOI.supports(
    ::Optimizer,
    ::Union{MOI.ObjectiveSense,MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}},
) where {T}
    return true
end

const SUPPORTED_CONES = Union{NNG,PSD}

function MOI.supports_constraint(::Optimizer{T}, ::Type{VAF{T}}, ::Type{<:SUPPORTED_CONES}) where {T}
    return true
end

SOLVER_OPTIONS = ["solver", "sub_solver", "ranks"]

function MOI.optimize!(model::Optimizer)
    options = Dict{Symbol, Any}(
        Symbol(key) => model.options[key] for key in keys(model.options) if !(key in SOLVER_OPTIONS)
    )
    if model.silent
        options[:verbose] = 0
    else
        options[:verbose] = 1
    end
    model.stats = SolverCore.GenericExecutionStats(model.model)
    SolverCore.solve!(model.solver, model.model, model.stats; options...)
    return
end

function MOI.copy_to(dest::Optimizer{T}, src::OptimizerCache{T}) where {T}
    MOI.empty!(dest)
    psd_AC = MOI.Utilities.constraints(src.constraints, VAF{T}, PSD)
    Cd_lin = MOI.Utilities.constraints(src.constraints, VAF{T}, NNG)
    SM = SparseArrays.SparseMatrixCSC{T,Int64}
    psd_A = convert(SM, psd_AC.coefficients)
    C_lin = convert(SM, Cd_lin.coefficients)
    C_lin = -convert(SM, C_lin')
    n = MOI.get(src, MOI.NumberOfVariables())
    nlmi = MOI.get(src, MOI.NumberOfConstraints{VAF{T},PSD}())
    A = Matrix{Tuple{Vector{Int64},Vector{Int64},Vector{T},Int64,Int64}}(undef, nlmi, n + 1)
    back = Vector{Tuple{Int64,Int64,Int64}}(undef, size(psd_A, 1))
    empty!(dest.lmi_id)
    row = 0
    msizes = Int64[]
    for (lmi_id, ci) in enumerate(MOI.get(src, MOI.ListOfConstraintIndices{VAF{T},PSD}()))
        dest.lmi_id[ci] = lmi_id
        set = MOI.get(src, MOI.ConstraintSet(), ci)
        d = set.side_dimension
        push!(msizes, d)
        for k = 1:(n+1)
            A[lmi_id, k] = (Int64[], Int64[], T[], d, d)
        end
        for j = 1:d
            for i = 1:j
                row += 1
                back[row] = (lmi_id, i, j)
            end
        end
    end
    function __add(lmi_id, k, i, j, v)
        I, J, V, _, _ = A[lmi_id, k]
        push!(I, i)
        push!(J, j)
        push!(V, v)
        return
    end
    function _add(lmi_id, k, i, j, coef)
        __add(lmi_id, k, i, j, coef)
        if i != j
            __add(lmi_id, k, j, i, coef)
        end
        return
    end
    for row in eachindex(back)
        lmi_id, i, j = back[row]
        _add(lmi_id, 1, i, j, -psd_AC.constants[row])
    end
    for var = 1:n
        for k in SparseArrays.nzrange(psd_A, var)
            lmi_id, i, j = back[SparseArrays.rowvals(psd_A)[k]]
            col = 1 + var
            _add(lmi_id, col, i, j, SparseArrays.nonzeros(psd_A)[k])
        end
    end
    dest.max_sense = MOI.get(src, MOI.ObjectiveSense()) == MOI.MAX_SENSE
    obj = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}())
    # objective_constant = MOI.constant(obj) # TODO # MK: done(?)
    b_const = obj.constant
    b_const = dest.max_sense ? -b_const : b_const
    b0 = zeros(T, n)
    for term in obj.terms
        b0[term.variable.value] += term.coefficient
    end
    b = dest.max_sense ? b0 : -b0
    # b = max_sense ? -b0 : b0

    AA = SparseArrays.SparseMatrixCSC{T,Int}[SparseArrays.sparse(IJV...) for IJV in A]
    dest.model = Model(
        -AA[:,1],
        AA[:,2:end],
        b,
        b_const,
        convert(SparseArrays.SparseVector{T,Int64}, SparseArrays.sparsevec(Cd_lin.constants)),
        C_lin,
        msizes,
    )
    # FIXME this does not work if an option is changed between `MOI.copy_to` and `MOI.optimize!`
    options = copy(dest.options)
    if dest.silent
        options["verb"] = 0
    end
    dest.lin_cones = Cd_lin.sets
    options = Dict{Symbol, Any}(
        Symbol(key) => dest.options[key] for key in keys(dest.options) if key in SOLVER_OPTIONS && key != "solver"
    )
    dest.solver = dest.options["solver"](dest.model; options...)
    return MOI.Utilities.identity_index_map(src)
end

function MOI.copy_to(dest::Optimizer{T}, src::MOI.ModelLike) where {T}
    cache = OptimizerCache{T}()
    index_map = MOI.copy_to(cache, src)
    MOI.copy_to(dest, cache)
    return index_map
end

function MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec)
    return optimizer.stats.elapsed_time
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return SolverCore.STATUSES[optimizer.stats.status]
end

struct RawStatus <: MOI.AbstractModelAttribute
    name::Symbol
end

MOI.is_set_by_optimize(::RawStatus) = true

function MOI.get(optimizer::Optimizer, attr::RawStatus)
    return getfield(optimizer.stats, attr.name)
end

function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
  if isnothing(optimizer.stats)
    return MOI.OPTIMIZE_NOT_CALLED
  end
  return NLPModelsJuMP.TERMINATION_STATUS[optimizer.stats.status]
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
    elseif MOI.get(optimizer, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
        return MOI.FEASIBLE_POINT
    else
        # TODO
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(optimizer::Optimizer{T}, attr::MOI.ObjectiveValue) where {T}
    MOI.check_result_index_bounds(optimizer, attr)
    val = optimizer.stats.objective
    return optimizer.max_sense ? -val : val
end

function MOI.get(optimizer::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
  MOI.check_result_index_bounds(optimizer, attr)
  return optimizer.stats.solution[vi.value]
end

function MOI.get(optimizer::Optimizer{T}, attr::MOI.DualObjectiveValue) where {T}
    MOI.check_result_index_bounds(optimizer, attr)
    val = Solvers.obj(optimizer.solver.model, optimizer.solver.X_lin, optimizer.solver.X)::T
    return optimizer.max_sense ? -val : val
end

function MOI.get(::Optimizer, ::MOI.DualStatus)
    # TODO
    return MOI.NO_SOLUTION
end

function MOI.get(
    optimizer::Optimizer{T},
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{VAF{T},PSD},
) where {T}
    MOI.check_result_index_bounds(optimizer, attr)
    lmi_id = optimizer.lmi_id[ci]
    X = optimizer.solver.X[lmi_id]
    n = optimizer.solver.model.msizes[lmi_id]
    return [X[i, j] for j = 1:n for i = 1:j]::Vector{T}
end

function MOI.get(
    optimizer::Optimizer{T},
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{VAF{T},NNG},
) where {T}
    MOI.check_result_index_bounds(optimizer, attr)
    rows = MOI.Utilities.rows(optimizer.lin_cones, ci)
    return optimizer.solver.X_lin[rows]::Vector{T}
end
