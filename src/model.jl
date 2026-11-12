# The public NLPModels interface uses free variables and cone constraints.
# The explicit DualModel retains the cone-variable formulation required by
# factorization algorithms and the existing primal-dual linear algebra.
abstract type AbstractModel{T} <: NLPModels.AbstractNLPModel{T,Vector{T}} end
Base.broadcastable(model::AbstractModel) = Ref(model)

"""
    Model(data::DualModel; objective_constant = 0)

Affine conic model with ordinary scalar decision variables. Constraint
outputs are ordered as the bounded scalar prefix followed by PSD blocks.
The scalar prefix has lower bound zero; PSD blocks are identified by `dim`
and have free componentwise bounds in `meta`. Their positive-semidefinite
restriction is a block constraint, not a componentwise bound.

The objective is `-data.b' * x + objective_constant`; scalar constraints
are `data.d_lin - data.C_lin' * x`, and PSD constraints are
`data.C[i] - sum(data.A[i,j] * x[j])`. Matrix outputs use full column-major
vectorization, so Jacobian adjoints use the Frobenius inner product.

Use [`dual_model`](@ref) explicitly for algorithms operating on cone
variables. This constructor shares their coefficient data; it does not
change the NLPModels variables to cone variables.
"""
mutable struct Model{T,D<:DualModel{T}} <: AbstractModel{T}
    data::D
    dim::ConeDimensions
    meta::NLPModels.NLPModelMeta{T,Vector{T}}
    counters::NLPModels.Counters
    jacobian::SparseArrays.SparseMatrixCSC{T,Int}
    constant::Vector{T}
    jac_rows::Vector{Int}
    jac_cols::Vector{Int}
    objective_constant::T
end

function Model(data::DualModel{T}; objective_constant = zero(T)) where {T}
    n = length(data.b)
    dim = data.dim
    m = Int(length(dim))
    rows, cols, values = Int[], Int[], T[]
    # The first block contains ordinary scalar inequalities.
    C = data.C_lin
    for k in axes(C, 2), p in SparseArrays.nzrange(C, k)
        push!(rows, k)
        push!(cols, SparseArrays.rowvals(C)[p])
        push!(values, -SparseArrays.nonzeros(C)[p])
    end
    constant = zeros(T, m)
    constraint = VectorizedConeVector(constant, dim)
    constraint[ScalarIndex] .= data.d_lin
    for idx in matrix_indices(data)
        i, d = idx.value, side_dimension(data, idx)
        constraint[idx] .= data.C[i]
        offset = dim.offsets[i]
        _foreach_column(data.A, i) do coefficients, slot
            for j in axes(coefficients, 1)
                A = coefficients[j, slot]
                if iszero(A)
                    continue
                end
                I, J, V = SparseArrays.findnz(SparseArrays.sparse(A))
                for k in eachindex(V)
                    push!(rows, offset + I[k] + (J[k] - 1) * d)
                    push!(cols, j)
                    push!(values, -V[k])
                end
            end
            return
        end
    end
    jacobian = SparseArrays.sparse(rows, cols, values, m, n)
    SparseArrays.dropzeros!(jacobian)
    rows, cols, _ = SparseArrays.findnz(jacobian)
    lower = fill(T(-Inf), m)
    lower[1:dim.num_scalars] .= zero(T)
    meta = NLPModels.NLPModelMeta{T,Vector{T}}(
        n;
        ncon = m,
        nnzj = length(rows),
        nnzh = 0,
        lin_nnzj = length(rows),
        nln_nnzj = 0,
        lcon = lower,
        ucon = fill(T(Inf), m),
        lin = collect(1:m),
        minimize = true,
    )
    return Model(
        data,
        dim,
        meta,
        NLPModels.Counters(),
        jacobian,
        constant,
        rows,
        cols,
        T(objective_constant),
    )
end

"""
    dual_model(model::Model)

Return the explicitly named cone-variable dual formulation. The public
model and its dual share affine coefficient data.
"""
dual_model(model::Model) = model.data
dual_model(model::AbstractDualModel) = model
primal_model(model::Model) = model
primal_model(model::DualModel) = Model(model)
primal_model(model::BufferedDualModel) = Model(model.model)
DualModel(model::Model) = dual_model(model)

num_scalars(model::Model) = model.dim.num_scalars
num_matrices(model::Model) = num_matrices(model.data)
matrix_indices(model::Model) = matrix_indices(model.data)
side_dimension(model::Model, idx::MatrixIndex) = side_dimension(model.data, idx)
NLPModels.unconstrained(model::Model) = iszero(model.meta.ncon)
NLPModels.has_bounds(::Model) = false

function NLPModels.obj(model::Model, x::AbstractVector)
    NLPModels.increment!(model, :neval_obj)
    return -LinearAlgebra.dot(model.data.b, x) + model.objective_constant
end

function NLPModels.grad!(
    model::Model,
    x::AbstractVector,
    gradient::AbstractVector,
)
    NLPModels.increment!(model, :neval_grad)
    gradient .= -model.data.b
    return gradient
end

function NLPModels.cons!(
    model::Model,
    x::AbstractVector,
    constraints::AbstractVector,
)
    NLPModels.increment!(model, :neval_cons)
    copyto!(constraints, model.constant)
    LinearAlgebra.mul!(constraints, model.jacobian, x, true, true)
    return constraints
end

function NLPModels.jprod!(
    model::Model,
    x::AbstractVector,
    direction::AbstractVector,
    result::AbstractVector,
)
    NLPModels.increment!(model, :neval_jprod)
    return LinearAlgebra.mul!(result, model.jacobian, direction)
end

function NLPModels.jtprod!(
    model::Model,
    x::AbstractVector,
    multipliers::AbstractVector,
    result::AbstractVector,
)
    NLPModels.increment!(model, :neval_jtprod)
    return LinearAlgebra.mul!(result, model.jacobian', multipliers)
end

function NLPModels.jac_structure!(
    model::Model,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    copyto!(rows, model.jac_rows)
    copyto!(cols, model.jac_cols)
    return rows, cols
end

function NLPModels.jac_coord!(
    model::Model,
    x::AbstractVector,
    values::AbstractVector,
)
    NLPModels.increment!(model, :neval_jac)
    copyto!(values, SparseArrays.nonzeros(model.jacobian))
    return values
end

function NLPModels.hess_structure!(
    model::Model,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    return rows, cols
end

function NLPModels.hess_coord!(
    model::Model,
    x::AbstractVector,
    multipliers::AbstractVector,
    values::AbstractVector;
    obj_weight = one(eltype(x)),
)
    NLPModels.increment!(model, :neval_hess)
    fill!(values, zero(eltype(values)))
    return values
end

function NLPModels.hprod!(
    model::Model,
    x::AbstractVector,
    multipliers::AbstractVector,
    direction::AbstractVector,
    result::AbstractVector;
    obj_weight = one(eltype(x)),
)
    NLPModels.increment!(model, :neval_hprod)
    fill!(result, zero(eltype(result)))
    return result
end

# This concrete model is affine; expose its linear block to NLPModels clients.
function NLPModels.cons_lin!(
    model::Model,
    x::AbstractVector,
    result::AbstractVector,
)
    NLPModels.increment!(model, :neval_cons_lin)
    copyto!(result, model.constant)
    LinearAlgebra.mul!(result, model.jacobian, x, true, true)
    return result
end

function NLPModels.cons_nln!(
    model::Model,
    x::AbstractVector,
    result::AbstractVector,
)
    NLPModels.increment!(model, :neval_cons_nln)
    fill!(result, zero(eltype(result)))
    return result
end

function NLPModels.jac_lin_structure!(
    model::Model,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    return NLPModels.jac_structure!(model, rows, cols)
end

function NLPModels.jac_nln_structure!(
    model::Model,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    return rows, cols
end

function NLPModels.jac_lin_coord!(
    model::Model,
    x::AbstractVector,
    values::AbstractVector,
)
    NLPModels.increment!(model, :neval_jac_lin)
    copyto!(values, SparseArrays.nonzeros(model.jacobian))
    return values
end

function NLPModels.jac_nln_coord!(
    model::Model,
    x::AbstractVector,
    values::AbstractVector,
)
    NLPModels.increment!(model, :neval_jac_nln)
    fill!(values, zero(eltype(values)))
    return values
end

function NLPModels.jprod_lin!(
    model::Model,
    x::AbstractVector,
    direction::AbstractVector,
    result::AbstractVector,
)
    NLPModels.increment!(model, :neval_jprod_lin)
    return LinearAlgebra.mul!(result, model.jacobian, direction)
end

function NLPModels.jprod_nln!(
    model::Model,
    x::AbstractVector,
    direction::AbstractVector,
    result::AbstractVector,
)
    NLPModels.increment!(model, :neval_jprod_nln)
    fill!(result, zero(eltype(result)))
    return result
end

function NLPModels.jtprod_lin!(
    model::Model,
    x::AbstractVector,
    multipliers::AbstractVector,
    result::AbstractVector,
)
    NLPModels.increment!(model, :neval_jtprod_lin)
    return LinearAlgebra.mul!(result, model.jacobian', multipliers)
end

function NLPModels.jtprod_nln!(
    model::Model,
    x::AbstractVector,
    multipliers::AbstractVector,
    result::AbstractVector,
)
    NLPModels.increment!(model, :neval_jtprod_nln)
    fill!(result, zero(eltype(result)))
    return result
end

function NLPModels.hess_coord!(
    model::Model,
    x::AbstractVector,
    values::AbstractVector;
    obj_weight = one(eltype(x)),
)
    NLPModels.increment!(model, :neval_hess)
    fill!(values, zero(eltype(values)))
    return values
end

function NLPModels.hprod!(
    model::Model,
    x::AbstractVector,
    direction::AbstractVector,
    result::AbstractVector;
    obj_weight = one(eltype(x)),
)
    NLPModels.increment!(model, :neval_hprod)
    fill!(result, zero(eltype(result)))
    return result
end
