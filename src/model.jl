# The public NLPModels interface uses free variables and cone constraints.
abstract type AbstractModel{T} <: NLPModels.AbstractNLPModel{T,Vector{T}} end
Base.broadcastable(model::AbstractModel) = Ref(model)

"""
    Model(C, A, b, d_lin, C_lin, msizes; objective_constant = 0)

Affine conic model with ordinary scalar decision variables. The model owns
its coefficient data and NLPModels metadata. Constraint outputs are ordered
as bounded scalar rows, SOC blocks, then PSD blocks identified by `dim`.
The cone entries have free componentwise metadata bounds; membership in
the cone is a block constraint.

The objective is `-b' * x + objective_constant`; scalar constraints are
`d_lin - C_lin' * x`, and PSD constraints are
`C[i] - sum(A[i,j] * x[j])`. Matrix outputs use full column-major
vectorization, so Jacobian adjoints use the Frobenius inner product.

`A` is a tuple of [`BlockColumns`](@ref) components. A dense outer matrix
in block-by-constraint order is also accepted. Use [`DualModel`](@ref) or
[`dual_model`](@ref) explicitly for algorithms operating on cone variables.
"""
mutable struct Model{T,C<:AbstractMatrix{T},A<:Tuple} <: AbstractModel{T}
    C::Vector{C}
    A::A
    b::Vector{T}
    d_lin::SparseArrays.SparseVector{T,Int64}
    C_lin::SparseArrays.SparseMatrixCSC{T,Int64}
    msizes::Vector{Int64}
    dim::ConeDimensions
    meta::NLPModels.NLPModelMeta{T,Vector{T}}
    counters::NLPModels.Counters
    jacobian::SparseArrays.SparseMatrixCSC{T,Int}
    constant::Vector{T}
    jac_rows::Vector{Int}
    jac_cols::Vector{Int}
    objective_constant::T
end

function Model(
    C::Vector{CT},
    A::Tuple,
    b::Vector{T},
    d_lin::SparseArrays.SparseVector{T},
    C_lin::SparseArrays.SparseMatrixCSC{T},
    msizes::AbstractVector{<:Integer};
    objective_constant = zero(T),
) where {T,CT<:AbstractMatrix{T}}
    components = map(_as_component, A)
    for component in components
        if size(component) != (length(b), length(C))
            throw(
                DimensionMismatch(
                    "constraint components must have one row per constraint and one logical column per PSD block",
                ),
            )
        end
        for (k, i) in enumerate(component.block_ids),
            j in axes(component.values, 1)

            coefficient = component.values[j, k]
            if !(coefficient isa AbstractMatrix{T})
                throw(
                    ArgumentError(
                        "constraint coefficients must have element type $T",
                    ),
                )
            end
            if size(coefficient) != (msizes[i], msizes[i])
                throw(
                    DimensionMismatch(
                        "constraint matrix has the wrong size for PSD block $i",
                    ),
                )
            end
        end
    end
    d_lin = convert(SparseArrays.SparseVector{T,Int64}, d_lin)
    C_lin = convert(SparseArrays.SparseMatrixCSC{T,Int64}, C_lin)
    msizes = convert(Vector{Int64}, msizes)
    n = length(b)
    ns = length(d_lin)
    dim = ConeDimensions(ns, msizes, ns .+ [0; cumsum(abs2.(msizes))])
    m = Int(length(dim))
    rows, cols, values = Int[], Int[], T[]
    for k in axes(C_lin, 2), p in SparseArrays.nzrange(C_lin, k)
        push!(rows, k)
        push!(cols, SparseArrays.rowvals(C_lin)[p])
        push!(values, -SparseArrays.nonzeros(C_lin)[p])
    end
    constant = zeros(T, m)
    constraint = VectorizedConeVector(constant, dim)
    constraint[ScalarIndex] .= d_lin
    for (i, d) in enumerate(msizes)
        constraint[MatrixIndex(i)] .= C[i]
        offset = dim.offsets[i]
        _foreach_column(components, i) do coefficients, slot
            for j in axes(coefficients, 1)
                coefficient = coefficients[j, slot]
                if iszero(coefficient)
                    continue
                end
                I, J, V = SparseArrays.findnz(SparseArrays.sparse(coefficient))
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
        C,
        components,
        b,
        d_lin,
        C_lin,
        msizes,
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

function Model(C, A::Matrix{<:AbstractMatrix}, b, d_lin, C_lin, msizes; kws...)
    return Model(C, _matrix_components(A), b, d_lin, C_lin, msizes; kws...)
end

# Each finite bound contributes one nonnegative slack. Equality rows can be
# retained separately by solvers with an augmented KKT system.
function scalar_bounds(model::Model{T}; equalities = true) where {T}
    rows, signs, constants = Int[], T[], T[]
    for i in 1:num_scalars(model)
        l, u = model.meta.lcon[i], model.meta.ucon[i]
        if !equalities && l == u
            continue
        end
        if isfinite(l)
            push!(rows, i)
            push!(signs, -one(T))
            push!(constants, model.constant[i] - l)
        end
        if isfinite(u)
            push!(rows, i)
            push!(signs, one(T))
            push!(constants, u - model.constant[i])
        end
    end
    C = SparseArrays.sparse(model.jacobian[rows, :]')
    for j in axes(C, 2), p in SparseArrays.nzrange(C, j)
        SparseArrays.nonzeros(C)[p] *= signs[j]
    end
    return C, constants, rows, signs
end

function cone_multipliers(model::Model, cone)
    if num_scalars(model) == length(model.d_lin) &&
       iszero(num_second_order_cones(model))
        return cone
    end
    _, _, rows, signs = scalar_bounds(model)
    scalars = zeros(eltype(cone), num_scalars(model))
    for k in eachindex(rows)
        scalars[rows[k]] -= signs[k] * cone[ScalarIndex][k]
    end
    soc = Vector{eltype(cone)}[]
    offset = num_matrices(model)
    for i in second_order_indices(model)
        matrix = cone[MatrixIndex(offset+i.value)]
        push!(soc, [LinearAlgebra.tr(matrix); 2 .* matrix[1, 2:end]])
    end
    return ShapedConeVector(
        scalars,
        soc,
        [cone[i] for i in matrix_indices(model)],
    )
end

function append_scalar_constraints!(
    model::Model{T},
    J,
    c,
    lower,
    upper,
) where {T}
    ns = num_scalars(model)
    jacobian = vcat(model.jacobian[1:ns, :], J, model.jacobian[(ns+1):end, :])
    constant = vcat(model.constant[1:ns], c, model.constant[(ns+1):end])
    lcon = vcat(model.meta.lcon[1:ns], lower, model.meta.lcon[(ns+1):end])
    ucon = vcat(model.meta.ucon[1:ns], upper, model.meta.ucon[(ns+1):end])
    nscalar = ns + length(c)
    model.dim = ConeDimensions(
        nscalar,
        model.dim.second_order_dimensions,
        model.dim.side_dimensions,
        cumsum(
            [
                nscalar+sum(model.dim.second_order_dimensions);
                model.dim.side_dimensions .^ 2
            ],
        ),
    )
    model.jacobian = jacobian
    model.constant = constant
    model.jac_rows, model.jac_cols, _ = SparseArrays.findnz(jacobian)
    m = length(constant)
    model.meta = NLPModels.NLPModelMeta{T,Vector{T}}(
        model.meta.nvar;
        ncon = m,
        x0 = copy(model.meta.x0),
        lcon,
        ucon,
        nnzj = length(model.jac_rows),
        nnzh = 0,
        lin_nnzj = length(model.jac_rows),
        nln_nnzj = 0,
        lin = collect(1:m),
    )
    return model
end

primal_model(model::Model) = model
num_scalars(model::Model) = model.dim.num_scalars
num_second_order_cones(model::Model) = num_second_order_cones(model.dim)
num_matrices(model::Model) = length(model.C)
function matrix_indices(model::Model)
    return MOI.Utilities.LazyMap{MatrixIndex}(
        _matrix_index,
        Base.OneTo(num_matrices(model)),
    )
end
side_dimension(model::Model, idx::MatrixIndex) = model.msizes[idx.value]
NLPModels.unconstrained(model::Model) = iszero(model.meta.ncon)
NLPModels.has_bounds(::Model) = false

function NLPModels.obj(model::Model, x::AbstractVector)
    NLPModels.increment!(model, :neval_obj)
    return -LinearAlgebra.dot(model.b, x) + model.objective_constant
end

function NLPModels.grad!(
    model::Model,
    x::AbstractVector,
    gradient::AbstractVector,
)
    NLPModels.increment!(model, :neval_grad)
    gradient .= -model.b
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
