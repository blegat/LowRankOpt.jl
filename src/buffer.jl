"""
    ModelWorkspace(model::Model; dense_condition = default_dense_condition(),
                   schur = true, jacobian = true)

Workspaces for native Jacobian products and Schur assembly. This is not an
NLPModels model: `model` retains the free variables, cone constraints, and
metadata. Coefficient columns store positive `vec(A[i,j])`; the native
Jacobian products apply the minus sign in `C[i] - sum(A[i,j] * x[j])`.

Use `schur = false` for Jacobian products without Schur assembly storage.
Use `jacobian = false, schur = false` when the solver builds its own KKT
operators. Scalar Jacobian products remain available. Reconfiguring an
existing workspace preserves any buffers that are still needed.
"""
mutable struct ModelWorkspace{T,M<:Model{T},CB,JB,SB}
    model::M
    coefficient_buffer::CB
    jprod_buffer::JB
    schur_buffer::SB
    scalar_jacobian_transpose::SparseArrays.SparseMatrixCSC{T,Int}
end

Base.broadcastable(workspace::ModelWorkspace) = Ref(workspace)

function ModelWorkspace(
    model::Model;
    dense_condition = default_dense_condition(),
    schur::Bool = true,
    jacobian::Bool = true,
)
    if schur && !jacobian
        throw(ArgumentError("Schur workspaces require Jacobian buffers."))
    end
    scalar_jacobian = model.jacobian[1:num_scalars(model), :]
    return ModelWorkspace(
        model,
        jacobian ? buffer_for_coefficients(model) : nothing,
        jacobian ? buffer_for_jprod(model) : nothing,
        schur ? buffer_for_schur_complement(model, dense_condition) : nothing,
        SparseArrays.sparse(scalar_jacobian'),
    )
end

function ModelWorkspace(
    workspace::ModelWorkspace;
    dense_condition = default_dense_condition(),
    schur::Bool = true,
    jacobian::Bool = true,
)
    if schur && !jacobian
        throw(ArgumentError("Schur workspaces require Jacobian buffers."))
    end
    has_jacobian = !isnothing(workspace.coefficient_buffer)
    if jacobian == has_jacobian && schur == !isnothing(workspace.schur_buffer)
        return workspace
    end
    coefficients, products = if !jacobian
        nothing, nothing
    elseif has_jacobian
        workspace.coefficient_buffer, workspace.jprod_buffer
    else
        buffer_for_coefficients(workspace.model), buffer_for_jprod(workspace.model)
    end
    schur_buffer = if !schur
        nothing
    elseif isnothing(workspace.schur_buffer)
        buffer_for_schur_complement(workspace.model, dense_condition)
    else
        workspace.schur_buffer
    end
    return ModelWorkspace(
        workspace.model,
        coefficients,
        products,
        schur_buffer,
        workspace.scalar_jacobian_transpose,
    )
end

function _require_jacobian(workspace::ModelWorkspace)
    if isnothing(workspace.coefficient_buffer)
        throw(
            ArgumentError(
                "Jacobian buffers are disabled; use jacobian = true.",
            ),
        )
    end
    return
end

function buffer_for_jprod(model::Model)
    if iszero(num_matrices(model))
        return nothing
    end
    return map(Base.Fix1(buffer_for_jprod, model), matrix_indices(model))
end

function _schur_buffer(workspace::ModelWorkspace)
    if isnothing(workspace.schur_buffer)
        throw(ArgumentError("Schur workspaces are disabled; use schur = true."))
    end
    return workspace.schur_buffer
end

num_scalars(workspace::ModelWorkspace) = num_scalars(workspace.model)
num_matrices(workspace::ModelWorkspace) = num_matrices(workspace.model)
matrix_indices(workspace::ModelWorkspace) = matrix_indices(workspace.model)
function side_dimension(workspace::ModelWorkspace, i::MatrixIndex)
    return side_dimension(workspace.model, i)
end

function _add_vec!(_, _, _, _, offset, ::FillArrays.Zeros)
    return offset
end

function _add_vec!(I, J, V, j, offset, A::SparseArrays.SparseMatrixCSC)
    Ai, Av = SparseArrays.findnz(A[:])
    K = offset .+ eachindex(Ai)
    I[K] = Ai
    J[K] .= j
    V[K] = Av
    return offset + length(Ai)
end

"""
    buffer_for_coefficients(model::Model, i::MatrixIndex)

Return the sparse matrix collecting the vectorization of every constraint
matrix of the `i`th PSD block:
```
𝐀ᵢ = [vec(Aᵢ₁) vec(Aᵢ₂) … vec(Aᵢₙ)] ∈ ℝ^(mᵢ² × n)
```
where `mᵢ` is the side dimension of the block and `n` is the number of
free variables. This is the matrix `𝒜` of [HKS24, Section 3.1], which is
stated there for a single PSD block, here built once per block.

The native PSD Jacobian is `-𝐀ᵢ`: its JVP maps a free-variable direction
to a matrix, and its adjoint contracts a matrix against the coefficient
columns. The positive coefficients are also reused by Schur assembly,
where the two Jacobian signs cancel.

`SparseMatrixCSC` is stored with an offset by column.
This means that getting view `view(A, :, I)` can be handles efficently,
these give `SparseMatrixCSCView` (if `I` is a `UnitRange`) and
`SparseMatrixCSCColumnSubset` otherwise.
In `schur.jl`, we therefore get a `SparseMatrixCSCColumnSubset`.
Columns index free variables; rows index vectorized matrix entries.
The subsets used in `schur.jl` are suffixes of the columns sorted by
decreasing number of nonzeros, following [FKN97].

[HKS24] Habibi, Soodeh, Michal Kočvara, and Michael Stingl. "Loraine -- an
interior-point solver for low-rank semidefinite programming."
Optimization Methods and Software 39.6 (2024): 1185-1215.
[FKN97] Fujisawa, Katsuki, Masakazu Kojima, and Kazuhide Nakata.
"Exploiting sparsity in primal-dual interior-point methods for
semidefinite programming." Mathematical Programming 79 (1997): 235-253.
"""
function buffer_for_coefficients(model::Model{T}, i::MatrixIndex) where {T}
    J = _rank_one_coefficients(model, i)
    if !isnothing(J)
        return J
    end
    return _coefficient_buffer(
        model.A,
        i.value,
        side_dimension(model, i),
        model.meta.nvar,
        T,
    )
end

function _coefficient_buffer(components, i, d, nvar, ::Type{T}) where {T}
    nnz = _sum_columns(components, i, 0) do column
        return sum(_nnz, column; init = 0)
    end
    I = zeros(Int64, nnz)
    J = zeros(Int64, nnz)
    V = zeros(T, nnz)
    offset = 0
    # Components contribute to the same sparse columns; `sparse` adds
    # duplicate entries, including overlaps between representations.
    _foreach_column(components, i) do values, slot
        for j in axes(values, 1)
            offset = _add_vec!(I, J, V, j, offset, values[j, slot])
        end
    end
    A = SparseArrays.sparse(I, J, V, d^2, nvar)
    return A
end

function buffer_for_coefficients(model::Model{T}) where {T}
    buffers =
        Union{SparseArrays.SparseMatrixCSC{T,Int64},RankOneCoefficients{T}}[
            buffer_for_coefficients(model, i) for i in matrix_indices(model)
        ]
    # Loraine accesses the sparse `𝐀ᵢ` directly so keep the concrete
    # element type when no block is rank-one.
    if all(J -> J isa SparseArrays.SparseMatrixCSC, buffers)
        return convert(Vector{SparseArrays.SparseMatrixCSC{T,Int64}}, buffers)
    end
    return buffers
end

"""
    struct RankOneCoefficients{T}
        U::Vector{Matrix{T}}
        weights::Vector{Vector{T}}
    end

Replaces the sparse `𝐀ᵢ` of `buffer_for_coefficients` for a PSD block whose
constraint matrices are all sums of rank-one matrices
`Aᵢⱼ = ∑ₚ weights[p][j] * U[p][j, :] * U[p][j, :]'`, with one `p` per
component. The sparse `𝐀ᵢ` would have `nnz(U[p][j, :])²` entries for each
constraint `j` while this stores the `m × d` factors `U[p]`. The products
are computed with dense BLAS calls on the factors:

* Coefficient adjoint:
  `⟨Aᵢⱼ, V⟩ = ∑ₚ weights[p][j] * (U[p] * V)[j, :] ⋅ U[p][j, :]`,
* Coefficient product:
  `∑ⱼ yⱼ Aᵢⱼ = ∑ₚ U[p]' * Diagonal(weights[p] .* y) * U[p]`.
"""
struct RankOneCoefficients{T}
    U::Vector{Matrix{T}}
    weights::Vector{Vector{T}}
    # Workspaces of size `m × d` and `m`
    scaled::Matrix{T}
    coef::Vector{T}
end

function _rank_one_coefficients(model::Model{T}, i::MatrixIndex) where {T}
    d = side_dimension(model, i)
    U = Matrix{T}[]
    weights = Vector{T}[]
    rank_one = true
    _foreach_column(model.A, i.value) do values, slot
        if values isa AbstractMatrix{<:_RankOneOrZeros{T}}
            Ui, wi = _rank_one_factors(values, slot, d)
            push!(U, Ui)
            push!(weights, wi)
        else
            rank_one = false
        end
        return
    end
    if !rank_one || isempty(U)
        return
    end
    m = model.meta.nvar
    # The products cost `Θ(nnz)` with the sparse `𝐀ᵢ` and `Θ(m d²)` with the
    # factors so we only use the factors if `𝐀ᵢ` has at least `m d`
    # nonzeros. This excludes, e.g., one-hot factors.
    nnz = _sum_columns(model.A, i.value, 0) do column
        return sum(_nnz, column; init = 0)
    end
    if nnz < m * d
        return
    end
    return RankOneCoefficients(U, weights, zeros(T, m, d), zeros(T, m))
end

function _add_coefficient_adjoint!(
    V,
    Jv::AbstractVector,
    J::RankOneCoefficients,
    α = true,
)
    for (U, w) in zip(J.U, J.weights)
        LinearAlgebra.mul!(J.scaled, U, V)
        for k in axes(U, 2), j in eachindex(Jv)
            Jv[j] += α * w[j] * J.scaled[j, k] * U[j, k]
        end
    end
    return Jv
end

# `vec(buffer) = 𝐀ᵢ * y`
function _mul_coefficients!(buffer, A::SparseArrays.SparseMatrixCSC, y, α)
    return LinearAlgebra.mul!(_vec(buffer), A, y, α, false)
end
function _mul_coefficients!(buffer, J::RankOneCoefficients, y, α)
    for (p, (U, w)) in enumerate(zip(J.U, J.weights))
        # `y` may be a `SparseVector`, compute `w .* y` once before
        # broadcasting it on the rows of `U`
        for j in eachindex(J.coef)
            J.coef[j] = α * w[j] * y[j]
        end
        J.scaled .= J.coef .* U
        LinearAlgebra.mul!(buffer, U', J.scaled, true, !isone(p))
    end
    return buffer
end

_vec(x::AbstractVector) = x
_vec(x::FillArrays.Zeros{T}) where {T} = FillArrays.Zeros{T}(length(x))
_vec(x::AbstractArray) = UnsafeArrays.uview(x, :)
_vec(x::Base.ReshapedArray) = _vec(parent(x))

function _add_coefficient_adjoint!(V, Jv::AbstractArray, A, α = true)
    return _add_mul!(Jv, A', _vec(V), α)
end

function _add_sub_coefficient_adjoint!(V, result, A, indices, α)
    return _add_coefficient_adjoint!(V, result, view(A, :, indices), α)
end

function _add_sub_coefficient_adjoint!(
    V,
    result,
    J::RankOneCoefficients,
    indices,
    α,
)
    selected = view(J.coef, indices)
    if length(result) != length(selected)
        throw(DimensionMismatch("Result length must match selected columns."))
    end
    if isempty(selected)
        return result
    end
    # Reuse the product workspace, then gather in the requested order.
    # Computing each column once also handles repeated selected indices.
    fill!(J.coef, zero(eltype(J.coef)))
    _add_coefficient_adjoint!(V, J.coef, J, α)
    result .+= selected
    return result
end

function _merge_sparsity(
    A::SparseArrays.SparseMatrixCSC,
    B::SparseArrays.SparseMatrixCSC,
)
    return A + B
end
_merge_sparsity(::FillArrays.Zeros, B::SparseArrays.SparseMatrixCSC) = B
_merge_sparsity(A::SparseArrays.SparseMatrixCSC, ::FillArrays.Zeros) = A
_merge_sparsity(A::FillArrays.Zeros, ::FillArrays.Zeros) = A

_abs(A::SparseArrays.SparseMatrixCSC) = abs.(A)
_abs(A::FillArrays.Zeros) = A

"""
    DENSE_JPROD_DENSITY

Maximum density of the merged constraint pattern for a sparse JVP buffer.
The work estimate controlled by `SPARSE_JPROD_WORK_RATIO` must also favor the
sparse path: density alone does not predict the cost of the Schur product.
"""
const DENSE_JPROD_DENSITY = 0.05

"""
    SPARSE_JPROD_WORK_RATIO

Maximum estimated sparse work relative to `d^3`, where `d` is the block size.
The estimate is `nvar * d + nnz_A * nnz_pattern`: the first term accounts for
column scans when filling the sparse buffer, and the second estimates the
pairwise nonzero work of the `_dot` contractions across all constraints.
`nnz_A` counts nonzeros across the constraint matrices, including overlaps;
`nnz_pattern` counts nonzeros in their merged pattern.

This is a conservative heuristic, not a flop-count crossover: dense BLAS and
sparse scalar loops have different costs per operation. Julia v1.13.0 with
Loraine, loading its SDPLIB examples, gave the following Schur-product timings
(including matrix-product fill, not full solves):

| problem  | dense   | sparse + `_dot` | selected |
|----------|---------|-----------------|----------|
| maxG11   | 8.66 ms | 1.96 ms         | sparse   |
| thetaG11 | 8.67 ms | 28.61 ms        | dense    |

The block dimensions and stored nonzero counts are:

| problem  | d   | nvar | nnz_A | nnz_pattern | pattern density |
|----------|-----|------|-------|-------------|-----------------|
| maxG11   | 800 | 800  | 800   | 800         | 0.125%          |
| thetaG11 | 801 | 2401 | 15201 | 5601        | 0.873%          |

Both pass the 5% density cutoff. The work cutoff distinguishes them:

- maxG11: `800*800 + 800*800 = 1_280_000`, below
  `0.05*800^3 = 25_600_000`. The work ratio is `0.0025 <= 0.05`, so sparse.
- thetaG11: `2401*801 + 15201*5601 = 87_064_002`, above
  `0.05*801^3 = 25_696_120.05`. The work ratio is approximately
  `0.1694 > 0.05`, so dense despite the low merged-pattern density.

These are the best of seven calls after two warmup calls per path, with four
BLAS threads and one Julia thread on an Intel Core Ultra 7 265H. Both paths
used identical inputs: the loaded constraint matrices, a random dense
positive-definite `W`, and a random `y`; their outputs agreed numerically.
The crossover can vary with hardware, threading, and the distribution of
nonzeros.
"""
const SPARSE_JPROD_WORK_RATIO = 0.05

function buffer_for_jprod(
    model::Model{T},
    mat_idx::MatrixIndex;
    density = DENSE_JPROD_DENSITY,
    work_ratio = SPARSE_JPROD_WORK_RATIO,
) where {T}
    d = side_dimension(model, mat_idx)
    nvar = model.meta.nvar
    # If every `Aᵢⱼ` is zero then so is the product, and callers rely on
    # getting a `Zeros` back (`_sub` then aliases `C` instead of copying).
    nnz_A = _sum_columns(model.A, mat_idx.value, 0) do column
        return sum(_nnz, column; init = 0)
    end
    if iszero(nnz_A)
        return FillArrays.Zeros{T}(d, d)
    end
    # The absolute values make the merged pattern monotone, so stop as soon
    # as either cutoff is exceeded instead of merging every dense constraint.
    # They also prevent the buffer from aliasing a single nonzero constraint.
    max_nnz =
        min(density * d^2, (work_ratio * float(d)^3 - float(nvar) * d) / nnz_A)
    pattern = FillArrays.Zeros{T}(d, d)
    dense = false
    _foreach_column(model.A, mat_idx.value) do values, slot
        for j in axes(values, 1)
            A = values[j, slot]
            if !(A isa Union{SparseArrays.SparseMatrixCSC,FillArrays.Zeros})
                dense = true
                break
            end
            pattern = _merge_sparsity(pattern, _abs(A))
            if _nnz(pattern) > max_nnz
                dense = true
                break
            end
        end
    end
    if dense
        return zeros(T, d, d)
    end
    return pattern
end

_zero!(A::SparseArrays.SparseMatrixCSC) = fill!(SparseArrays.nonzeros(A), 0.0)

# Computes `A .+= B * α`
function _add_mul!(A::SparseArrays.SparseMatrixCSC, ::FillArrays.Zeros, _)
    return A
end

function _add_mul!(
    A::SparseArrays.SparseMatrixCSC,
    B::SparseArrays.SparseMatrixCSC,
    α,
)
    for col in axes(A, 2)
        range_B = SparseArrays.nzrange(B, col)
        # `B` is one constraint matrix while `A` is the merged pattern of all
        # of them, so most columns of `B` are empty. Skipping them with a
        # `colptr` comparison avoids setting up the `A` iterator for nothing.
        isempty(range_B) && continue
        range_A = SparseArrays.nzrange(A, col)
        it_A = iterate(range_A)
        for k in range_B
            row_B = SparseArrays.rowvals(B)[k]
            while SparseArrays.rowvals(A)[it_A[1]] < row_B
                it_A = iterate(range_A, it_A[2])
            end
            # By construction, since we constructed `A` with `_merge_sparsity`
            @assert row_B == SparseArrays.rowvals(A)[it_A[1]]
            SparseArrays.nonzeros(A)[it_A[1]] += SparseArrays.nonzeros(B)[k] * α
        end
    end
end

function _coefficient_product!(
    buffer::FillArrays.Zeros,
    ::ModelWorkspace,
    ::AbstractVector,
    ::MatrixIndex,
    α,
)
    return buffer
end

# Dense buffer: `𝐀ᵢ`'s `j`th column is `vec(Aᵢⱼ)`, so this computes
# `vec(buffer) = 𝐀ᵢ * y = vec(∑ⱼ Aᵢⱼ yⱼ)` with a single `Θ(nnz(𝐀ᵢ))` sparse
# matrix-vector product. `_vec` is `UnsafeArrays.uview` so it does not
# allocate, unlike `vec` which would allocate a reshaped wrapper every call.
function _coefficient_product!(
    buffer::StridedMatrix,
    model::ModelWorkspace,
    # `y` is a `SparseVector` when called from Loraine's `H_alpha`
    # preconditioner, so this must stay an `AbstractVector`.
    y::AbstractVector,
    i::MatrixIndex,
    α,
)
    _mul_coefficients!(buffer, model.coefficient_buffer[i.value], y, α)
    return buffer
end

# Sparse buffer: the result keeps the merged sparsity pattern, so it is
# accumulated one constraint at a time. See `DENSE_JPROD_DENSITY` for when
# this is preferred over the dense buffer above.
function _coefficient_product!(
    buffer::SparseArrays.SparseMatrixCSC,
    model::ModelWorkspace,
    y::AbstractVector,
    i::MatrixIndex,
    α,
)
    _zero!(buffer)
    _foreach_column(model.model.A, i.value) do values, slot
        return _add_sparse_coefficient_product!(buffer, values, slot, y, α)
    end
    return buffer
end

function _add_sparse_coefficient_product!(buffer, values, slot, y, α)
    for j in eachindex(y)
        _add_mul!(buffer, values[j, slot], α * y[j])
    end
    return nothing
end

# Setup fallback for arbitrary matrix representations. Specializations for
# sparse and rank-one matrices avoid visiting their implicit zeros.
function _add_vec!(I, J, V, j, offset, A::AbstractMatrix)
    for col in axes(A, 2), row in axes(A, 1)
        value = A[row, col]
        if !iszero(value)
            offset += 1
            I[offset] = row + (col - 1) * size(A, 1)
            J[offset] = j
            V[offset] = value
        end
    end
    return offset
end

# Products are zero, dense, or sparse matrices. Returning an alias to C or
# the product buffer is allowed by unsafe_cons's contract.
# Avoid subtraction of lazy zeros (FillArrays.jl issue #412).
_sub(A::AbstractMatrix, ::FillArrays.Zeros{<:Any,2}) = A

function _sub(A::AbstractMatrix, B::Matrix)
    B .= A .- B
    return B
end

function _sub(
    ::FillArrays.Zeros{<:Any,2},
    B::SparseArrays.SparseMatrixCSC{T},
) where {T}
    LinearAlgebra.rmul!(B, -one(T))
    return B
end

# Merging the objective may change the sparse product's cached pattern.
_sub(A::AbstractMatrix, B::SparseArrays.SparseMatrixCSC) = A - B

function _unsafe_coefficient_product(
    workspace::ModelWorkspace,
    v,
    i::MatrixIndex,
    α = true,
)
    _require_jacobian(workspace)
    return _coefficient_product!(
        workspace.jprod_buffer[i.value],
        workspace,
        v,
        i,
        α,
    )
end

"""
    unsafe_jprod(workspace::ModelWorkspace, v, i::MatrixIndex)

Return the native block JVP `-sum(A[i,j] * v[j])`. The result aliases
workspace storage and is invalidated by the next product or constraint
calculation for this block. Do not mutate it.
"""
function unsafe_jprod(
    workspace::ModelWorkspace,
    v::AbstractVector,
    i::MatrixIndex,
)
    return _unsafe_coefficient_product(workspace, v, i, -one(eltype(v)))
end

"""
    add_jtprod!(workspace::ModelWorkspace, V, result, i::MatrixIndex)

Add the native block VJP: `result[j] -= dot(A[i,j], V)`.
"""
function add_jtprod!(
    workspace::ModelWorkspace,
    V::AbstractMatrix,
    result::AbstractVector,
    i::MatrixIndex,
)
    _require_jacobian(workspace)
    return _add_coefficient_adjoint!(
        V,
        result,
        workspace.coefficient_buffer[i.value],
        -one(eltype(result)),
    )
end

function add_sub_jtprod!(
    workspace::ModelWorkspace,
    i::MatrixIndex,
    V::AbstractMatrix,
    result::AbstractVector,
    indices,
)
    _require_jacobian(workspace)
    return _add_sub_coefficient_adjoint!(
        V,
        result,
        workspace.coefficient_buffer[i.value],
        indices,
        -one(eltype(result)),
    )
end

function jprod!(
    workspace::ModelWorkspace,
    v::AbstractVector,
    result::AbstractVector,
    ::Type{ScalarIndex},
)
    return LinearAlgebra.mul!(result, workspace.scalar_jacobian_transpose', v)
end

function jtprod!(
    workspace::ModelWorkspace,
    multipliers::AbstractVector,
    result::AbstractVector,
    ::Type{ScalarIndex},
)
    return LinearAlgebra.mul!(
        result,
        workspace.scalar_jacobian_transpose,
        multipliers,
    )
end

_cone_vector(x::AbstractConeVector, ::Model) = x
function _cone_vector(x::AbstractVector, model::Model)
    return VectorizedConeVector(x, model.dim)
end

"""
    jprod!(workspace::ModelWorkspace, v, result)

Compute the native constraint Jacobian times a free-variable direction.
`v` has length `model.meta.nvar`, and `result` has length `model.meta.ncon`.
"""
function jprod!(
    workspace::ModelWorkspace,
    v::AbstractVector,
    result::AbstractVector,
)
    cone = _cone_vector(result, workspace.model)
    jprod!(workspace, v, cone[ScalarIndex], ScalarIndex)
    for i in matrix_indices(workspace)
        cone[i] .= unsafe_jprod(workspace, v, i)
    end
    return result
end

"""
    jtprod!(workspace::ModelWorkspace, multipliers, result)

Compute the native constraint Jacobian adjoint times cone multipliers.
`multipliers` has length `model.meta.ncon`, and `result` has length
`model.meta.nvar`. PSD blocks use the Frobenius inner product.
"""
function jtprod!(
    workspace::ModelWorkspace,
    multipliers::AbstractVector,
    result::AbstractVector,
)
    # Julia 1.10 otherwise allocates each flat block view passed to sparse
    # multiplication. Keep the owning vector alive while using unsafe views.
    UnsafeArrays.@uviews multipliers begin
        cone = _cone_vector(multipliers, workspace.model)
        jtprod!(workspace, cone[ScalarIndex], result, ScalarIndex)
        for i in matrix_indices(workspace)
            add_jtprod!(workspace, cone[i], result, i)
        end
    end
    return result
end

function cons!(
    workspace::ModelWorkspace,
    x::AbstractVector,
    result::AbstractVector,
    ::Type{ScalarIndex},
)
    copyto!(result, view(workspace.model.constant, 1:num_scalars(workspace)))
    return LinearAlgebra.mul!(
        result,
        workspace.scalar_jacobian_transpose',
        x,
        true,
        true,
    )
end

"""
    unsafe_cons(workspace::ModelWorkspace, x, i::MatrixIndex)

Return the PSD constraint `C[i] - sum(A[i,j] * x[j])`. The result can alias
`C[i]` or workspace storage. Do not mutate it or retain it across calls.
"""
function unsafe_cons(
    workspace::ModelWorkspace,
    x::AbstractVector,
    i::MatrixIndex,
)
    return _sub(
        workspace.model.C[i.value],
        _unsafe_coefficient_product(workspace, x, i),
    )
end

function cons!(
    workspace::ModelWorkspace,
    x::AbstractVector,
    result::AbstractVector,
)
    cone = _cone_vector(result, workspace.model)
    cons!(workspace, x, cone[ScalarIndex], ScalarIndex)
    for i in matrix_indices(workspace)
        cone[i] .= unsafe_cons(workspace, x, i)
    end
    return result
end
