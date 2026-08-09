mutable struct BufferedModelForSchur{
    T,
    C<:AbstractMatrix{T},
    A<:AbstractMatrix{T},
    JB,
    JTB,
    SB,
} <: AbstractModel{T}
    model::Model{T,C,A}
    meta::NLPModels.NLPModelMeta{T,Vector{T}}
    jprod_buffer::JB
    jtprod_buffer::JTB
    schur_buffer::SB
end

function BufferedModelForSchur(model, datasparsity)
    return BufferedModelForSchur(
        model,
        model.meta,
        buffer_for_jprod(model),
        buffer_for_jtprod(model),
        buffer_for_schur_complement(model, datasparsity),
    )
end

num_scalars(model::BufferedModelForSchur) = num_scalars(model.model)
num_matrices(model::BufferedModelForSchur) = num_matrices(model.model)
matrix_indices(model::BufferedModelForSchur) = matrix_indices(model.model)
side_dimension(model::BufferedModelForSchur, i) = side_dimension(model.model, i)

#######################
###### Objective ######
#######################

function NLPModels.obj(model::BufferedModelForSchur, x::AbstractVector)
    return NLPModels.obj(model.model, x)
end

function dual_obj(model::BufferedModelForSchur, y::AbstractVector)
    return dual_obj(model.model, y)
end

function grad(model::BufferedModelForSchur, ::Type{ScalarIndex})
    return grad(model.model, ScalarIndex)
end

function grad(model::BufferedModelForSchur, i::MatrixIndex)
    return grad(model.model, i)
end

#########################
###### Constraints ######
#########################

cons_constant(model::BufferedModelForSchur) = cons_constant(model.model)
function jac(model::BufferedModelForSchur, j::Integer, ::Type{ScalarIndex})
    return jac(model.model, j, ScalarIndex)
end
function norm_jac(model::BufferedModelForSchur, i::MatrixIndex)
    return norm_jac(model.model, i)
end

#######################
###### J product ######
#######################

function jprod!(model, x, v, Jv, ::Type{ScalarIndex})
    return jprod!(model.model, x, v, Jv, ScalarIndex)
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
    buffer_for_jprod(model::Model, i::MatrixIndex)

Return the sparse matrix collecting the vectorization of every constraint
matrix of the `i`th PSD block:
```
𝐀ᵢ = [vec(Aᵢ₁) vec(Aᵢ₂) … vec(Aᵢₙ)] ∈ ℝ^(mᵢ² × n)
```
where `mᵢ` is the side dimension of the block and `n` is the number of
constraints. This is the matrix `𝒜` of [HKS24, Section 3.1], which is
stated there for a single PSD block, here built once per block.

It represents the linear equality-constraint operator
`Xᵢ ↦ (⟨Aᵢⱼ, Xᵢ⟩)ⱼ` as a single matrix, so that `add_jprod!` is one
sparse matrix-vector product. It is computed once at problem setup and
reused at every interior-point iteration, both for the Jacobian products
and for the dense columns of the Schur complement assembled in
`schur.jl`.

`SparseMatrixCSC` is stored with an offset by column.
This means that getting view `view(A, :, I)` can be handles efficently,
these give `SparseMatrixCSCView` (if `I` is a `UnitRange`) and
`SparseMatrixCSCColumnSubset` otherwise.
In `schur.jl`, we therefore get a `SparseMatrixCSCColumnSubset`.
Since we want to use subsets of constraint indices, we use the columns
of `A` for constraint indices and the rows of `A` for matrix indices.
The subsets used in `schur.jl` are suffixes of the constraints sorted by
decreasing number of nonzeros, following [FKN97].

[HKS24] Habibi, Soodeh, Michal Kočvara, and Michael Stingl. "Loraine -- an
interior-point solver for low-rank semidefinite programming."
Optimization Methods and Software 39.6 (2024): 1185-1215.
[FKN97] Fujisawa, Katsuki, Masakazu Kojima, and Kazuhide Nakata.
"Exploiting sparsity in primal-dual interior-point methods for
semidefinite programming." Mathematical Programming 79 (1997): 235-253.
"""
function buffer_for_jprod(model::Model{T}, i::MatrixIndex) where {T}
    nnz = sum(1:(model.meta.ncon); init = 0) do j
        return _nnz(model.A[i.value, j])
    end
    I = zeros(Int64, nnz)
    J = zeros(Int64, nnz)
    V = zeros(T, nnz)
    offset = 0
    for j in 1:(model.meta.ncon)
        offset = _add_vec!(I, J, V, j, offset, model.A[i.value, j])
    end
    A = SparseArrays.sparse(
        I,
        J,
        V,
        side_dimension(model, i)^2,
        model.meta.ncon,
    )
    return A
end

function buffer_for_jprod(model::Model{T}) where {T}
    return SparseArrays.SparseMatrixCSC{T,Int64}[
        buffer_for_jprod(model, i) for i in matrix_indices(model)
    ]
end

_vec(x::AbstractVector) = x
_vec(x::FillArrays.Zeros{T}) where {T} = FillArrays.Zeros{T}(length(x))
_vec(x::AbstractArray) = UnsafeArrays.uview(x, :)
_vec(x::Base.ReshapedArray) = _vec(parent(x))

function _add_jprod!(V, Jv::AbstractArray, A)
    return _add_mul!(Jv, A', _vec(V), true)
end

"""
    add_sub_jprod!(
        model::BufferedModelForSchur,
        i::MatrixIndex,
        V::AbstractMatrix,
        Jv::AbstractVector,
        I,
    )

Same as `add_jprod!` but restricted to the constraints of `I`:
adds `⟨Aᵢⱼ, V⟩` to `Jv[k]` for the `k`th entry `j` of `I`. This is the
product with the column subset `𝐀ᵢ[:, I]` of the buffer described in
`buffer_for_jprod`, which is why the buffer is stored with
constraints as columns.
"""
function add_sub_jprod!(
    model::BufferedModelForSchur,
    i::MatrixIndex,
    V::AbstractMatrix,
    Jv::AbstractVector,
    I,
)
    # `view(cache, I)` would be terribly slow, only the number of elements of `I` matter here
    A = model.jprod_buffer[i.value]
    return _add_jprod!(V, Jv, view(A, :, I))
end

"""
    add_jprod!(
        model::BufferedModelForSchur,
        V::AbstractMatrix,
        Jv::AbstractVector,
        i::MatrixIndex,
    )

Add the contribution of the `i`th PSD block to the product between the
Jacobian of the equality constraints and `V`, that is, add `𝐀ᵢᵀ vec(V)`
to `Jv`, whose `j`th entry is `⟨Aᵢⱼ, V⟩`. Here `𝐀ᵢ` is the buffer built
by `buffer_for_jprod` so this is a single sparse matrix-vector
product.
"""
function add_jprod!(
    model::BufferedModelForSchur,
    V::AbstractMatrix,
    Jv::AbstractVector,
    i::MatrixIndex,
)
    return _add_jprod!(V, Jv, model.jprod_buffer[i.value])
end

########################
###### Jᵀ product ######
########################

function jtprod!(
    model::BufferedModelForSchur,
    y::AbstractVector,
    vJ::AbstractVector,
    ::Type{ScalarIndex},
)
    return jtprod!(model.model, y, vJ, ScalarIndex)
end

function buffer_for_jtprod(model::Model)
    if iszero(num_matrices(model))
        return
    end
    return map(Base.Fix1(buffer_for_jtprod, model), matrix_indices(model))
end

function buffer_for_jtprod(model::Model{T}, mat_idx::MatrixIndex) where {T}
    d = side_dimension(model, mat_idx)
    # If every `Aᵢⱼ` is zero then so is the product, and callers rely on
    # getting a `Zeros` back (`_sub` then aliases `C` instead of copying).
    if iszero(model.meta.ncon) ||
       all(j -> iszero(_nnz(model.A[mat_idx.value, j])), 1:(model.meta.ncon))
        return FillArrays.Zeros{T}(d, d)
    end
    # Dense: `unsafe_jtprod` fills this with a single sparse matrix-vector
    # product against the `buffer_for_jprod` matrix, and the result is
    # immediately multiplied by dense matrices by `eval_schur_complement!`.
    return zeros(T, d, d)
end

function NLPModels.jtprod!(
    model::BufferedModelForSchur,
    _::AbstractVector,
    y::AbstractVector,
    vJ::AbstractVector,
)
    jtprod!(model, y, vJ[ScalarIndex], ScalarIndex)
    for mat_idx in matrix_indices(model)
        vJ[mat_idx] .= unsafe_jtprod(model, y, mat_idx)
    end
end

function _jtprod!(
    buffer::FillArrays.Zeros,
    ::SparseArrays.SparseMatrixCSC,
    ::AbstractVector,
)
    return buffer
end

function _jtprod!(
    buffer::StridedMatrix,
    A::SparseArrays.SparseMatrixCSC,
    # `y` is a `SparseVector` when called from Loraine's `H_alpha`
    # preconditioner, so this must stay an `AbstractVector`.
    y::AbstractVector,
)
    # `A` is the `buffer_for_jprod` matrix: its `j`th column is `vec(Aᵢⱼ)`, so
    # this computes `vec(buffer) = A * y = vec(∑ⱼ Aᵢⱼ yⱼ)` in `Θ(nnz(A))`.
    # Scattering each `Aᵢⱼ` into `buffer` one at a time instead costs
    # `Θ(ncon * side_dimension)` in loop overhead alone, which dominates
    # everything else when this is called once per CG iteration.
    # `_vec` is `UnsafeArrays.uview` so it does not allocate, unlike `vec`
    # which would allocate a reshaped wrapper on every call.
    LinearAlgebra.mul!(_vec(buffer), A, y)
    return buffer
end

"""
    unsafe_jtprod(model::BufferedModelForSchur, y, i::MatrixIndex)

Return the product between the sub-Jacobian associated to `i` and `y`:
`∑ⱼ Aᵢⱼ yⱼ`. This function is **unsafe** because it returns
an alias to an internal buffer `model.jtprod_buffer[i.value]` that is
going to be reused whenever `unsafe_jprod` or `unsafe_dual_cons` is called
is called for the same `i`.
"""
function unsafe_jtprod(model::BufferedModelForSchur, y, i::MatrixIndex)
    buffer = model.jtprod_buffer[i.value]
    return _jtprod!(buffer, model.jprod_buffer[i.value], y)
end

function dual_cons!(
    model::BufferedModelForSchur,
    y::AbstractVector,
    res,
    ::Type{ScalarIndex},
)
    return dual_cons!(model.model, y, res, ScalarIndex)
end

# TODO If we rename `dual_cons!` to `unsafe_dual_cons`,
#      we can remove the `copy` and remplace `-B` with a mutation

# Note that we can't use `-` because of https://github.com/JuliaArrays/FillArrays.jl/issues/412
_sub(A::FillArrays.Zeros, ::FillArrays.Zeros) = A
# /!\ We don't copy, we warn about it in the docstring.
_sub(A::AbstractArray, ::FillArrays.Zeros) = A
function _sub(::FillArrays.Zeros, B::AbstractArray{T}) where {T}
    # `B` is an alias to an entry of `jtprod_buffer`
    # so it is safe to modify.
    # The fact that we may return an alias to this matrix
    # since we don't copy is warned in the docstring/`
    LinearAlgebra.rmul!(B, -one(T))
    return B
end
_sub(A::AbstractArray, B::AbstractArray) = A - B

"""
    unsafe_dual_cons(model::BufferedModelForSchur, y, i::MatrixIndex)

Return value of the dual constraint of index `i`:
`C - ∑ⱼ Aᵢⱼ yⱼ`. This function is **unsafe** because it may return
an alias to the matrix `C` or the internal buffer
`model.jtprod_buffer[i.value]`. So throw away the reference after using
the returned value and don't call any mutating operation on it.
"""
function unsafe_dual_cons(
    model::BufferedModelForSchur,
    y::AbstractVector,
    i::MatrixIndex,
)
    return _sub(model.model.C[i.value], unsafe_jtprod(model, y, i))
end
