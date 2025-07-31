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

function NLPModels.grad(model::BufferedModelForSchur, ::Type{ScalarIndex})
    return NLPModels.grad(model.model, ScalarIndex)
end

function NLPModels.grad(model::BufferedModelForSchur, i::MatrixIndex)
    return NLPModels.grad(model.model, i)
end

#########################
###### Constraints ######
#########################

cons_constant(model::BufferedModelForSchur) = cons_constant(model.model)
function NLPModels.jac(
    model::BufferedModelForSchur,
    j::Integer,
    ::Type{ScalarIndex},
)
    return NLPModels.jac(model.model, j, ScalarIndex)
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

# `SparseMatrixCSC` is stored with an offset by column.
# This means that getting view `view(A, :, I)` can be handles efficently,
# these give `SparseMatrixCSCView` (if `I` is a `UnitRange`) and
# `SparseMatrixCSCColumnSubset` otherwise.
# In `schur.jl`, we therefore get a `SparseMatrixCSCColumnSubset`.
# Since we want to use subsets of constraint indices, we use the columns
# of `A` for constraint indices and the rows of `A` for matrix indices.
function buffer_for_jprod(model::Model{T}, i::MatrixIndex) where {T}
    nnz = sum(1:model.meta.ncon; init = 0) do j
        return _nnz(model.A[i.value, j])
    end
    I = zeros(Int64, nnz)
    J = zeros(Int64, nnz)
    V = zeros(T, nnz)
    offset = 0
    for j in 1:model.meta.ncon
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

function _add_jprod!(V, Jv::AbstractArray{T}, A) where {T}
    return LinearAlgebra.mul!(Jv, A', _vec(V), true, true)
end

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

function buffer_for_jtprod(model::Model{T}, mat_idx::MatrixIndex) where {T}
    if iszero(model.meta.ncon)
        d = side_dimension(model, mat_idx)
        return FillArrays.Zeros{T}(d, d)
    end
    # FIXME: at some point, switch to dense
    # /!\ If there is only one nonzero matrix and we didn't have `_abs`,
    #     we would return an alias of that only matrix so that `_abs` has the
    #     non-obvious role of avoid this as well as avoiding cancellations.
    return reduce(
        _merge_sparsity,
        _abs(model.A[mat_idx.value, j]) for j in 1:model.meta.ncon
    )
end

function NLPModels.jtprod!(
    model::BufferedModelForSchur,
    _::AbstractVector,
    y::AbstractVector,
    vJ::AbstractVector,
)
    jtprod!(model, y, vJ[ScalarIndex], ScalarIndex)
    for mat_idx in matrix_indices(model)
        vJ[mat_idx] .= jtprod!(model, y, mat_idx)
    end
end

_zero!(A::FillArrays.Zeros) = (@show @__LINE__; A)
_zero!(A::SparseArrays.SparseMatrixCSC) = fill!(SparseArrays.nonzeros(A), 0.0)

# Computes `A .+= B * α`
function _add_mul!(::FillArrays.Zeros, ::FillArrays.Zeros, _) end

function _add_mul!(A::SparseArrays.SparseMatrixCSC, ::FillArrays.Zeros, _)
    return A
end

function _add_mul!(
    A::SparseArrays.SparseMatrixCSC,
    B::SparseArrays.SparseMatrixCSC,
    α,
)
    for col in axes(A, 2)
        range_A = SparseArrays.nzrange(A, col)
        it_A = iterate(range_A)
        for k in SparseArrays.nzrange(B, col)
            row_B = SparseArrays.rowvals(B)[k]
            while SparseArrays.rowvals(A)[it_A[1]] < row_B
                it_A = iterate(range_A, it_A[2])
            end
            @assert row_B == SparseArrays.rowvals(A)[it_A[1]]
            SparseArrays.nonzeros(A)[it_A[1]] += SparseArrays.nonzeros(B)[k] * α
        end
    end
end

function jtprod!(model::Model, y, buffer, mat_idx::MatrixIndex)
    _zero!(buffer)
    for j in eachindex(y)
        _add_mul!(buffer, model.A[mat_idx.value, j], y[j])
    end
    return buffer
end

function jtprod!(model::BufferedModelForSchur, y, mat_idx::MatrixIndex)
    return jtprod!(model.model, y, model.jtprod_buffer[mat_idx.value], mat_idx)
end

function dual_cons!(
    model::BufferedModelForSchur,
    y::AbstractVector,
    res,
    ::Type{ScalarIndex},
)
    return dual_cons!(model.model, y, res, ScalarIndex)
end

# Note that we can't use `-` because of https://github.com/JuliaArrays/FillArrays.jl/issues/412
_sub(A::AbstractArray, ::FillArrays.Zeros) = copy(A)
_sub(A::AbstractArray, B::AbstractArray) = A - B

function dual_cons!(
    model::BufferedModelForSchur,
    y::AbstractVector,
    i::MatrixIndex,
)
    return _sub(model.model.C[i.value], jtprod!(model, y, i))
end
