# This code computes the Schur complement using the ideas detailed in [FKN97, Section 3]
# This is useful to compute search direction in primal-dual interior-point methods for semidefinite programs [FKN97]
# [FKN97] Fujisawa, Katsuki, Masakazu Kojima, and Kazuhide Nakata. "Exploiting sparsity in primal-dual interior-point methods for semidefinite programming." Mathematical Programming 79 (1997): 235-253.
# [HKS24] Habibi, Soodeh, Michal Kočvara, and Michael Stingl. "Loraine -- an interior-point solver for low-rank semidefinite programming." Optimization Methods and Software 39.6 (2024): 1185-1215.
# It was adapted from dapted from Michal Kocvara's code in
# https://github.com/kocvara/Loraine.jl/blob/bd2821ba830786a78f04081d7e8f5cac25e56cac/src/makeBBBB.jl

# Computes `⟨A * W, W * B⟩` for symmetric sparse matrices `A` and `B`
function _dot(
    A::SparseArrays.SparseMatrixCSC,
    B::SparseArrays.SparseMatrixCSC,
    W::AbstractMatrix,
)
    @assert LinearAlgebra.checksquare(W) ==
            LinearAlgebra.checksquare(A) ==
            LinearAlgebra.checksquare(B)
    # After these asserts, we know that `A`, `B` and `W` are square and
    # have the same sizes so we can safely use `@inbounds`
    result = zero(eltype(A))
    @inbounds for i in axes(A, 2)
        nzA = SparseArrays.nzrange(A, i)
        if !isempty(nzA)
            for j in axes(B, 2)
                nzB = SparseArrays.nzrange(B, j)
                if !isempty(nzB)
                    AW = zero(result)
                    for k in nzA
                        AW +=
                            SparseArrays.nonzeros(A)[k] *
                            W[SparseArrays.rowvals(A)[k], j]
                    end
                    WB = zero(result)
                    for k in nzB
                        WB +=
                            W[i, SparseArrays.rowvals(B)[k]] *
                            SparseArrays.nonzeros(B)[k]
                    end
                    result += AW * WB
                end
            end
        end
    end
    return result
end

_nnz(::FillArrays.Zeros) = 0
_nnz(A::SparseArrays.SparseMatrixCSC) = SparseArrays.nnz(A)

# The `jprod!` buffer is guaranteed to be the first argument of the tuple.
# This assumption is used by Loraine.

function add_schur_complement!(
    model::BufferedModelForSchur,
    W,
    ::Type{MatrixIndex},
    H,
)
    for i in matrix_indices(model)
        add_schur_complement!(model, i, W[i], H)
    end
    return H
end

# Adds one sparse component's contribution `𝐀ᵢᵀ (W ⊗ W) 𝐀ᵢ` to `H`.
# Its Jacobian cache follows the layout documented in `buffer_for_jprod`.
# `H` is assembled column by column. For a constraint `i` in the dense
# regime, that is, one of the `last_dense[ilmi]` first entries of the
# permutation `σ` sorting the constraints by decreasing number of nonzeros,
# the dense matrix `W Aᵢ W` is formed once and all the remaining inner
# products `⟨Aⱼ, W Aᵢ W⟩`, `j ∈ I`, are then obtained with the single
# sparse matrix-vector product `𝐀ᵢ[:, I]ᵀ vec(W Aᵢ W)` of
# `add_sub_jprod!`. The constraints with at most one nonzero use the
# dedicated low-nnz paths below, also following [FKN97].
# /!\ W needs to be symmetric
function _add_sparse_component!(workspace, slot, W, AW, WAW, H)
    σ, last_dense = workspace.permutation, workspace.last_dense
    column = view(workspace.component.values, :, slot)
    n = length(column)

    for ii in axes(H, 1)
        i = σ[ii, slot]
        Ai = column[i]
        if _nnz(Ai) > 0
            if ii <= last_dense[slot]
                LinearAlgebra.mul!(AW, W, Ai)
                LinearAlgebra.mul!(WAW, AW, W)
                I = view(σ, ii:n, slot)
                _add_jprod!(
                    WAW,
                    view(H, I, i),
                    view(workspace.jacobian[slot], :, I),
                )
                for jj in (ii+1):n
                    j = σ[jj, slot]
                    H[i, j] = H[j, i]
                end
            else
                if _nnz(Ai) > 1
                    @inbounds for jj in ii:n
                        j = σ[jj, slot]
                        Aj = column[j]
                        if !iszero(_nnz(Aj))
                            ttt = _dot(Ai, Aj, W)
                            H[i, j] += ttt
                            if i != j
                                H[j, i] += ttt
                            end
                        end
                    end
                elseif _nnz(Ai) == 1
                    # A is symmetric
                    iiiiAi = jjjiAi = only(SparseArrays.rowvals(Ai))
                    vvvi = only(SparseArrays.nonzeros(Ai))
                    @inbounds for jj in ii:n
                        j = σ[jj, slot]
                        Ajjj = column[j]
                        # As we sort the matrices in decreasing `nnz` order,
                        # the rest of matrices is either zero or have only
                        # one entry
                        if !iszero(_nnz(Ajjj))
                            iiijAj = jjjjAj = only(SparseArrays.rowvals(Ajjj))
                            vvvj = only(SparseArrays.nonzeros(Ajjj))
                            ttt =
                                vvvi *
                                W[iiiiAi, iiijAj] *
                                W[jjjiAi, jjjjAj] *
                                vvvj
                            H[i, j] += ttt
                            if i != j
                                H[j, i] += ttt
                            end
                        end
                    end
                end
            end
        end
    end
    return H
end

# Keep sparse output on the sparse-product path: repeated sparse axpy! updates
# would search its index structure once per contribution.
function add_schur_complement!(
    model::BufferedModelForSchur,
    w::StridedVector,
    ::Type{ScalarIndex},
    H::StridedMatrix,
)
    C = model.model.C_lin
    # C * Diagonal(w) * C'
    # = sum_k w[k] * c_k * c_k'.
    # = sum_k sum_j w[k] * c_k * (conj(c_{jk}) * e_j')
    # Sparse `axpy!` visits only stored entries and needs no intermediate matrix.
    rows, values = SparseArrays.rowvals(C), SparseArrays.nonzeros(C)
    for k in axes(C, 2)
        column = view(C, :, k)
        for p in SparseArrays.nzrange(C, k)
            j = Int(rows[p])
            LinearAlgebra.axpy!(w[k] * conj(values[p]), column, view(H, :, j))
        end
    end
    return H
end

function _add_sparse_eval_column!(result, values, slot, Ay, W)
    for k in eachindex(result)
        A = values[k, slot]
        if !iszero(_nnz(A))
            result[k] += _dot(A, Ay, W)
        end
    end
    return nothing
end

# [HKS24, (5b)]
# Returns the matrix equal to the sum, for each equation, of
# ⟨A_i, WA_jW⟩
# [HKS24, (5b)] is stated for a single PSD block; summing over the blocks
# and adding the contribution of the scalar block gives
# H = ∑ᵢ 𝐀ᵢᵀ (Wᵢ ⊗ Wᵢ) 𝐀ᵢ + C_lin * Diagonal(w) * C_linᵀ
function schur_complement!(model::BufferedModelForSchur, W::AbstractVector, H)
    fill!(H, zero(eltype(H)))
    if num_matrices(model) > 0
        add_schur_complement!(model, W, MatrixIndex, H)
    end
    if num_scalars(model) > 0
        add_schur_complement!(model, W[ScalarIndex], ScalarIndex, H)
    end
    return H
end

# Add `⟨Aᵢₖ, W A(y) W⟩` to `result[k]`, where `Ay` is `A(y) = ∑ⱼ Aᵢⱼ yⱼ` as
# returned by `unsafe_jtprod`.
#
# `A(y)` dense: form `W A(y) W` and contract it against `𝐀ᵢ` with one sparse
# matrix-vector product. Two `d×d` products, so `Θ(d³)` however sparse the
# `Aᵢₖ` are.
function _add_eval_schur_complement!(
    model::BufferedModelForSchur,
    W::AbstractMatrix,
    Ay::AbstractMatrix,
    result,
    i::MatrixIndex,
)
    AW, WAW, _, _ = model.schur_buffer
    LinearAlgebra.mul!(AW[i.value], W, Ay)
    LinearAlgebra.mul!(WAW[i.value], AW[i.value], W)
    return add_jprod!(model, WAW[i.value], result, i)
end

# `A(y)` sparse: contract directly, never forming the `d×d` product, so the
# cost follows the nonzeros instead of `d`. This is the same trade-off
# `add_schur_complement!` makes per constraint matrix via `κ`/`last_dense`,
# here made once for `A(y)`.
function _add_eval_schur_complement!(
    model::BufferedModelForSchur,
    W::AbstractMatrix,
    Ay::SparseArrays.SparseMatrixCSC,
    result,
    i::MatrixIndex,
)
    _foreach_column(model.model.A, i.value) do values, slot
        return _add_sparse_eval_column!(result, values, slot, Ay, W)
    end
    return result
end

# [HKS24, (5b)]
# Returns the matrix equal to the sum, for each equation, of
# ⟨A_i, WA(y)W⟩
function eval_schur_complement!(model::BufferedModelForSchur, W, y, result)
    fill!(result, zero(eltype(result)))
    components = model.schur_buffer[3]
    for i in matrix_indices(model)
        if _has_general_component(components, i.value)
            _add_eval_schur_complement!(
                model,
                W[i],
                unsafe_jtprod(model, y, i),
                result,
                i,
            )
        else
            _prepare_components!(components, i.value, W[i])
            _add_component_matvec!(components, i.value, y, result)
        end
    end
    # In the future, we'd want to allocate intermediate buffers for this operations
    # to make it allocation-free. In the short-term, we just add this `if`
    # to be allocation-free when there are no scalars to prevent a regerssion
    # in https://github.com/kocvara/Loraine.jl/pull/29
    if !iszero(num_scalars(model))
        result .+=
            model.model.C_lin * (W[ScalarIndex] .* (model.model.C_lin' * y))
    end
    return result
end

# Schur workspaces follow individual components, not the type of the whole
# model. A rank-one component can coexist with arbitrary other components.
const _RankOneFactorization{T} =
    Factorization{T,<:AbstractVector{T},<:AbstractArray{T,0}}

_nnz(A::AbstractMatrix) = count(!iszero, A)

struct _GeneralSchurComponent{C,J}
    component::C
    jacobian::J
    permutation::Matrix{Int}
    last_dense::Vector{Int}
    sparse::Vector{Bool}
end

struct _RankOneSchurComponent{T,C}
    component::C
    U::Vector{Matrix{T}}
    weights::Matrix{T}
    UW::Vector{Matrix{T}}
    gram::Matrix{T}
end

function _schur_component(component::BlockColumns, model::Model{T}, κ) where {T}
    n, p = size(component.values)
    permutation = Matrix{Int}(undef, n, p)
    last_dense = zeros(Int, p)
    sparse = Vector{Bool}(undef, p)
    jacobian = SparseArrays.SparseMatrixCSC{T,Int64}[]
    for (slot, i) in enumerate(component.block_ids)
        column = view(component.values, :, slot)
        counts = _nnz.(column)
        permutation[:, slot] = sortperm(counts; rev = true)
        last_dense[slot] = count(>(κ), counts)
        sparse[slot] = all(
            A -> A isa Union{SparseArrays.SparseMatrixCSC,FillArrays.Zeros},
            column,
        )
        push!(
            jacobian,
            _jacobian_buffer((component,), i, model.msizes[i], n, T),
        )
    end
    return _GeneralSchurComponent(
        component,
        jacobian,
        permutation,
        last_dense,
        sparse,
    )
end

function _schur_component(
    component::BlockColumns{<:_RankOneFactorization{T}},
    model::Model{T},
    _,
) where {T}
    n, p = size(component.values)
    U = Matrix{T}[]
    UW = Matrix{T}[]
    weights = Matrix{T}(undef, n, p)
    for (slot, i) in enumerate(component.block_ids)
        Ui = Matrix{T}(undef, n, model.msizes[i])
        for j in 1:n
            A = component.values[j, slot]
            Ui[j, :] = A.factor
            weights[j, slot] = only(A.scaling)
        end
        push!(U, Ui)
        push!(UW, similar(Ui))
    end
    return _RankOneSchurComponent(component, U, weights, UW, zeros(T, n, n))
end

function buffer_for_schur_complement(model::Model{T}, κ) where {T}
    AW = [zeros(T, d, d) for d in model.msizes]
    WAW = copy.(AW)
    components = map(A -> _schur_component(A, model, κ), model.A)
    factor_work = map(matrix_indices(model)) do i
        rank = _sum_columns(model.A, i.value, 0) do column
            return maximum(
                A -> A isa AbstractFactorization ? max_rank(A) : 0,
                column;
                init = 0,
            )
        end
        return zeros(T, model.msizes[i.value], rank)
    end
    # Loraine uses the first two per-block dense workspaces directly.
    return AW, WAW, components, factor_work
end

_prepare_schur!(::_GeneralSchurComponent, _, _) = nothing
function _prepare_schur!(workspace::_RankOneSchurComponent, i, W)
    slot = _block_slot(workspace.component, i)
    if !iszero(slot)
        LinearAlgebra.mul!(workspace.UW[slot], workspace.U[slot], W)
    end
    return nothing
end

@inline _prepare_components!(::Tuple{}, i, W) = nothing
@inline function _prepare_components!(components::Tuple, i, W)
    _prepare_schur!(first(components), i, W)
    _prepare_components!(Base.tail(components), i, W)
    return nothing
end

function _sandwich!(WAW, AW, W, A::AbstractMatrix, _)
    LinearAlgebra.mul!(AW, W, A)
    return LinearAlgebra.mul!(WAW, AW, W)
end
function _sandwich!(
    WAW,
    AW,
    W,
    A::AbstractFactorization{T,<:AbstractVector},
    work,
) where {T}
    buffered_mul!(AW, A, W, true, false, view(work, :, 1))
    return LinearAlgebra.mul!(WAW, W, AW)
end
function _sandwich!(
    WAW,
    AW,
    W,
    A::AbstractFactorization{T,<:AbstractMatrix},
    work,
) where {T}
    buffered_mul!(AW, A, W, true, false, view(work, :, 1:max_rank(A)))
    return LinearAlgebra.mul!(WAW, W, AW)
end

function _add_schur_pair!(
    a::_GeneralSchurComponent,
    b::_GeneralSchurComponent,
    i,
    W,
    AW,
    WAW,
    work,
    H,
    same,
)
    ia, ib = _block_slot(a.component, i), _block_slot(b.component, i)
    (iszero(ia) || iszero(ib)) && return nothing
    if same && a.sparse[ia]
        _add_sparse_component!(a, ia, W, AW, WAW, H)
        return nothing
    end
    ca, cb = view(a.component.values, :, ia), view(b.component.values, :, ib)
    for j in eachindex(ca)
        iszero(_nnz(ca[j])) && continue
        _sandwich!(WAW, AW, W, ca[j], work)
        for k in (same ? j : 1):length(cb)
            value = _coefficient_dot(cb[k], WAW)
            H[k, j] += value
            if !same || k != j
                H[j, k] += value
            end
        end
    end
    return nothing
end

# For A_j = d_j b_j b_j', the pair contraction is
# d_j e_k (b_j' W c_k)^2. Distinct components contribute this matrix
# AND its transpose; omitting the transpose would lose cross terms.
function _add_schur_pair!(
    a::_RankOneSchurComponent,
    b::_RankOneSchurComponent,
    i,
    W,
    AW,
    WAW,
    work,
    H,
    same,
)
    ia, ib = _block_slot(a.component, i), _block_slot(b.component, i)
    (iszero(ia) || iszero(ib)) && return nothing
    LinearAlgebra.mul!(a.gram, a.UW[ia], b.U[ib]')
    for k in axes(H, 2), j in axes(H, 1)
        value = a.weights[j, ia] * b.weights[k, ib] * abs2(a.gram[j, k])
        H[j, k] += value
        if !same
            H[k, j] += value
        end
    end
    return nothing
end

_quadratic_form(u, A::AbstractMatrix) = LinearAlgebra.dot(u, A, u)
function _quadratic_form(
    u,
    A::AbstractFactorization{T,<:AbstractVector},
) where {T}
    return only(A.scaling) *
           LinearAlgebra.dot(u, left_factor(A)) *
           LinearAlgebra.dot(right_factor(A), u)
end
function _quadratic_form(
    u,
    A::AbstractFactorization{T,<:AbstractMatrix},
) where {T}
    result = zero(T)
    for k in eachindex(A.scaling)
        result +=
            A.scaling[k] *
            LinearAlgebra.dot(u, view(left_factor(A), :, k)) *
            LinearAlgebra.dot(view(right_factor(A), :, k), u)
    end
    return result
end

function _add_schur_pair!(
    a::_RankOneSchurComponent,
    b::_GeneralSchurComponent,
    i,
    W,
    AW,
    WAW,
    work,
    H,
    same,
)
    ia, ib = _block_slot(a.component, i), _block_slot(b.component, i)
    (iszero(ia) || iszero(ib)) && return nothing
    column = view(b.component.values, :, ib)
    for j in axes(H, 1)
        u = view(a.UW[ia], j, :)
        for k in eachindex(column)
            value = a.weights[j, ia] * _quadratic_form(u, column[k])
            H[j, k] += value
            H[k, j] += value
        end
    end
    return nothing
end
function _add_schur_pair!(
    a::_GeneralSchurComponent,
    b::_RankOneSchurComponent,
    i,
    W,
    AW,
    WAW,
    work,
    H,
    same,
)
    return _add_schur_pair!(b, a, i, W, AW, WAW, work, H, same)
end

@inline _add_cross_pairs!(a, ::Tuple{}, args...) = nothing
@inline function _add_cross_pairs!(a, others::Tuple, args...)
    _add_schur_pair!(a, first(others), args..., false)
    _add_cross_pairs!(a, Base.tail(others), args...)
    return nothing
end
@inline _add_component_pairs!(::Tuple{}, args...) = nothing
@inline function _add_component_pairs!(components::Tuple, args...)
    a, others = first(components), Base.tail(components)
    _add_schur_pair!(a, a, args..., true)
    _add_cross_pairs!(a, others, args...)
    _add_component_pairs!(others, args...)
    return nothing
end

function add_schur_complement!(
    model::BufferedModelForSchur,
    i::MatrixIndex,
    W::AbstractMatrix,
    H,
)
    AW, WAW, components, work = model.schur_buffer
    _prepare_components!(components, i.value, W)
    _add_component_pairs!(
        components,
        i.value,
        W,
        AW[i.value],
        WAW[i.value],
        work[i.value],
        H,
    )
    return H
end

_has_general_component(::Tuple{}, i) = false
function _has_general_component(components::Tuple, i)
    a = first(components)
    return (
        a isa _GeneralSchurComponent && !iszero(_block_slot(a.component, i))
    ) || _has_general_component(Base.tail(components), i)
end

function _add_rank_one_matvec!(
    a::_RankOneSchurComponent,
    b::_RankOneSchurComponent,
    i,
    y,
    result,
    same,
)
    ia, ib = _block_slot(a.component, i), _block_slot(b.component, i)
    (iszero(ia) || iszero(ib)) && return nothing
    LinearAlgebra.mul!(a.gram, a.UW[ia], b.U[ib]')
    for k in eachindex(y), j in eachindex(result)
        value = a.weights[j, ia] * b.weights[k, ib] * abs2(a.gram[j, k])
        result[j] += value * y[k]
        if !same
            result[k] += value * y[j]
        end
    end
    return nothing
end
_add_rank_one_matvec!(::_GeneralSchurComponent, _, args...) = nothing
function _add_rank_one_matvec!(
    ::_RankOneSchurComponent,
    ::_GeneralSchurComponent,
    args...,
)
    return nothing
end
@inline _add_cross_matvec!(a, ::Tuple{}, args...) = nothing
@inline function _add_cross_matvec!(a, others::Tuple, args...)
    _add_rank_one_matvec!(a, first(others), args..., false)
    _add_cross_matvec!(a, Base.tail(others), args...)
    return nothing
end
@inline _add_component_matvec!(::Tuple{}, args...) = nothing
@inline function _add_component_matvec!(components::Tuple, args...)
    a, others = first(components), Base.tail(components)
    _add_rank_one_matvec!(a, a, args..., true)
    _add_cross_matvec!(a, others, args...)
    _add_component_matvec!(others, args...)
    return nothing
end

# Rank-one entries in the setup-time sparse Jacobian buffer. The component
# Schur kernels use their vector factors directly, while the combined
# Jacobian buffer supports products of sums without extra iteration storage.

# Nonzero (index, value) pairs of a rank-one factor, for both sparse and dense
# factor vectors (dense factors are used e.g. by the SOS rank-one batch).
_rank_one_nz(b::SparseArrays.SparseVector) = SparseArrays.findnz(b)
_rank_one_nz(b::AbstractVector) = (eachindex(b), b)

# `vec(s b b')` has `nnz(b)²` nonzeros at `support(b) × support(b)`.
function _nnz(F::_RankOneFactorization)
    bi, _ = _rank_one_nz(F.factor)
    return length(bi)^2
end

function _add_vec!(I, J, V, j, offset, F::_RankOneFactorization)
    s = only(F.scaling)
    d = length(F.factor)
    bi, bv = _rank_one_nz(F.factor)
    k = offset
    for q in eachindex(bi), p in eachindex(bi)
        k += 1
        I[k] = (bi[q] - 1) * d + bi[p]
        J[k] = j
        V[k] = s * bv[p] * bv[q]
    end
    return k
end
