# This code computes the Schur complement using the ideas detailed in [FKN97, Section 3]
# This is useful to compute search direction in primal-dual interior-point methods for semidefinite programs [FKN97]
# [FKN97] Fujisawa, Katsuki, Masakazu Kojima, and Kazuhide Nakata. "Exploiting sparsity in primal-dual interior-point methods for semidefinite programming." Mathematical Programming 79 (1997): 235-253.
# [HKS24] Habibi, Soodeh, Michal Kočvara, and Michael Stingl. "Loraine -- an interior-point solver for low-rank semidefinite programming." Optimization Methods and Software 39.6 (2024): 1185-1215.
# It was adapted from dapted from Michal Kocvara's code in
# https://github.com/kocvara/Loraine.jl/blob/bd2821ba830786a78f04081d7e8f5cac25e56cac/src/makeBBBB.jl
#
# The Schur complement of a PSD block is `H[j, k] = ⟨Aⱼ, L Aₖ R⟩` for
# symmetric scaling matrices `L` and `R`. It is symmetric in `j` and `k`
# since `⟨Aⱼ, L Aₖ R⟩ = tr(Aⱼ L Aₖ R) = tr(Aₖ R Aⱼ L) = ⟨Aₖ, L Aⱼ R⟩`.
# The Nesterov-Todd (NT) direction uses `L = R = W` while the
# Helmberg-Kojima-Monteiro (HKM) direction uses `L = S⁻¹` and `R = X`.
# The methods taking a single `W` are the NT special case.

# Computes `tr(A L B R)` for symmetric sparse matrices `A` and `B`
function _dot(
    A::SparseArrays.SparseMatrixCSC,
    B::SparseArrays.SparseMatrixCSC,
    L::AbstractMatrix,
    R::AbstractMatrix,
)
    @assert LinearAlgebra.checksquare(L) ==
            LinearAlgebra.checksquare(R) ==
            LinearAlgebra.checksquare(A) ==
            LinearAlgebra.checksquare(B)
    # After these asserts, we know that `A`, `B`, `L` and `R` are square and
    # have the same sizes so we can safely use `@inbounds`
    result = zero(eltype(A))
    @inbounds for i in axes(A, 2)
        nzA = SparseArrays.nzrange(A, i)
        if !isempty(nzA)
            for j in axes(B, 2)
                nzB = SparseArrays.nzrange(B, j)
                if !isempty(nzB)
                    # `(A R)[i, j]`
                    AR = zero(result)
                    for k in nzA
                        AR +=
                            SparseArrays.nonzeros(A)[k] *
                            R[SparseArrays.rowvals(A)[k], j]
                    end
                    # `(L B)[i, j]`
                    LB = zero(result)
                    for k in nzB
                        LB +=
                            L[i, SparseArrays.rowvals(B)[k]] *
                            SparseArrays.nonzeros(B)[k]
                    end
                    result += AR * LB
                end
            end
        end
    end
    return result
end

# Computes `⟨A * W, W * B⟩` for symmetric sparse matrices `A` and `B`
_dot(A, B, W::AbstractMatrix) = _dot(A, B, W, W)

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

function add_schur_complement!(
    model::BufferedModelForSchur,
    L,
    R,
    ::Type{MatrixIndex},
    H,
)
    for i in matrix_indices(model)
        add_schur_complement!(model, i, L[i], R[i], H)
    end
    return H
end

# Adds one sparse component's contribution `𝐀ᵢᵀ (R ⊗ L) 𝐀ᵢ` to `H`.
# Its Jacobian cache follows the layout documented in `buffer_for_jprod`.
# `H` is assembled column by column. For a constraint `i` in the dense
# regime, that is, one of the `last_dense[ilmi]` first entries of the
# permutation `σ` sorting the constraints by decreasing number of nonzeros,
# the dense matrix `L Aᵢ R` is formed once and all the remaining inner
# products `⟨Aⱼ, L Aᵢ R⟩`, `j ∈ I`, are then obtained with the single
# sparse matrix-vector product `𝐀ᵢ[:, I]ᵀ vec(L Aᵢ R)` of
# `add_sub_jprod!`. The constraints with at most one nonzero use the
# dedicated low-nnz paths below, also following [FKN97].
# /!\ L and R need to be symmetric
function _add_sparse_component!(workspace, slot, L, R, AW, WAW, H)
    σ, last_dense = workspace.permutation, workspace.last_dense
    column = view(workspace.component.values, :, slot)
    n = length(column)

    for ii in axes(H, 1)
        i = σ[ii, slot]
        Ai = column[i]
        if _nnz(Ai) > 0
            if ii <= last_dense[slot]
                LinearAlgebra.mul!(AW, L, Ai)
                LinearAlgebra.mul!(WAW, AW, R)
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
                            ttt = _dot(Ai, Aj, L, R)
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
                                L[iiiiAi, iiijAj] *
                                R[jjjjAj, jjjiAi] *
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
    return _add_scalar_schur_complement!(model, k -> w[k], H)
end

function _add_scalar_schur_complement!(
    model::BufferedModelForSchur,
    weight::F,
    H,
) where {F<:Function}
    C = model.model.C_lin
    # C * Diagonal(w) * C'
    # = sum_k w[k] * c_k * c_k'.
    # = sum_k sum_j w[k] * c_k * (conj(c_{jk}) * e_j')
    # Sparse `axpy!` visits only stored entries and needs no intermediate matrix.
    rows, values = SparseArrays.rowvals(C), SparseArrays.nonzeros(C)
    for k in axes(C, 2)
        column = view(C, :, k)
        wk = weight(k)
        for p in SparseArrays.nzrange(C, k)
            j = Int(rows[p])
            LinearAlgebra.axpy!(wk * conj(values[p]), column, view(H, :, j))
        end
    end
    return H
end

function _add_sparse_eval_column!(result, values, slot, Ay, L, R)
    for k in eachindex(result)
        A = values[k, slot]
        if !iszero(_nnz(A))
            result[k] += _dot(A, Ay, L, R)
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
# where the scalar block `w = W[ScalarIndex]` is the weight itself.
function schur_complement!(model::BufferedModelForSchur, W::AbstractVector, H)
    return _schur_complement!(model, W, W, k -> W[ScalarIndex][k], H)
end

"""
    schur_complement!(model::BufferedModelForSchur, L, R, H)

Computes the Schur complement
`H[j, k] = ∑ᵢ ⟨Aᵢⱼ, Lᵢ Aᵢₖ Rᵢ⟩ + ∑ₗ C_lin[j, l] L[l] R[l] C_lin[k, l]`
for symmetric `Lᵢ` and `Rᵢ`, where `Lᵢ = L[MatrixIndex(i)]` and
`L[l]` is the `l`th entry of `L[ScalarIndex]`. The Nesterov-Todd direction
is obtained with `Lᵢ = Rᵢ = Wᵢ` and the Helmberg-Kojima-Monteiro direction
with `Lᵢ = Sᵢ⁻¹` and `Rᵢ = Xᵢ`.
"""
function schur_complement!(
    model::BufferedModelForSchur,
    L::AbstractVector,
    R::AbstractVector,
    H,
)
    weight = k -> L[ScalarIndex][k] * R[ScalarIndex][k]
    return _schur_complement!(model, L, R, weight, H)
end

function _schur_complement!(model, L, R, scalar_weight::F, H) where {F}
    fill!(H, zero(eltype(H)))
    if num_matrices(model) > 0
        add_schur_complement!(model, L, R, MatrixIndex, H)
    end
    if num_scalars(model) > 0
        _add_scalar_schur_complement!(model, scalar_weight, H)
    end
    return H
end

# Add `⟨Aᵢₖ, L A(y) R⟩` to `result[k]`, where `Ay` is `A(y) = ∑ⱼ Aᵢⱼ yⱼ` as
# returned by `unsafe_jtprod`.
#
# `A(y)` dense: form `L A(y) R` and contract it against `𝐀ᵢ` with one sparse
# matrix-vector product. Two `d×d` products, so `Θ(d³)` however sparse the
# `Aᵢₖ` are.
function _add_eval_schur_complement!(
    model::BufferedModelForSchur,
    L::AbstractMatrix,
    R::AbstractMatrix,
    Ay::AbstractMatrix,
    result,
    i::MatrixIndex,
)
    AW, WAW, _, _ = model.schur_buffer
    LinearAlgebra.mul!(AW[i.value], L, Ay)
    LinearAlgebra.mul!(WAW[i.value], AW[i.value], R)
    return add_jprod!(model, WAW[i.value], result, i)
end

# `A(y)` sparse: contract directly, never forming the `d×d` product, so the
# cost follows the nonzeros instead of `d`. This is the same trade-off
# `add_schur_complement!` makes per constraint matrix via `κ`/`last_dense`,
# here made once for `A(y)`.
function _add_eval_schur_complement!(
    model::BufferedModelForSchur,
    L::AbstractMatrix,
    R::AbstractMatrix,
    Ay::SparseArrays.SparseMatrixCSC,
    result,
    i::MatrixIndex,
)
    _foreach_column(model.model.A, i.value) do values, slot
        return _add_sparse_eval_column!(result, values, slot, Ay, L, R)
    end
    return result
end

# [HKS24, (5b)]
# Returns the matrix equal to the sum, for each equation, of
# ⟨A_i, WA(y)W⟩
function eval_schur_complement!(model::BufferedModelForSchur, W, y, result)
    weight = k -> W[ScalarIndex][k]
    return _eval_schur_complement!(model, W, W, weight, y, result)
end

"""
    eval_schur_complement!(model::BufferedModelForSchur, L, R, y, result)

Computes the product `H * y` of the Schur complement `H` computed by
[`schur_complement!(model, L, R, H)`](@ref) with `y` without forming `H`.
"""
function eval_schur_complement!(model::BufferedModelForSchur, L, R, y, result)
    weight = k -> L[ScalarIndex][k] * R[ScalarIndex][k]
    return _eval_schur_complement!(model, L, R, weight, y, result)
end

function _eval_schur_complement!(
    model,
    L,
    R,
    scalar_weight::F,
    y,
    result,
) where {F}
    fill!(result, zero(eltype(result)))
    components = model.schur_buffer[3]
    for i in matrix_indices(model)
        if _has_general_component(components, i.value)
            _add_eval_schur_complement!(
                model,
                L[i],
                R[i],
                unsafe_jtprod(model, y, i),
                result,
                i,
            )
        else
            _prepare_components!(components, i.value, L[i], R[i])
            _add_component_matvec!(components, i.value, L[i], R[i], y, result)
        end
    end
    C = model.model.C_lin
    for k in axes(C, 2)
        column = view(C, :, k)
        weight = scalar_weight(k) * LinearAlgebra.dot(column, y)
        LinearAlgebra.axpy!(weight, column, result)
    end
    return result
end

# Schur workspaces follow individual components, not the type of the whole
# model. A rank-one component can coexist with arbitrary other components.
_nnz(A::AbstractMatrix) = count(!iszero, A)

struct _GeneralSchurComponent{C,J}
    component::C
    jacobian::J
    permutation::Matrix{Int}
    last_dense::Vector{Int}
    sparse::Vector{Bool}
end

# Component `Aⱼ = wⱼ * U[j, :] * U[j, :]'`. The rows `U * L` and `U * R`
# are stored in `UL` and `UR` and `gram` and `gram_right` are `n × n`
# workspaces.
struct _RankOneSchurComponent{T,C}
    component::C
    U::Vector{Matrix{T}}
    weights::Matrix{T}
    UL::Vector{Matrix{T}}
    UR::Vector{Matrix{T}}
    gram::Matrix{T}
    gram_right::Matrix{T}
end

function _RankOneSchurComponent(
    component,
    U::Vector{Matrix{T}},
    weights::Matrix{T},
) where {T}
    n = size(weights, 1)
    return _RankOneSchurComponent(
        component,
        U,
        weights,
        similar.(U),
        similar.(U),
        zeros(T, n, n),
        zeros(T, n, n),
    )
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
    component::BlockColumns{<:_RankOneOrZeros{T}},
    model::Model{T},
    _,
) where {T}
    n, p = size(component.values)
    U = Matrix{T}[]
    weights = zeros(T, n, p)
    for (slot, i) in enumerate(component.block_ids)
        Ui, weights[:, slot] =
            _rank_one_factors(component.values, slot, model.msizes[i])
        push!(U, Ui)
    end
    return _RankOneSchurComponent(component, U, weights)
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

# `U * L` and `U * R`, the latter is not computed if `R === L`
_left_rows(a::_RankOneSchurComponent, slot, _) = a.UL[slot]
function _right_rows(a::_RankOneSchurComponent, slot, L, R)
    if R === L
        return _left_rows(a, slot, L)
    end
    return a.UR[slot]
end

_prepare_schur!(::_GeneralSchurComponent, _, _, _) = nothing
function _prepare_schur!(workspace::_RankOneSchurComponent, i, L, R)
    slot = _block_slot(workspace.component, i)
    if !iszero(slot)
        LinearAlgebra.mul!(workspace.UL[slot], workspace.U[slot], L)
        if R !== L
            LinearAlgebra.mul!(workspace.UR[slot], workspace.U[slot], R)
        end
    end
    return nothing
end

@inline _prepare_components!(::Tuple{}, i, L, R) = nothing
@inline function _prepare_components!(components::Tuple, i, L, R)
    _prepare_schur!(first(components), i, L, R)
    _prepare_components!(Base.tail(components), i, L, R)
    return nothing
end

# `WAW = L * A * R`
function _sandwich!(WAW, AW, L, R, A::AbstractMatrix, _)
    LinearAlgebra.mul!(AW, L, A)
    return LinearAlgebra.mul!(WAW, AW, R)
end
function _sandwich!(
    WAW,
    AW,
    L,
    R,
    A::AbstractFactorization{T,<:AbstractVector},
    work,
) where {T}
    buffered_mul!(AW, A, R, true, false, view(work, :, 1))
    return LinearAlgebra.mul!(WAW, L, AW)
end
function _sandwich!(
    WAW,
    AW,
    L,
    R,
    A::AbstractFactorization{T,<:AbstractMatrix},
    work,
) where {T}
    buffered_mul!(AW, A, R, true, false, view(work, :, 1:max_rank(A)))
    return LinearAlgebra.mul!(WAW, L, AW)
end

function _add_schur_pair!(
    a::_GeneralSchurComponent,
    b::_GeneralSchurComponent,
    i,
    L,
    R,
    AW,
    WAW,
    work,
    H,
    same,
)
    ia, ib = _block_slot(a.component, i), _block_slot(b.component, i)
    (iszero(ia) || iszero(ib)) && return nothing
    if same && a.sparse[ia]
        _add_sparse_component!(a, ia, L, R, AW, WAW, H)
        return nothing
    end
    ca, cb = view(a.component.values, :, ia), view(b.component.values, :, ib)
    for j in eachindex(ca)
        iszero(_nnz(ca[j])) && continue
        _sandwich!(WAW, AW, L, R, ca[j], work)
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

# For Aⱼ = aⱼ uⱼ uⱼ' and Bₖ = bₖ vₖ vₖ', the pair contraction is
# ⟨Aⱼ, L Bₖ R⟩ = aⱼ bₖ (uⱼ' L vₖ) (uⱼ' R vₖ). The `n × n` matrices of
# `uⱼ' L vₖ` and `uⱼ' R vₖ` are `U L V'` and `U R V'`. Distinct components
# contribute this matrix AND its transpose; omitting the transpose would
# lose cross terms.
function _add_schur_pair!(
    a::_RankOneSchurComponent,
    b::_RankOneSchurComponent,
    i,
    L,
    R,
    AW,
    WAW,
    work,
    H,
    same,
)
    ia, ib = _block_slot(a.component, i), _block_slot(b.component, i)
    (iszero(ia) || iszero(ib)) && return nothing
    left, right = _rank_one_grams(a, ia, b, ib, L, R)
    for k in axes(H, 2), j in axes(H, 1)
        value = a.weights[j, ia] * b.weights[k, ib] * left[j, k] * right[j, k]
        H[j, k] += value
        if !same
            H[k, j] += value
        end
    end
    return nothing
end

# `U L V'` and `U R V'` where `U` and `V` are the factors of `a` and `b`
function _rank_one_grams(a, ia, b, ib, L, R)
    left = LinearAlgebra.mul!(a.gram, _left_rows(a, ia, L), b.U[ib]')
    if R === L
        return left, left
    end
    right = LinearAlgebra.mul!(a.gram_right, _right_rows(a, ia, L, R), b.U[ib]')
    return left, right
end

# Computes `u' * A * v`
_bilinear_form(u, A::AbstractMatrix, v) = LinearAlgebra.dot(u, A, v)
function _bilinear_form(u, ::FillArrays.Zeros{T}, v) where {T}
    return zero(promote_type(eltype(u), T, eltype(v)))
end
function _bilinear_form(
    u,
    A::AbstractFactorization{T,<:AbstractVector},
    v,
) where {T}
    return only(A.scaling) *
           LinearAlgebra.dot(u, left_factor(A)) *
           LinearAlgebra.dot(right_factor(A), v)
end
function _bilinear_form(
    u,
    A::AbstractFactorization{T,<:AbstractMatrix},
    v,
) where {T}
    result = zero(T)
    for k in eachindex(A.scaling)
        result +=
            A.scaling[k] *
            LinearAlgebra.dot(u, view(left_factor(A), :, k)) *
            LinearAlgebra.dot(view(right_factor(A), :, k), v)
    end
    return result
end

# For Aⱼ = aⱼ uⱼ uⱼ', `⟨Bₖ, L Aⱼ R⟩ = aⱼ (R uⱼ)' Bₖ (L uⱼ)`.
function _add_schur_pair!(
    a::_RankOneSchurComponent,
    b::_GeneralSchurComponent,
    i,
    L,
    R,
    AW,
    WAW,
    work,
    H,
    same,
)
    ia, ib = _block_slot(a.component, i), _block_slot(b.component, i)
    (iszero(ia) || iszero(ib)) && return nothing
    left, right = _left_rows(a, ia, L), _right_rows(a, ia, L, R)
    column = view(b.component.values, :, ib)
    for j in axes(H, 1)
        u, v = view(right, j, :), view(left, j, :)
        for k in eachindex(column)
            value = a.weights[j, ia] * _bilinear_form(u, column[k], v)
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
    L,
    R,
    AW,
    WAW,
    work,
    H,
    same,
)
    return _add_schur_pair!(b, a, i, L, R, AW, WAW, work, H, same)
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
    return add_schur_complement!(model, i, W, W, H)
end

"""
    add_schur_complement!(
        model::BufferedModelForSchur,
        i::MatrixIndex,
        L::AbstractMatrix,
        R::AbstractMatrix,
        H,
    )

Adds `⟨Aᵢⱼ, L Aᵢₖ R⟩` to `H[j, k]` for symmetric matrices `L` and `R`.
"""
function add_schur_complement!(
    model::BufferedModelForSchur,
    i::MatrixIndex,
    L::AbstractMatrix,
    R::AbstractMatrix,
    H,
)
    AW, WAW, components, work = model.schur_buffer
    _prepare_components!(components, i.value, L, R)
    _add_component_pairs!(
        components,
        i.value,
        L,
        R,
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
    L,
    R,
    y,
    result,
    same,
)
    ia, ib = _block_slot(a.component, i), _block_slot(b.component, i)
    (iszero(ia) || iszero(ib)) && return nothing
    left, right = _rank_one_grams(a, ia, b, ib, L, R)
    for k in eachindex(y), j in eachindex(result)
        value = a.weights[j, ia] * b.weights[k, ib] * left[j, k] * right[j, k]
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
