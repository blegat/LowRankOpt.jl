# This code computes the Schur complement using the ideas detailed in [FKN97, Section 3]
# This is useful to compute search direction in primal-dual interior-point methods for semidefinite programs [FKN97]
# [FKN97] Fujisawa, Katsuki, Masakazu Kojima, and Kazuhide Nakata. "Exploiting sparsity in primal-dual interior-point methods for semidefinite programming." Mathematical Programming 79 (1997): 235-253.
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
function buffer_for_schur_complement(model::Model{T}, κ) where {T}
    n = model.meta.ncon
    σ = zeros(Int64, n, num_matrices(model))
    last_dense = zeros(Int64, num_matrices(model))

    for mat_idx in matrix_indices(model)
        i = mat_idx.value
        nzA = [_nnz(model.A[i, j]) for j in 1:n]
        σ[:, i] = sortperm(nzA, rev = true)
        sorted = nzA[σ[:, i]]

        # Last index for which nnz > κ
        last_dense[i] = something(findlast(Base.Fix1(isless, κ), sorted), 0)
    end

    AW = [zeros(T, dim, dim) # /!\ it's the same zero everywhere, might be an issue with BigFloat
          for dim in model.msizes]
    WAW = copy.(AW)

    return AW, WAW, σ, last_dense
end

function add_schur_complement!(model::BufferedModelForSchur, W, ::Type{MatrixIndex}, H)
    for i in matrix_indices(model)
        add_schur_complement!(model, i, W[i], H)
    end
    return H
end

# /!\ W needs to be symmetric
function add_schur_complement!(
    model::BufferedModelForSchur,
    mat_idx::MatrixIndex,
    W::AbstractMatrix{T},
    H,
) where {T}
    AW, WAW, σ, last_dense = model.schur_buffer
    ilmi = mat_idx.value
    n = model.meta.ncon

    for ii in axes(H, 1)
        i = σ[ii, ilmi]
        Ai = model.model.A[ilmi, i]
        if _nnz(Ai) > 0
            if ii <= last_dense[ilmi]
                LinearAlgebra.mul!(AW[ilmi], W, Ai)
                LinearAlgebra.mul!(WAW[ilmi], AW[ilmi], W)
                I = view(σ, ii:n, ilmi)
                add_sub_jprod!(model, mat_idx, WAW[ilmi], view(H, I, i), I)
                for jj in (ii+1):n
                    j = σ[jj, ilmi]
                    H[i, j] = H[j, i]
                end
            else
                if _nnz(Ai) > 1
                    @inbounds for jj in ii:n
                        j = σ[jj, ilmi]
                        Aj = model.model.A[ilmi, j]
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
                        j = σ[jj, ilmi]
                        Ajjj = model.model.A[ilmi, j]
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

function add_schur_complement!(model::BufferedModelForSchur, w, ::Type{ScalarIndex}, H)
    H .+= model.model.C_lin * SparseArrays.spdiagm(w) * model.model.C_lin'
    return H
end

# [HKS24, (5b)]
# Returns the matrix equal to the sum, for each equation, of
# ⟨A_i, WA_jW⟩
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

# [HKS24, (5b)]
# Returns the matrix equal to the sum, for each equation, of
# ⟨A_i, WA(y)W⟩
function eval_schur_complement!(
    model::BufferedModelForSchur,
    W,
    y,
    result,
)
    fill!(result, zero(eltype(result)))
    for i in matrix_indices(model)
        add_jprod!(
            model,
            W[i] * jtprod!(model, y, i) * W[i],
            result,
            i,
        )
    end
    result .+= model.model.C_lin * (W[ScalarIndex] .* (model.model.C_lin' * y))
    return result
end
