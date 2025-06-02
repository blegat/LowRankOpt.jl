# Adapted from Loraine.jl

# Computes `⟨A * W, W * B⟩` for symmetric sparse matrices `A` and `B`
function _dot(A::SparseArrays.SparseMatrixCSC, B::SparseArrays.SparseMatrixCSC, W::AbstractMatrix)
    @assert LinearAlgebra.checksquare(W) == LinearAlgebra.checksquare(A) == LinearAlgebra.checksquare(B)
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
                        AW += SparseArrays.nonzeros(A)[k] * W[SparseArrays.rowvals(A)[k], j]
                    end
                    WB = zero(result)
                    for k in nzB
                        WB += W[i, SparseArrays.rowvals(B)[k]] * SparseArrays.nonzeros(B)[k]
                    end
                    result += AW * WB
                end
            end
        end
    end
    return result
end

function buffer_for_schur_complement(model::Model, κ)
    n = num_constraints(model)
    σ = zeros(Int64, n, num_matrices(model))
    last_dense = zeros(Int64, num_matrices(model))

    for mat_idx in matrix_indices(model)
        i = mat_idx.value
        nzA = [SparseArrays.nnz(model.A[i, j]) for j in 1:n]
        σ[:,i] = sortperm(nzA, rev = true)
        sorted = nzA[σ[:,i]]

        last_dense[i] = something(findlast(Base.Fix1(isless, κ), sorted), 0)
    end

    return σ, last_dense
end

function makeBBBB_rank1(n,nlmi,B,G)
    tmp = zeros(Float64, n, n)
    BBBB = zeros(Float64, n, n)
    for ilmi = 1:nlmi
        BB = transpose(B[ilmi] * G[ilmi])
        mul!(tmp,BB',BB)
        if ilmi == 1
            BBBB = tmp .^ 2
        else
            BBBB += tmp .^ 2
        end
    end
    return BBBB
end

#########################

function schur_complement(buffer, model::Model, W, ::Type{MatrixIndex})
    n = num_constraints(model)
    H = zeros(eltype(eltype(W)), n, n)
    for i in matrix_indices(model)
        H += schur_complement(buffer, model, i, W[i])
    end
    return H
end

#####
function schur_complement(buffer, model, mat_idx, W::AbstractMatrix{T}) where {T}
    σ, last_dense = buffer
    ilmi = mat_idx.value
    n = num_constraints(model)
    BBBB = zeros(T, n, n)
    dim = side_dimension(model, mat_idx)
    @assert dim == size(W, 1) == size(W, 2)
    tmp1 = Matrix{T}(undef, size(W, 2), dim)
    tmp2 = Vector{T}(undef, num_constraints(model))
    tmp  = zeros(T, size(W, 2), dim)

    for ii = 1:n
        i = σ[ii,ilmi]
        Ai = model.A[ilmi, i]
        if SparseArrays.nnz(Ai) > 0
            if ii <= last_dense[ilmi]
                LinearAlgebra.mul!(tmp1, W, Ai)
                LinearAlgebra.mul!(tmp, tmp1, W)
                fill!(tmp2, zero(T))
                add_jprod!(model, mat_idx, tmp, tmp2)
                indi = σ[ii:end,ilmi]
                BBBB[indi,i] .= tmp2[indi]
                BBBB[i,indi] .= tmp2[indi]
            else
                if !iszero(SparseArrays.nnz(Ai))
                    if SparseArrays.nnz(Ai) > 1
                        @inbounds for jj = ii:n
                            j = σ[jj,ilmi]
                            Aj = model.A[ilmi, j]
                            if !iszero(SparseArrays.nnz(Aj))
                                ttt = _dot(Ai, Aj, W)
                                if i >= j
                                    BBBB[i,j] = ttt
                                else
                                    BBBB[j,i] = ttt
                                end
                            end  
                        end   
                    else
                        # A is symmetric
                        iiiiAi = jjjiAi = only(SparseArrays.rowvals(Ai))
                        vvvi = only(SparseArrays.nonzeros(Ai))
                        @inbounds for jj = ii:n
                            j = σ[jj,ilmi]
                            Ajjj = model.A[ilmi, j]
                            # As we sort the matrices in decreasing `nnz` order,
                            # the rest of matrices is either zero or have only
                            # one entry
                            if !iszero(SparseArrays.nnz(Ajjj))
                                iiijAj = jjjjAj = only(SparseArrays.rowvals(Ajjj))
                                vvvj = only(SparseArrays.nonzeros(Ajjj))
                                ttt = vvvi * W[iiiiAi,iiijAj] * W[jjjiAi,jjjjAj] * vvvj
                                if i >= j
                                    BBBB[i,j] = ttt
                                else
                                    BBBB[j,i] = ttt
                                end
                            end
                        end 
                    end   
                end
            end
        end
    end
    return BBBB
end

# [HKS24, (5b)]
# Returns the matrix equal to the sum, for each equation, of
# ⟨A_i, WA_jW⟩
function schur_complement(buffer, model::Model, W::AbstractVector)
    H = MA.Zero()
    if num_matrices(model) > 0
        H = MA.add!!(H, schur_complement(buffer, model, W, MatrixIndex))
    end
    if num_scalars(model) > 0
        H = MA.add!!(H, schur_complement(model, W[ScalarIndex], ScalarIndex))
    end
    if H isa MA.Zero
        n = num_constraints(model)
        H = zeros(eltype(W), n, n)
    end
    return LinearAlgebra.Hermitian(H, :L)
end

function schur_complement(model::Model, w, ::Type{ScalarIndex})
    return model.C_lin * SparseArrays.spdiagm(w) * model.C_lin'
end

# [HKS24, (5b)]
# Returns the matrix equal to the sum, for each equation, of
# ⟨A_i, WA(y)W⟩
function eval_schur_complement!(buffer, result, model::Model, W, y)
    fill!(result, zero(eltype(result)))
    for i in matrix_indices(model)
        add_jprod!(model, i, -W[i] * jtprod!(buffer[i.value], model, i, y) * W[i], result)
    end
    result .+= model.C_lin * (W[ScalarIndex] .* (model.C_lin' * y))
    return result
end
