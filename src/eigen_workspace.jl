"""
    EigenWorkspace(scratch::Matrix)

Reusable symmetric eigensolver storage. `scratch` is borrowed for nonmutating
minimum-eigenvalue calculations; eigensolver results alias the workspace and
are overwritten by the next call. A workspace must not be used concurrently.
"""
struct EigenWorkspace{T<:AbstractFloat}
    scratch::Matrix{T}
    lapack::Union{Nothing,FastLapackInterface.HermitianEigenWs{T,Matrix{T},T}}
end

function EigenWorkspace(scratch::Matrix{T}) where {T}
    return EigenWorkspace{T}(scratch, nothing)
end
function EigenWorkspace(scratch::Matrix{T}) where {T<:Union{Float32,Float64}}
    return EigenWorkspace{T}(
        scratch,
        FastLapackInterface.HermitianEigenWs(scratch; vecs = true),
    )
end

"""
    eigen!(workspace::EigenWorkspace, A; vectors=false, abstol=-1.0)

Compute symmetric eigenvalues, optionally with eigenvectors, overwriting `A`.
Returned arrays are borrowed until the next use of `workspace`.
"""
function eigen!(
    ws::EigenWorkspace{T},
    A;
    vectors = false,
    abstol = -1.0,
) where {T<:Union{Float32,Float64}}
    lapack = ws.lapack::FastLapackInterface.HermitianEigenWs{T,Matrix{T},T}
    LinearAlgebra.LAPACK.chkuplofinite(A, 'U')
    return LinearAlgebra.LAPACK.syevr!(
        lapack,
        vectors ? 'V' : 'N',
        'A',
        'U',
        A,
        0.0,
        0.0,
        0,
        0,
        abstol;
        resize = false,
    )
end
function eigen!(::EigenWorkspace, A; vectors = false, abstol = -1.0)
    if vectors
        F = LinearAlgebra.eigen!(LinearAlgebra.Symmetric(A))
        return F.values, F.vectors
    end
    return LinearAlgebra.eigvals!(LinearAlgebra.Symmetric(A)), nothing
end

function eigmin!(workspace::EigenWorkspace, A)
    copyto!(workspace.scratch, A)
    return minimum(first(eigen!(workspace, workspace.scratch)))
end
