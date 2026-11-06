"""
    EigenWorkspace(scratch::Matrix)

Reusable symmetric eigensolver storage. `scratch` is borrowed for nonmutating
minimum-eigenvalue calculations; eigensolver results alias the workspace and
are overwritten by the next call. A workspace must not be used concurrently.
"""
struct EigenWorkspace{T<:AbstractFloat}
    scratch::Matrix{T}
    lapack::Union{Nothing,FastLapackInterface.HermitianEigenWs{T,Matrix{T},T}}
    # Orthonormal Lanczos basis, its last column is the next Lanczos vector
    krylov::Matrix{T}
    # Diagonal and subdiagonal of the Lanczos tridiagonal matrix
    alpha::Vector{T}
    beta::Vector{T}
    # Coefficients of the reorthogonalization in the Lanczos basis
    coefficients::Vector{T}
end

const _LANCZOS_MAXITER = 30

# Interior-point directions are often close to a multiple of the identity in
# the scaled space so every vector is close to an eigenvector. Stopping after
# a few iterations would miss the eigenvectors that deviate from it. We use
# the same number as `brisk`.
const _LANCZOS_MINITER = 6

function EigenWorkspace{T}(scratch::Matrix{T}, lapack) where {T}
    n = LinearAlgebra.checksquare(scratch)
    k = min(n, _LANCZOS_MAXITER)
    return EigenWorkspace{T}(
        scratch,
        lapack,
        Matrix{T}(undef, n, k + 1),
        Vector{T}(undef, k),
        Vector{T}(undef, k),
        Vector{T}(undef, k),
    )
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

"""
    lanczos_eigmin!(workspace::EigenWorkspace, A; rtol = 0.02)

Return an estimate `θ - r` of the minimum eigenvalue of the symmetric matrix
`A` where `θ` is the smallest Ritz value of a Lanczos iteration with full
reorthogonalization and `r` is the norm of its residual. An eigenvalue of `A`
lies within `r` of `θ` and the minimum eigenvalue is at most `θ` so the
estimate is usually below the minimum eigenvalue. It is above when the Krylov
subspace has not captured its eigenvector yet. The iteration stops once
`r ≤ rtol * |θ|`; if it does not within $_LANCZOS_MAXITER iterations,
the minimum eigenvalue is computed by `eigmin!` instead.

Unlike `eigmin!`, the cost does not include the `O(n³)` reduction to
tridiagonal form but only one product with `A` per iteration, which makes it
suitable for the step lengths of interior-point methods.
"""
function lanczos_eigmin!(
    ws::EigenWorkspace{T},
    A;
    rtol = T(0.02),
) where {T<:Union{Float32,Float64}}
    Q = ws.krylov
    n = size(Q, 1)
    # A fixed start vector with no zero entry makes the result deterministic,
    # it is the one of the Lanczos iteration of `brisk`.
    q = view(Q, :, 1)
    q .= 1 .+ sin.(T(1.7) .* (0:(n-1)) .+ T(0.3)) ./ 2
    LinearAlgebra.normalize!(q)
    scale = zero(T)
    for j in eachindex(ws.alpha)
        w = view(Q, :, j + 1)
        LinearAlgebra.mul!(w, LinearAlgebra.Symmetric(A, :U), view(Q, :, j))
        ws.alpha[j] = LinearAlgebra.dot(view(Q, :, j), w)
        scale = max(scale, abs(ws.alpha[j]))
        # Two passes of classical Gram-Schmidt are enough to make `w`
        # orthogonal to the basis up to rounding errors.
        basis = view(Q, :, 1:j)
        coefficients = view(ws.coefficients, 1:j)
        for _ in 1:2
            LinearAlgebra.mul!(coefficients, basis', w)
            LinearAlgebra.mul!(w, basis, coefficients, -1, 1)
        end
        ws.beta[j] = LinearAlgebra.norm(w)
        values, vectors =
            LinearAlgebra.LAPACK.stev!('V', ws.alpha[1:j], ws.beta[1:(j-1)])
        θ = values[1]
        r = ws.beta[j] * abs(vectors[j, 1])
        if (j >= min(n, _LANCZOS_MINITER) && r <= rtol * abs(θ)) ||
           r <= eps(T) * scale
            return θ - r
        end
        w ./= ws.beta[j]
    end
    return eigmin!(ws, A)
end
lanczos_eigmin!(ws::EigenWorkspace, A; rtol = nothing) = eigmin!(ws, A)
