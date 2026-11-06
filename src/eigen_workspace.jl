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
    # Copies of `alpha` and `beta` overwritten by the eigenvalues of the
    # tridiagonal matrix and its eigenvectors, and the work of `stev`.
    ritz_values::Vector{T}
    subdiagonal::Vector{T}
    ritz_vectors::Matrix{T}
    stev_work::Vector{T}
    # Work arrays of `syevd`, resized by its first call
    syevd_work::Vector{T}
    syevd_iwork::Vector{LinearAlgebra.BlasInt}
end

# Number of iterations before restarting
const _LANCZOS_MAXITER = 30

# Number of cycles before falling back to `eigmin!`
const _LANCZOS_MAXCYCLES = 4

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
        Vector{T}(undef, k),
        Vector{T}(undef, k),
        Matrix{T}(undef, k, k),
        Vector{T}(undef, max(1, 2k - 2)),
        T[],
        LinearAlgebra.BlasInt[],
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
Returned arrays are borrowed until the next use of `workspace`. With
`vectors = true`, the eigenvectors are stored in `A` and `abstol` is ignored.
"""
function eigen!(
    ws::EigenWorkspace{T},
    A;
    vectors = false,
    abstol = -1.0,
) where {T<:Union{Float32,Float64}}
    lapack = ws.lapack::FastLapackInterface.HermitianEigenWs{T,Matrix{T},T}
    LinearAlgebra.LAPACK.chkuplofinite(A, 'U')
    if vectors
        # The divide-and-conquer algorithm of `syevd` is about twice as fast
        # as `syevr` when the eigenvalues are clustered, as for the scaling
        # of interior-point methods near the central path.
        return _syevd!(ws, lapack.w, A)
    end
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
# `LinearAlgebra.LAPACK.syevd!` allocates its outputs and work arrays.
# TODO: Replace by `FastLapackInterface.syevd!` once
# https://github.com/DynareJulia/FastLapackInterface.jl/pull/55 is released,
# as done in the branch `bl/fli_syevd`.
for (syevd, T) in ((:dsyevd_, :Float64), (:ssyevd_, :Float32))
    @eval begin
        function _syevd!(ws::EigenWorkspace{$T}, w, A)
            LinearAlgebra.BLAS.chkstride1(A)
            n = LinearAlgebra.checksquare(A)
            # Minimal sizes documented by LAPACK for the eigenvectors. They
            # are the sizes returned by its workspace query except for tiny
            # matrices.
            lwork = n <= 1 ? 1 : 1 + 6n + 2n^2
            liwork = n <= 1 ? 1 : 3 + 5n
            if length(ws.syevd_work) < lwork
                resize!(ws.syevd_work, lwork)
            end
            if length(ws.syevd_iwork) < liwork
                resize!(ws.syevd_iwork, liwork)
            end
            info = Ref{LinearAlgebra.BlasInt}()
            ccall(
                (
                    LinearAlgebra.BLAS.@blasfunc($syevd),
                    LinearAlgebra.BLAS.libblastrampoline,
                ),
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{LinearAlgebra.BlasInt},
                    Ptr{$T},
                    Ref{LinearAlgebra.BlasInt},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{LinearAlgebra.BlasInt},
                    Ptr{LinearAlgebra.BlasInt},
                    Ref{LinearAlgebra.BlasInt},
                    Ptr{LinearAlgebra.BlasInt},
                    Clong,
                    Clong,
                ),
                'V',
                'U',
                n,
                A,
                max(1, stride(A, 2)),
                w,
                ws.syevd_work,
                length(ws.syevd_work),
                ws.syevd_iwork,
                length(ws.syevd_iwork),
                info,
                1,
                1,
            )
            LinearAlgebra.LAPACK.chklapackerror(info[])
            return w, A
        end
    end
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
    lanczos_eigmin!(workspace::EigenWorkspace, A; rtol = 1e-3)

Return an estimate `θ - r` of the minimum eigenvalue of `A` where `θ` is the
smallest Ritz value of a Lanczos iteration with full
reorthogonalization and `r` is the norm of its residual. An eigenvalue of `A`
lies within `r` of `θ` and the minimum eigenvalue is at most `θ` so the
estimate is usually below the minimum eigenvalue. It is above when the Krylov
subspace has not captured its eigenvector yet. The iteration stops once
`r ≤ rtol * |θ|`. After $_LANCZOS_MAXITER iterations, it restarts from the
Ritz vector of `θ`. If it does not converge within $_LANCZOS_MAXCYCLES
cycles, the minimum eigenvalue is computed by `eigmin!` instead.

Unlike `eigmin!`, the cost does not include the `O(n³)` reduction to
tridiagonal form but only one product with `A` per iteration, which makes it
suitable for the step lengths of interior-point methods.

`A` is either a symmetric matrix, of which only the upper triangle is used, or
a symmetric linear operator of the size of the workspace supporting
`LinearAlgebra.mul!(w, A, v)`. Products with an operator avoid forming
matrices such as `L \\ Δ / L'` whose minimum eigenvalue gives the step length.
For an operator, the fallback forms its matrix with one product per column.
"""
function lanczos_eigmin!(
    ws::EigenWorkspace{T},
    A;
    # `brisk` uses `0.02` but it changes the iterates enough to make Loraine
    # need 3 more iterations on maxG11 or stop early with a less accurate
    # objective with `0.01`.
    rtol = T(1e-3),
) where {T<:Union{Float32,Float64}}
    # `KrylovKit.eigsolve` implements the Krylov-Schur method, that is, a thick
    # restart keeping several Ritz vectors, which converges in fewer products
    # than our restart from a single Ritz vector. However, it allocates its
    # Krylov basis at each call while `Krylov.jl`, which supports preallocated
    # workspaces, has no eigensolver. The step lengths only need a rough
    # estimate so a restart is rare and we use this simpler allocation-free
    # implementation.
    Q = ws.krylov
    n = size(Q, 1)
    k = length(ws.alpha)
    # A fixed start vector with no zero entry makes the result deterministic,
    # it is the one of the Lanczos iteration of `brisk`. For instance, the
    # vector of ones would be in an invariant subspace for the Lovász theta
    # number of a vertex-transitive graph.
    q = view(Q, :, 1)
    q .= 1 .+ sin.(T(1.7) .* (0:(n-1)) .+ T(0.3)) ./ 2
    LinearAlgebra.normalize!(q)
    for _ in 1:_LANCZOS_MAXCYCLES
        estimate = _lanczos_cycle!(ws, A, rtol)
        if !isnothing(estimate)
            return estimate
        end
        # Restart from the Ritz vector of the smallest Ritz value
        y = view(Q, :, k + 1)
        LinearAlgebra.mul!(y, view(Q, :, 1:k), view(ws.ritz_vectors, 1:k, 1))
        copyto!(q, y)
        LinearAlgebra.normalize!(q)
    end
    return _eigmin_fallback!(ws, A)
end
function lanczos_eigmin!(ws::EigenWorkspace, A; rtol = nothing)
    return _eigmin_fallback!(ws, A)
end

_eigmin_fallback!(ws::EigenWorkspace, A::AbstractMatrix) = eigmin!(ws, A)
function _eigmin_fallback!(ws::EigenWorkspace, A)
    e = view(ws.krylov, :, 1)
    for j in axes(ws.scratch, 2)
        fill!(e, false)
        e[j] = true
        LinearAlgebra.mul!(view(ws.scratch, :, j), A, e)
    end
    return minimum(first(eigen!(ws, ws.scratch)))
end

_symmetric_operator(A::AbstractMatrix) = LinearAlgebra.Symmetric(A, :U)
_symmetric_operator(A) = A

# Lanczos iteration from the first column of `ws.krylov`. Return the estimate
# `θ - r` if it converges and `nothing` otherwise.
function _lanczos_cycle!(ws::EigenWorkspace{T}, A, rtol) where {T}
    op = _symmetric_operator(A)
    Q = ws.krylov
    n = size(Q, 1)
    scale = zero(T)
    for j in eachindex(ws.alpha)
        w = view(Q, :, j + 1)
        LinearAlgebra.mul!(w, op, view(Q, :, j))
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
        _ritz!(ws, j)
        θ = ws.ritz_values[1]
        r = ws.beta[j] * abs(ws.ritz_vectors[j, 1])
        if (j >= min(n, _LANCZOS_MINITER) && r <= rtol * abs(θ)) ||
           r <= eps(T) * scale
            return θ - r
        end
        w ./= ws.beta[j]
    end
    return
end

# `LinearAlgebra.LAPACK.stev!` allocates the eigenvectors and the work
for (stev, T) in ((:dstev_, :Float64), (:sstev_, :Float32))
    @eval begin
        function _ritz!(ws::EigenWorkspace{$T}, j)
            copyto!(ws.ritz_values, 1, ws.alpha, 1, j)
            copyto!(ws.subdiagonal, 1, ws.beta, 1, j - 1)
            info = Ref{LinearAlgebra.BlasInt}()
            ccall(
                (
                    LinearAlgebra.BLAS.@blasfunc($stev),
                    LinearAlgebra.BLAS.libblastrampoline,
                ),
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{LinearAlgebra.BlasInt},
                    Ptr{$T},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{LinearAlgebra.BlasInt},
                    Ptr{$T},
                    Ptr{LinearAlgebra.BlasInt},
                    Clong,
                ),
                'V',
                j,
                ws.ritz_values,
                ws.subdiagonal,
                ws.ritz_vectors,
                size(ws.ritz_vectors, 1),
                ws.stev_work,
                info,
                1,
            )
            LinearAlgebra.LAPACK.chklapackerror(info[])
            return
        end
    end
end
