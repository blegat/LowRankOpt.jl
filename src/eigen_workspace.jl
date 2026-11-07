"""
    EigenWorkspace(scratch::Matrix)

Reusable symmetric eigensolver storage. `scratch` is borrowed for nonmutating
minimum-eigenvalue calculations; eigensolver results alias the workspace and
are overwritten by the next call. A workspace must not be used concurrently.
"""
struct EigenWorkspace{T<:AbstractFloat}
    scratch::Matrix{T}
    lapack::Union{Nothing,FastLapackInterface.HermitianEigenWs{T,Matrix{T},T}}
    # Work arrays of `syevd`, resized by its first call
    syevd_work::Vector{T}
    syevd_iwork::Vector{LinearAlgebra.BlasInt}
end

function EigenWorkspace{T}(scratch::Matrix{T}, lapack) where {T}
    return EigenWorkspace{T}(scratch, lapack, T[], LinearAlgebra.BlasInt[])
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
