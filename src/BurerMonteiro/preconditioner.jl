"""
    GramPreconditioner(model::Model; regularization = sqrt(eps(T)))

Inverse metric preconditioner for the Burer--Monteiro factors. After
`update!(M, x)`, `mul!(z, M, v)` applies `V / (U'U + regularization * I)`
to each matrix block, leaving scalar variables unchanged. `ldiv!` applies
the metric itself. The regularization must be finite and strictly positive.

For JSOSolvers' `trunk`, pass `M = M` and a callback
`(nlp, solver, stats) -> update!(M, solver.x)`. The constructor initializes
at `model.meta.x0`; call `update!(M, x0)` before solving if using another
initial point. This preconditions the inner
trust-region CG solve; it does not change the NLP derivatives or implement
Riemannian nonlinear CG. Recreate `M` after changing the model's ranks.
"""
struct GramPreconditioner{T,S}
    dim::Dimensions{S}
    regularization::T
    grams::Vector{Matrix{T}}
    factors::Vector{LinearAlgebra.Cholesky{T,Matrix{T}}}
end

function GramPreconditioner(
    model::Model{S,T};
    regularization = sqrt(eps(T)),
) where {S,T}
    δ = T(regularization)
    if !isfinite(δ) || δ <= zero(T)
        throw(ArgumentError("regularization must be finite and positive"))
    end
    dim = deepcopy(model.dim)
    grams = [zeros(T, r, r) for r in dim.ranks]
    factors = [
        LinearAlgebra.cholesky(Matrix{T}(LinearAlgebra.I, r, r)) for
        r in dim.ranks
    ]
    M = GramPreconditioner(dim, δ, grams, factors)
    return update!(M, model.meta.x0)
end

Base.size(M::GramPreconditioner) = (length(M.dim), length(M.dim))
Base.size(M::GramPreconditioner, i::Integer) = size(M)[i]
Base.eltype(::GramPreconditioner{T}) where {T} = T

function update!(M::GramPreconditioner, x::AbstractVector)
    if length(x) != length(M.dim)
        throw(DimensionMismatch("factor vector has the wrong length"))
    end
    X = Solution(x, M.dim)
    for i in eachindex(M.grams)
        U = X[LRO.MatrixIndex(i)].factor
        A = M.grams[i]
        LinearAlgebra.mul!(A, U', U)
        for j in axes(A, 1)
            A[j, j] += M.regularization
        end
        M.factors[i] = LinearAlgebra.cholesky(LinearAlgebra.Hermitian(A))
    end
    return M
end

function LinearAlgebra.mul!(
    z::AbstractVector,
    M::GramPreconditioner,
    v::AbstractVector,
)
    if length(z) != length(v) || length(v) != length(M.dim)
        throw(DimensionMismatch("preconditioner vectors have the wrong length"))
    end
    copyto!(z, v)
    Z = Solution(z, M.dim)
    for i in eachindex(M.factors)
        V = Z[LRO.MatrixIndex(i)].factor
        LinearAlgebra.rdiv!(V, M.factors[i])
    end
    return z
end

function LinearAlgebra.ldiv!(
    z::AbstractVector,
    M::GramPreconditioner,
    v::AbstractVector,
)
    if length(z) != length(v) || length(v) != length(M.dim)
        throw(DimensionMismatch("preconditioner vectors have the wrong length"))
    end
    # Copy the input so that applying the metric also supports z === v.
    V = Solution(copy(v), M.dim)
    Z = Solution(z, M.dim)
    copyto!(z, v)
    for i in eachindex(M.grams)
        LinearAlgebra.mul!(
            Z[LRO.MatrixIndex(i)].factor,
            V[LRO.MatrixIndex(i)].factor,
            M.grams[i],
        )
    end
    return z
end
