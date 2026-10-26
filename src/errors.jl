@inline function block_errors(
    model,
    x,
    i,
    dual_err,
    dual_slack,
    workspace,
    errors,
)
    err2, err3, err4, err6, C_norm = errors
    X = x[i]
    isempty(X) && return (err2, err3, err4, err6, C_norm)
    C = grad(model, i)
    C_norm = max(C_norm, LinearAlgebra.norm(C, Inf))
    min_x = i === ScalarIndex ? minimum(X) : matrix_eigmin(X, i, workspace)
    err2 = max(err2, -min_x)
    if !isnothing(dual_err)
        err3 += LinearAlgebra.norm(dual_err[i])
    end
    if !isnothing(dual_slack)
        S = dual_slack[i]
        min_s = i === ScalarIndex ? minimum(S) : matrix_eigmin(S, i, workspace)
        err4 = max(err4, -min_s)
        err6 += LinearAlgebra.dot(S, X)
    end
    return (err2, err3, err4, err6, C_norm)
end

function matrix_eigmin(A, ::MatrixIndex, ::Nothing)
    return LinearAlgebra.eigmin(LinearAlgebra.Symmetric(A))
end
function matrix_eigmin(A, i::MatrixIndex, workspace::Vector{<:EigenWorkspace})
    return eigmin!(workspace[i.value], A)
end

"""
    errors(model::Model, x, y)

Return [the six DIMACS errors](https://plato.asu.edu/dimacs/node3.html).

Residual norms sum the PSD blocks' Frobenius norms and the scalar vector's
Euclidean norm. Cone violations use the worst eigenvalue or scalar across
all blocks. Data scales use the largest absolute entry of `b` and `C`.
The signed duality gap and complementarity share the denominator
`1 + abs(pobj) + abs(dobj)`, including the scalar objective.

`workspace` may be a vector of `EigenWorkspace`s, one per PSD block, to
reuse eigensolver storage. The primal and dual solutions are preserved.
"""
function errors(
    model::AbstractModel,
    x;
    y = nothing,
    primal_err = NLPModels.cons(model, x),
    dual_slack = nothing,
    dual_err = nothing,
    pobj = NLPModels.obj(model, x),
    dobj = dual_obj(model, y),
    workspace::Union{Nothing,Vector{<:EigenWorkspace}} = nothing,
)
    b_den = 1 + LinearAlgebra.norm(cons_constant(model), Inf)
    block_err =
        (zero(b_den), zero(b_den), zero(b_den), zero(b_den), zero(b_den))
    # Keep matrix and scalar indices in separate calls so each block's
    # arithmetic specializes on its solution and objective storage.
    for i in matrix_indices(model)
        block_err = block_errors(
            model,
            x,
            i,
            dual_err,
            dual_slack,
            workspace,
            block_err,
        )
    end
    err2, err3, err4, err6, C_norm = block_errors(
        model,
        x,
        ScalarIndex,
        dual_err,
        dual_slack,
        workspace,
        block_err,
    )
    C_den = 1 + C_norm
    obj_den = 1 + abs(pobj) + abs(dobj)
    return (
        LinearAlgebra.norm(primal_err) / b_den,
        err2 / b_den,
        err3 / C_den,
        err4 / C_den,
        (pobj - dobj) / obj_den,
        err6 / obj_den,
    )
end
