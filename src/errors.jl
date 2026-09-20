"""
    errors(model::Model, x, y)

Return [the six DIMACS errors](https://plato.asu.edu/dimacs/node3.html),
temporarily using Loraine main's blockwise normalization for comparison
with its `nlpmodel` branch.
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
)
    b_den = 1 + LinearAlgebra.norm(cons_constant(model))
    # TODO: Reconsider global normalization after Loraine's `nlpmodel` branch
    # is merged. For now, match main's check_convergence so benchmark stopping
    # tolerances have the same meaning. In particular, the standard DIMACS
    # complementarity uses the total objective, not a separate block objective:
    # obj_den = 1 + abs(pobj) + abs(dobj)
    # err6 = LinearAlgebra.dot(x, dual_slack) / obj_den
    # The previous global residual/cone formulas were:
    # C_den = 1 + LinearAlgebra.norm(grad(model, ScalarIndex)) +
    #     sum(i -> LinearAlgebra.norm(grad(model, i)), matrix_indices(model))
    # err2 = max(0, -LinearAlgebra.eigmin(x)) / b_den
    # err3 = LinearAlgebra.norm(dual_err) / C_den
    # err4 = max(0, -LinearAlgebra.eigmin(dual_slack)) / C_den
    # Revisit the consistency of the numerator/denominator norms as well.
    inv_b_den = inv(b_den)
    err2 = err3 = err4 = err6 = zero(b_den)
    matrix_pobj = zero(pobj)
    for i in Iterators.flatten((matrix_indices(model), (ScalarIndex,)))
        X = x[i]
        isempty(X) && continue
        C = grad(model, i)
        inv_C_den = inv(1 + LinearAlgebra.norm(C))
        block_pobj = LinearAlgebra.dot(C, X)
        if i !== ScalarIndex
            matrix_pobj += block_pobj
        end
        min_x =
            i === ScalarIndex ? minimum(X) :
            LinearAlgebra.eigmin(LinearAlgebra.Symmetric(X))
        err2 += max(0, -min_x * inv_b_den)
        if !isnothing(dual_err)
            err3 += LinearAlgebra.norm(dual_err[i]) * inv_C_den
        end
        if !isnothing(dual_slack)
            S = dual_slack[i]
            min_s =
                i === ScalarIndex ? minimum(S) :
                LinearAlgebra.eigmin(LinearAlgebra.Symmetric(S))
            err4 += max(0, -min_s * inv_C_den)
            err6 += LinearAlgebra.dot(S, X) / (1 + abs(block_pobj) + abs(dobj))
        end
    end
    # Loraine main omits the scalar objective from this denominator, even
    # though it includes it in the numerator. Keep this convention temporarily.
    # TODO: Restore obj_den = 1 + abs(pobj) + abs(dobj) after the merge.
    obj_den = 1 + abs(matrix_pobj) + abs(dobj)
    return (
        LinearAlgebra.norm(primal_err) / b_den,
        err2,
        err3,
        err4,
        (pobj - dobj) / obj_den,
        err6,
    )
end

# As defined in https://plato.asu.edu/dimacs/node3.html
function LinearAlgebra.eigmin(x::AbstractSolution{T}) where {T}
    return min(
        minimum(x[ScalarIndex], init = zero(T)) +
        minimum(matrix_indices(x), init = zero(T)) do i
            return LinearAlgebra.eigmin(LinearAlgebra.Symmetric(x[i]))
        end,
    )
end
