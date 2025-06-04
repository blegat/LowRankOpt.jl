"""
    errors(model::Model, x, y)

Return [the 6 standard DIMACS errors](https://plato.asu.edu/dimacs/node3.html).
"""
function errors(
    model::Model,
    x;
    y = nothing,
    primal_err = NLPModels.cons(model, x),
    dual_slack = nothing,
    dual_err = nothing,
    pobj = NLPModels.obj(model, x),
    dobj = dual_obj(model, y),
)
    b_den = 1 + LinearAlgebra.norm(cons_constant(model), 1)
    C_den =
        1 +
        LinearAlgebra.norm(NLPModels.grad(model, ScalarIndex), 1) +
        sum(matrix_indices(model), init = zero(b_den)) do i
            return LinearAlgebra.norm(NLPModels.grad(model, i), 1)
        end
    obj_den = 1 + abs(pobj) + abs(dobj)
    return (
        LinearAlgebra.norm(primal_err) / b_den,
        max(0, -LinearAlgebra.eigmin(x)) / b_den,
        isnothing(dual_err) ? zero(b_den) :
        LinearAlgebra.norm(dual_err) / C_den,
        isnothing(dual_slack) ? zero(b_den) :
        LinearAlgebra.norm(dual_err) / C_den,
        (pobj - dobj) / obj_den,
        LinearAlgebra.dot(x, dual_slack) / obj_den,
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
