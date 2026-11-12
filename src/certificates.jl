"Maximum scalar-bound or cone violation at a primal point."
function constraint_violation(model::Model{T}, x) where {T}
    values = VectorizedConeVector(NLPModels.cons(model, x), model.dim)
    residual = zero(T)
    for i in 1:num_scalars(model)
        residual = max(
            residual,
            model.meta.lcon[i]-values[i],
            values[i]-model.meta.ucon[i],
        )
    end
    for idx in second_order_indices(model)
        vector = values[idx]
        residual = max(
            residual,
            LinearAlgebra.norm(view(vector, 2:length(vector)))-vector[1],
        )
    end
    for idx in matrix_indices(model)
        if isempty(values[idx])
            continue
        end
        residual = max(
            residual,
            -LinearAlgebra.eigmin(LinearAlgebra.Symmetric(values[idx])),
        )
    end
    return residual
end

"""
    infeasibility_certificate(model, multipliers; tol = 1e-8)

Check a conic Farkas ray. Return its positive normalization if the cone
multipliers have a zero Jacobian adjoint and strictly negative constant
pairing; otherwise return `nothing`. The objective is irrelevant to this
certificate. Interval multipliers use their signed lower/upper component.
"""
function infeasibility_certificate(
    model::Model{T},
    multipliers;
    tol = T(1e-8),
) where {T}
    if !all(isfinite, multipliers)
        return nothing
    end
    cone = VectorizedConeVector(collect(multipliers), model.dim)
    pairing = zero(T)
    for i in 1:num_scalars(model)
        lambda = multipliers[i]
        if iszero(lambda)
            continue
        end
        bound = lambda > 0 ? model.meta.lcon[i] : model.meta.ucon[i]
        if !isfinite(bound)
            return nothing
        end
        pairing += lambda*(model.constant[i]-bound)
    end
    constants = VectorizedConeVector(model.constant, model.dim)
    for idx in second_order_indices(model)
        pairing += LinearAlgebra.dot(constants[idx], cone[idx])
    end
    for idx in matrix_indices(model)
        pairing += LinearAlgebra.dot(constants[idx], cone[idx])
    end
    scale = -pairing
    if !isfinite(scale) || scale <= zero(T)
        return nothing
    end
    for idx in second_order_indices(model)
        vector = cone[idx]
        if (LinearAlgebra.norm(view(vector, 2:length(vector)))-vector[1])/scale >
           tol
            return nothing
        end
    end
    for idx in matrix_indices(model)
        if isempty(cone[idx])
            continue
        end
        if LinearAlgebra.norm(cone[idx]-cone[idx]', Inf)/scale > tol
            return nothing
        end
        if LinearAlgebra.eigmin(LinearAlgebra.Symmetric(cone[idx]))/scale < -tol
            return nothing
        end
    end
    adjoint = model.jacobian' * multipliers
    if LinearAlgebra.norm(adjoint, Inf)/scale > tol
        return nothing
    end
    return scale
end

"Check a homogeneous primal recession ray with strictly decreasing objective."
function recession_certificate(
    model::Model{T},
    direction;
    tol = T(1e-8),
) where {T}
    if !all(isfinite, direction)
        return nothing
    end
    scale = LinearAlgebra.dot(model.b, direction)
    if !isfinite(scale) || scale <= zero(T)
        return nothing
    end
    values = VectorizedConeVector((model.jacobian*direction)/scale, model.dim)
    for i in 1:num_scalars(model)
        if isfinite(model.meta.lcon[i]) && values[i] < -tol
            return nothing
        end
        if isfinite(model.meta.ucon[i]) && values[i] > tol
            return nothing
        end
    end
    for idx in second_order_indices(model)
        vector = values[idx]
        if LinearAlgebra.norm(view(vector, 2:length(vector)))-vector[1] > tol
            return nothing
        end
    end
    for idx in matrix_indices(model)
        if !isempty(values[idx]) &&
           LinearAlgebra.eigmin(LinearAlgebra.Symmetric(values[idx])) < -tol
            return nothing
        end
    end
    return scale
end
