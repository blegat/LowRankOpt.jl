"""
    struct InnerAttribute{A<:MOI.AbstractConstraintAttribute} <: MOI.AbstractConstraintAttribute
        inner::A
    end

This attributes goes through all `Bridges.Variable.SetMapBridge` and `Bridges.Constraint.SetMapBridge`, ignoring the corresponding linear transformation.
"""
struct InnerAttribute{A<:MOI.AbstractConstraintAttribute} <:
       MOI.AbstractConstraintAttribute
    inner::A
end

MOI.is_copyable(a::InnerAttribute) = MOI.is_copyable(a.inner)
MOI.is_set_by_optimize(a::InnerAttribute) = MOI.is_set_by_optimize(a.inner)

function MOI.get_fallback(
    model::MOI.ModelLike,
    attr::InnerAttribute,
    ci::MOI.ConstraintIndex,
)
    return MOI.get(model, attr.inner, ci)
end
