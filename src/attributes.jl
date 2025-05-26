# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

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

function MOI.is_copyable(a::InnerAttribute)
    return MOI.is_copyable(a.inner)
end

function MOI.is_set_by_optimize(a::InnerAttribute)
    return MOI.is_set_by_optimize(a.inner)
end

function MOI.get(
    model::MOI.Utilities.UniversalFallback,
    attr::InnerAttribute,
    ci::MOI.ConstraintIndex,
)
    return MOI.get(model, attr.inner, ci)
end

function MOI.get_fallback(
    model::MOI.ModelLike,
    attr::InnerAttribute,
    ci::MOI.ConstraintIndex,
)
    return MOI.get(model, attr.inner, ci)
end
