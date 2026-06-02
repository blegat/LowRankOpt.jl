# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

struct DotProductsBridge{T,S,V,Vs} <: MOI.Bridges.Variable.SetMapBridge{
    T,
    S,
    LRO.SetDotProducts{LRO.WITH_SET,S,V,Vs},
}
    variables::Vector{MOI.VariableIndex}
    constraint::MOI.ConstraintIndex{MOI.VectorOfVariables,S}
    set::LRO.SetDotProducts{LRO.WITH_SET,S,V,Vs}
end

function MOI.Bridges.Variable.supports_constrained_variable(
    ::Type{<:DotProductsBridge},
    ::Type{<:LRO.SetDotProducts{LRO.WITH_SET}},
)
    return true
end

# Spell out `added_constrained_variable_types` — the default
# `SetMapBridge{T,S1}` version fails to extract `S1` from the 4-parameter
# `DotProductsBridge{T}` UnionAll once we thread `Vs` through. Without
# this, `MOI.Bridges.add_bridge` trips over `BridgeableConstraint`
# registration.
function MOI.Bridges.added_constrained_variable_types(
    ::Type{DotProductsBridge{T,S,V,Vs}},
) where {T,S,V,Vs}
    return Tuple{Type}[(LRO.SetDotProducts{LRO.WITH_SET,S,V,Vs},)]
end
function MOI.Bridges.added_constrained_variable_types(
    ::Type{<:DotProductsBridge},
)
    return Tuple{Type}[(LRO.SetDotProducts{LRO.WITH_SET},)]
end

function MOI.Bridges.Variable.concrete_bridge_type(
    ::Type{<:DotProductsBridge{T}},
    ::Type{<:LRO.SetDotProducts{LRO.WITH_SET,S,V,Vs}},
) where {T,S,V,Vs}
    return DotProductsBridge{T,S,V,Vs}
end

function MOI.Bridges.bridging_cost(::Type{<:DotProductsBridge})
    # We use a larger cost to make sure that `DotProducts` is used only if it is not possible
    # to keep the set `SetDotProducts`.
    # For instance, we prefer combining the bridges `ToRankOneBridge` and `ToPositiveBridge`
    # rather than using `DotProductsBridge`.
    return 10.0
end

function MOI.Bridges.Variable.bridge_constrained_variable(
    BT::Type{DotProductsBridge{T,S,V,Vs}},
    model::MOI.ModelLike,
    set::LRO.SetDotProducts{LRO.WITH_SET,S,V,Vs},
) where {T,S,V,Vs}
    variables, constraint = MOI.add_constrained_variables(
        model,
        MOI.Bridges.inverse_map_set(BT, set),
    )
    return BT(variables, constraint, set)
end

function MOI.Bridges.map_set(bridge::DotProductsBridge{T,S}, ::S) where {T,S}
    return bridge.set
end

function MOI.Bridges.inverse_map_set(
    ::Type{<:DotProductsBridge},
    set::LRO.SetDotProducts,
)
    return set.set
end

function MOI.Bridges.map_function(
    bridge::DotProductsBridge,
    func,
    i::MOI.Bridges.IndexInVector,
)
    scalars = MOI.Utilities.eachscalar(func)
    if i.value in eachindex(bridge.set.vectors)
        return MOI.Utilities.set_dot(
            bridge.set.vectors[i.value],
            scalars,
            bridge.set.set,
        )
    else
        return scalars[i.value-length(bridge.set.vectors)]
    end
end

function MOI.Bridges.map_function(bridge::DotProductsBridge, func)
    scalars = MOI.Utilities.eachscalar(func)
    return MOI.Utilities.vectorize(
        vcat(
            [
                MOI.Utilities.set_dot(vector, scalars, bridge.set.set) for
                vector in bridge.set.vectors
            ],
            scalars,
        ),
    )
end

# This returns `true` by default for `SetMapBridge` but setting
# `VariablePrimalStart` or `ConstraintPrimalStart` is not supported
# for this bridge because `inverse_map_function` is not implemented
function MOI.supports(
    ::MOI.ModelLike,
    ::MOI.VariablePrimalStart,
    ::Type{<:DotProductsBridge},
)
    return false
end

function MOI.supports(
    ::MOI.ModelLike,
    ::MOI.ConstraintPrimalStart,
    ::Type{<:DotProductsBridge},
)
    return false
end

# For setting `MOI.ConstraintDualStart`, we need to implement `adjoint_map_function`
# we leave it as future work
function MOI.supports(
    ::MOI.ModelLike,
    ::MOI.ConstraintDualStart,
    ::Type{<:DotProductsBridge},
)
    return false
end

function MOI.Bridges.Variable.unbridged_map(
    ::DotProductsBridge,
    ::Vector{MOI.VariableIndex},
)
    return nothing
end
