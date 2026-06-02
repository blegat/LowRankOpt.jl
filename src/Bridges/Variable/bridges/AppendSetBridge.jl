# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

struct AppendSetBridge{T,S,V,Vs} <: MOI.Bridges.Variable.SetMapBridge{
    T,
    LRO.SetDotProducts{LRO.WITH_SET,S,V,Vs},
    LRO.SetDotProducts{LRO.WITHOUT_SET,S,V,Vs},
}
    variables::Vector{MOI.VariableIndex}
    constraint::MOI.ConstraintIndex{
        MOI.VectorOfVariables,
        LRO.SetDotProducts{LRO.WITH_SET,S,V,Vs},
    }
    num_vectors::Int
end

function MOI.Bridges.Variable.supports_constrained_variable(
    ::Type{<:AppendSetBridge},
    ::Type{<:LRO.SetDotProducts{LRO.WITHOUT_SET}},
)
    return true
end

# The default `added_constrained_variable_types(::Type{<:SetMapBridge{T,S1}})`
# fails to extract `S1` from a 4-parameter `AppendSetBridge{T}` UnionAll
# (Julia gets confused by the nested `where` chain through `SetDotProducts`).
# Spell it out so the bridge registers cleanly with `MOI.Bridges.add_bridge`.
function MOI.Bridges.added_constrained_variable_types(
    ::Type{AppendSetBridge{T,S,V,Vs}},
) where {T,S,V,Vs}
    return Tuple{Type}[(LRO.SetDotProducts{LRO.WITH_SET,S,V,Vs},)]
end
function MOI.Bridges.added_constrained_variable_types(
    ::Type{<:AppendSetBridge},
)
    return Tuple{Type}[(LRO.SetDotProducts{LRO.WITH_SET},)]
end

function MOI.Bridges.Variable.concrete_bridge_type(
    ::Type{<:AppendSetBridge{T}},
    ::Type{<:LRO.SetDotProducts{LRO.WITHOUT_SET,S,V,Vs}},
) where {T,S,V,Vs}
    return AppendSetBridge{T,S,V,Vs}
end

function MOI.Bridges.Variable.bridge_constrained_variable(
    BT::Type{AppendSetBridge{T,S,V,Vs}},
    model::MOI.ModelLike,
    set::LRO.SetDotProducts{LRO.WITHOUT_SET,S,V,Vs},
) where {T,S,V,Vs}
    variables, constraint = MOI.add_constrained_variables(
        model,
        MOI.Bridges.inverse_map_set(BT, set),
    )
    return BT(variables, constraint, length(set.vectors))
end

function MOI.Bridges.map_set(
    ::AppendSetBridge,
    set::LRO.SetDotProducts{LRO.WITH_SET},
)
    return LRO.SetDotProducts{LRO.WITHOUT_SET}(set.set, set.vectors)
end

function MOI.Bridges.inverse_map_set(
    ::Type{<:AppendSetBridge},
    set::LRO.SetDotProducts{LRO.WITHOUT_SET},
)
    return LRO.SetDotProducts{LRO.WITH_SET}(set.set, set.vectors)
end

function MOI.Bridges.map_function(
    ::AppendSetBridge,
    func,
    i::MOI.Bridges.IndexInVector,
)
    return MOI.Utilities.eachscalar(func)[i.value]
end

function MOI.Bridges.map_function(bridge::AppendSetBridge, func)
    return MOI.Utilities.eachscalar(func)[1:bridge.num_vectors]
end

# This returns `true` by default for `SetMapBridge` but setting
# `VariablePrimalStart` or `ConstraintPrimalStart` is not supported
# for this bridge because `inverse_map_function` is not implemented
function MOI.supports(
    ::MOI.ModelLike,
    ::MOI.VariablePrimalStart,
    ::Type{<:AppendSetBridge},
)
    return false
end

function MOI.supports(
    ::MOI.ModelLike,
    ::MOI.ConstraintPrimalStart,
    ::Type{<:AppendSetBridge},
)
    return false
end

# For setting `MOI.ConstraintDualStart`, we need to implement `adjoint_map_function`
# we leave it as future work
function MOI.supports(
    ::MOI.ModelLike,
    ::MOI.ConstraintDualStart,
    ::Type{<:AppendSetBridge},
)
    return false
end

function MOI.Bridges.Variable.unbridged_map(
    ::AppendSetBridge,
    ::Vector{MOI.VariableIndex},
)
    return nothing
end
