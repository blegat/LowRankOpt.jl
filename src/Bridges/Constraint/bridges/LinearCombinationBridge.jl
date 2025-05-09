# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

struct LinearCombinationBridge{T,S,V,F,G} <:
       MOI.Bridges.Constraint.SetMapBridge{
    T,
    S,
    LRO.LinearCombinationInSet{LRO.WITH_SET,S,V},
    F,
    G,
}
    constraint::MOI.ConstraintIndex{F,S}
    set::LRO.LinearCombinationInSet{LRO.WITH_SET,S,V}
end

function MOI.supports_constraint(
    ::Type{<:LinearCombinationBridge},
    ::Type{<:MOI.AbstractVectorFunction},
    ::Type{<:LRO.LinearCombinationInSet{LRO.WITH_SET}},
)
    return true
end

function MOI.Bridges.Constraint.concrete_bridge_type(
    ::Type{<:LinearCombinationBridge{T}},
    G::Type{<:MOI.AbstractVectorFunction},
    ::Type{LRO.LinearCombinationInSet{LRO.WITH_SET,S,V}},
) where {T,S,V}
    U = MOI.Utilities.promote_operation(*, T, MOI.Utilities.scalar_type(G), T)
    F = MOI.Utilities.promote_operation(vcat, T, U)
    return LinearCombinationBridge{T,S,V,F,G}
end

function _map_function(set::LRO.LinearCombinationInSet, func)
    scalars = MOI.Utilities.eachscalar(func)
    return MOI.Utilities.vectorize([
        sum(
            j -> scalars[j] * set.vectors[j][i],
            eachindex(set.vectors);
            init = scalars[length(set.vectors)+i],
        ) for i in 1:MOI.dimension(set.set)
    ])
end

function MOI.Bridges.Constraint.bridge_constraint(
    BT::Type{LinearCombinationBridge{T,S,V,F,G}},
    model::MOI.ModelLike,
    func::G,
    set::LRO.LinearCombinationInSet{LRO.WITH_SET,S,V},
) where {T,S,F,G,V}
    mapped_func = _map_function(set, func)
    constraint =
        MOI.add_constraint(model, mapped_func, MOI.Bridges.map_set(BT, set))
    return LinearCombinationBridge{T,S,V,F,G}(constraint, set)
end

function MOI.Bridges.map_set(
    ::Type{<:LinearCombinationBridge},
    set::LRO.LinearCombinationInSet,
)
    return set.set
end

function MOI.Bridges.inverse_map_set(
    bridge::LinearCombinationBridge,
    ::MOI.AbstractSet,
)
    return bridge.set
end

function MOI.Bridges.map_function(bridge::LinearCombinationBridge, func)
    return _map_function(bridge.set, func)
end

function MOI.Bridges.adjoint_map_function(bridge::LinearCombinationBridge, func)
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

function MOI.Bridges.inverse_map_function(::LinearCombinationBridge, _)
    throw(
        MOI.Bridges.MapNotInvertible(
            "The linear map is not always invertible for `LinearCombinationBridge`, use a `CachingOptimizer` layer",
        ),
    )
end

function MOI.Bridges.inverse_adjoint_map_function(
    bridge::LinearCombinationBridge,
    func,
)
    scalars = MOI.Utilities.eachscalar(func)
    return scalars[(length(bridge.set.vectors)+1):end]
end
