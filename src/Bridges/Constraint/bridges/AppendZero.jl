struct AppendZeroBridge{T,S,V,F,G} <: MOI.Bridges.Constraint.SetMapBridge{
    T,
    LRO.LinearCombinationInSet{LRO.WITH_SET,S,V},
    LRO.LinearCombinationInSet{LRO.WITHOUT_SET,S,V},
    F,
    G,
}
    constraint::MOI.ConstraintIndex{
        F,
        LRO.LinearCombinationInSet{LRO.WITH_SET,S,V},
    }
    set_dimension::Int
end

function MOI.supports_constraint(
    ::Type{<:AppendZeroBridge},
    ::Type{<:MOI.AbstractVectorFunction},
    ::Type{<:LRO.LinearCombinationInSet{LRO.WITHOUT_SET}},
)
    return true
end

function MOI.Bridges.Constraint.concrete_bridge_type(
    ::Type{<:AppendZeroBridge{T}},
    G::Type{<:MOI.AbstractVectorFunction},
    ::Type{LRO.LinearCombinationInSet{LRO.WITHOUT_SET,S,V}},
) where {T,S,V}
    F = MOI.Utilities.promote_operation(vcat, T, G, T)
    return AppendZeroBridge{T,S,V,F,G}
end

function _map_function(::Type{T}, func, set_dimension) where {T}
    return MOI.Utilities.operate(vcat, T, func, zeros(T, set_dimension))
end

function MOI.Bridges.Constraint.bridge_constraint(
    BT::Type{AppendZeroBridge{T,S,V,F,G}},
    model::MOI.ModelLike,
    func::G,
    set::LRO.LinearCombinationInSet{LRO.WITHOUT_SET,S,V},
) where {T,S,F,G,V}
    mapped_func = _map_function(T, func, MOI.dimension(set.set))
    constraint =
        MOI.add_constraint(model, mapped_func, MOI.Bridges.map_set(BT, set))
    return AppendZeroBridge{T,S,V,F,G}(constraint, MOI.dimension(set.set))
end

function MOI.Bridges.map_set(
    ::Type{<:AppendZeroBridge},
    set::LRO.LinearCombinationInSet{LRO.WITHOUT_SET},
)
    return LRO.LinearCombinationInSet{LRO.WITH_SET}(set.set, set.vectors)
end

function MOI.Bridges.inverse_map_set(
    ::Type{<:AppendZeroBridge},
    set::LRO.LinearCombinationInSet{LRO.WITH_SET},
)
    return LRO.LinearCombinationInSet{LRO.WITHOUT_SET}(set.set, set.vectors)
end

function MOI.Bridges.map_function(bridge::AppendZeroBridge{T}, func) where {T}
    return _map_function(T, func, bridge.set_dimension)
end

function MOI.Bridges.adjoint_map_function(bridge::AppendZeroBridge, func)
    scalars = MOI.Utilities.eachscalar(func)
    return scalars[1:(length(scalars)-bridge.set_dimension)]
end

function MOI.Bridges.inverse_map_function(bridge::AppendZeroBridge, func)
    return MOI.Bridges.adjoint_map_function(bridge, func)
end

# This is used for `MOI.ConstraintDualStart`. The adjoint
# is not really invertible so we should maybe throw `MOI.MapNotInvertible`.
# However here we just set zero as initial guess for the additional part
# it's still better than failing.
function MOI.Bridges.inverse_adjoint_map_function(
    bridge::AppendZeroBridge,
    func,
)
    return MOI.Bridges.map_function(bridge, func)
end
