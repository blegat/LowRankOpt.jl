# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

const _TriFact{T,F,D} = LRO.TriangleVectorization{T,LRO.Factorization{T,F,D}}

struct ToPositiveBridge{T,W,S,F,D} <: MOI.Bridges.Variable.SetMapBridge{
    T,
    LRO.SetDotProducts{W,S,_TriFact{T,F,LRO.One{T}}},
    LRO.SetDotProducts{W,S,_TriFact{T,F,D}},
}
    variables::Vector{MOI.VariableIndex}
    constraint::MOI.ConstraintIndex{
        MOI.VectorOfVariables,
        LRO.SetDotProducts{W,S,_TriFact{T,F,LRO.One{T}}},
    }
    scaling::Vector{T}
end

function MOI.Bridges.Variable.supports_constrained_variable(
    ::Type{<:ToPositiveBridge{T}},
    ::Type{<:LRO.SetDotProducts{W,S,_TriFact{T,F,D}}},
) where {T,W,S,F<:AbstractVector{T},D<:AbstractArray{T,0}}
    return D !== LRO.One{T}
end

function MOI.Bridges.Variable.concrete_bridge_type(
    ::Type{<:ToPositiveBridge{T}},
    ::Type{<:LRO.SetDotProducts{W,S,_TriFact{T,F,D}}},
) where {T,W,S,F<:AbstractVector{T},D<:AbstractArray{T,0}}
    return ToPositiveBridge{T,W,S,F,D}
end

_scaling(m::LRO.Factorization) = only(m.scaling)
_scaling(t::LRO.TriangleVectorization) = _scaling(t.matrix)

function MOI.Bridges.Variable.bridge_constrained_variable(
    BT::Type{ToPositiveBridge{T,W,S,F,D}},
    model::MOI.ModelLike,
    set::LRO.SetDotProducts{W,S,_TriFact{T,F,D}},
) where {T,W,S,F,D}
    variables, constraint = MOI.add_constrained_variables(
        model,
        MOI.Bridges.inverse_map_set(BT, set),
    )
    return BT(variables, constraint, _scaling.(set.vectors))
end

function _rescale(m::LRO.Factorization, D, scaling)
    return LRO.Factorization(m.factor, convert(D, reshape([scaling], tuple())))
end

function _rescale(t::LRO.TriangleVectorization, D, scaling)
    return LRO.TriangleVectorization(_rescale(t.matrix, D, scaling))
end

function MOI.Bridges.map_set(
    bridge::ToPositiveBridge{T,W,S,F,D},
    set::LRO.SetDotProducts{W},
) where {T,W,S,F,D}
    return LRO.SetDotProducts{W}(
        set.set,
        _rescale.(set.vectors, D, bridge.scaling),
    )
end

function _unscale(m::LRO.Factorization{T}) where {T}
    return LRO.Factorization(m.factor, LRO.One{T}(tuple()))
end

function _unscale(t::LRO.TriangleVectorization)
    return LRO.TriangleVectorization(_unscale(t.matrix))
end

function MOI.Bridges.inverse_map_set(
    ::Type{<:ToPositiveBridge},
    set::LRO.SetDotProducts{W},
) where {W}
    return LRO.SetDotProducts{W}(set.set, _unscale.(set.vectors))
end

function MOI.Bridges.map_function(
    bridge::ToPositiveBridge{T},
    func,
    i::MOI.Bridges.IndexInVector,
) where {T}
    scalar = MOI.Utilities.eachscalar(func)[i.value]
    if i.value in eachindex(bridge.scaling)
        return MOI.Utilities.operate(*, T, bridge.scaling[i.value], scalar)
    else
        return scalar
    end
end

# This returns `true` by default for `SetMapBridge` but setting
# `VariablePrimalStart` or `ConstraintPrimalStart` is not supported
# for this bridge because `inverse_map_function` is not implemented
function MOI.supports(
    ::MOI.ModelLike,
    ::MOI.VariablePrimalStart,
    ::Type{<:ToPositiveBridge},
)
    return false
end

function MOI.supports(
    ::MOI.ModelLike,
    ::MOI.ConstraintPrimalStart,
    ::Type{<:ToPositiveBridge},
)
    return false
end

# For setting `MOI.ConstraintDualStart`, we need to implement `adjoint_map_function`
# we leave it as future work
function MOI.supports(
    ::MOI.ModelLike,
    ::MOI.ConstraintDualStart,
    ::Type{<:ToPositiveBridge},
)
    return false
end

function MOI.Bridges.Variable.unbridged_map(
    ::ToPositiveBridge,
    ::Vector{MOI.VariableIndex},
)
    return nothing
end
