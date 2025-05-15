# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

struct ToRankOneBridge{T,W,S,V} <: MOI.Bridges.Variable.SetMapBridge{
    T,
    LRO.SetDotProducts{W,S,V},
    LRO.SetDotProducts{W,S,V},
}
    variables::Vector{MOI.VariableIndex}
    constraint::MOI.ConstraintIndex{
        MOI.VectorOfVariables,
        LRO.SetDotProducts{W,S,V},
    }
    num_vectors::Int
end

function MOI.Bridges.Variable.supports_constrained_variable(
    ::Type{<:ToRankOneBridge},
    ::Type{<:LRO.SetDotProducts{
        W,
        S,
        LRO.TriangleVectorization{
            T,
            LRO.Factorization{
                T,
                F,
                D,
            },
        },
    }},
) where {T,W,S,F<:AbstractMatrix{T},D<:AbstractVector{T}}
    return true
end

function MOI.Bridges.Variable.concrete_bridge_type(
    ::Type{<:ToRankOneBridge{T}},
    ::Type{<:LRO.SetDotProducts{
        W,
        S,
        LRO.TriangleVectorization{
            T,
            LRO.Factorization{T,F,D},
        },
    }},
) where {T,W,S,F<:AbstractMatrix{T},D<:AbstractVector{T}}
    F1 = MA.promote_operation(getindex, F, Colon, Int)
    D1 = MA.promote_operation(getindex, D, Array{Int,0})
    V = LRO.TriangleVectorization{
        T,
        LRO.Factorization{T,F1,D1},
    }
    return ToRankOneBridge{T,W,S,V}
end

function MOI.Bridges.Variable.bridge_constrained_variable(
    BT::Type{ToRankOneBridge{T,W,S,V}},
    model::MOI.ModelLike,
    set::LRO.SetDotProducts{W,S,V},
) where {T,W,S,V}
    variables, constraint = MOI.add_constrained_variables(
        model,
        MOI.Bridges.inverse_map_set(BT, set),
    )
    return BT(variables, constraint, length(set.vectors))
end

function MOI.Bridges.map_set(
    ::ToRankOneBridge,
    set::LRO.SetDotProducts{LRO.WITH_SET},
)
    return LRO.SetDotProducts{LRO.WITHOUT_SET}(set.set, set.vectors)
end

function MOI.Bridges.inverse_map_set(
    ::Type{<:ToRankOneBridge},
    set::LRO.SetDotProducts{LRO.WITHOUT_SET},
)
    return LRO.SetDotProducts{LRO.WITH_SET}(set.set, set.vectors)
end

function MOI.Bridges.map_function(
    ::ToRankOneBridge,
    func,
    i::MOI.Bridges.IndexInVector,
)
    return MOI.Utilities.eachscalar(func)[i.value]
end

function MOI.Bridges.map_function(bridge::ToRankOneBridge, func)
    return MOI.Utilities.eachscalar(func)[1:bridge.num_vectors]
end

# This returns `true` by default for `SetMapBridge` but setting
# `VariablePrimalStart` or `ConstraintPrimalStart` is not supported
# for this bridge because `inverse_map_function` is not implemented
function MOI.supports(
    ::MOI.ModelLike,
    ::MOI.VariablePrimalStart,
    ::Type{<:ToRankOneBridge},
)
    return false
end

function MOI.supports(
    ::MOI.ModelLike,
    ::MOI.ConstraintPrimalStart,
    ::Type{<:ToRankOneBridge},
)
    return false
end

# For setting `MOI.ConstraintDualStart`, we need to implement `adjoint_map_function`
# we leave it as future work
function MOI.supports(
    ::MOI.ModelLike,
    ::MOI.ConstraintDualStart,
    ::Type{<:ToRankOneBridge},
)
    return false
end

function MOI.Bridges.Variable.unbridged_map(
    ::ToRankOneBridge,
    ::Vector{MOI.VariableIndex},
)
    return nothing
end
