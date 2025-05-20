# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

struct ToRankOneBridge{T,W,S,V1,V2} <: MOI.Bridges.Variable.SetMapBridge{
    T,
    LRO.SetDotProducts{W,S,V1},
    LRO.SetDotProducts{W,S,V2},
}
    variables::Vector{MOI.VariableIndex}
    constraint::MOI.ConstraintIndex{
        MOI.VectorOfVariables,
        LRO.SetDotProducts{W,S,V1},
    }
    ranges::Vector{UnitRange{Int}}
end

function MOI.Bridges.Variable.supports_constrained_variable(
    ::Type{<:ToRankOneBridge},
    ::Type{
        <:LRO.SetDotProducts{
            W,
            S,
            LRO.TriangleVectorization{T,LRO.Factorization{T,F,D}},
        },
    },
) where {T,W,S,F<:AbstractMatrix{T},D<:AbstractVector{T}}
    return true
end

lower_dimensional_type(::Type{Array{T,N}}) where {T,N} = Array{T,N-1}
import FillArrays
function lower_dimensional_type(::Type{FillArrays.Ones{T,1,I}}) where {T,I}
    return FillArrays.Ones{T,0,Tuple{}}
end

function MOI.Bridges.Variable.concrete_bridge_type(
    ::Type{<:ToRankOneBridge{T}},
    ::Type{<:LRO.SetDotProducts{W,S,V2}},
) where {
    T,
    W,
    S,
    F<:AbstractMatrix{T},
    D<:AbstractVector{T},
    V2<:LRO.TriangleVectorization{T,LRO.Factorization{T,F,D}},
}
    V1 = LRO.TriangleVectorization{
        T,
        LRO.Factorization{
            T,
            lower_dimensional_type(F),
            lower_dimensional_type(D),
        },
    }
    return ToRankOneBridge{T,W,S,V1,V2}
end

function MOI.Bridges.Variable.bridge_constrained_variable(
    BT::Type{ToRankOneBridge{T,W,S,V1,V2}},
    model::MOI.ModelLike,
    set::LRO.SetDotProducts{W,S},
) where {T,W,S,V1,V2}
    ranks = Int[size(v.matrix.factor, 2) for v in set.vectors]
    cs = cumsum(ranks)
    ranges = UnitRange.([1; (cs[1:(end-1)] .+ 1)], cs)
    variables, constraint = MOI.add_constrained_variables(
        model,
        MOI.Bridges.inverse_map_set(BT, set),
    )
    return BT(variables, constraint, ranges)
end

function _split_into_rank_ones(F::LRO.Factorization)
    return [
        LRO.Factorization(F.factor[:, i], F.scaling[reshape([i], tuple())]) for
        i in axes(F.factor, 2)
    ]
end

function _split_into_rank_ones(v::LRO.TriangleVectorization)
    return LRO.TriangleVectorization.(_split_into_rank_ones(v.matrix))
end

# It is not impemente in FillArrays, see
# https://github.com/JuliaArrays/FillArrays.jl/issues/23
function _reduce_vcat(v::Vector{FillArrays.Ones{T,0,Tuple{}}}) where {T}
    return FillArrays.Ones{T}(length(v))
end
# We also cannot rely on `Base` to return the right type:
# julia> reduce(vcat, [reshape([1], tuple())])
# 0-dimensional Array{Int64, 0}:
# 1
# which is a bit since it works for the next dimension
# julia> reduce(hcat, [reshape([1], 1)])
# 1×1 Matrix{Int64}:
# 1
_reduce_vcat(v::Vector{Array{T,0}}) where {T} = only.(v)

function _merge_rank_ones(Fs::Vector{<:LRO.Factorization})
    return LRO.Factorization(
        reduce(hcat, [F.factor for F in Fs]),
        _reduce_vcat([F.scaling for F in Fs]),
    )
end

function _merge_rank_ones(vectors::Vector{<:LRO.TriangleVectorization})
    matrices = [vector.matrix for vector in vectors]
    return LRO.TriangleVectorization(_merge_rank_ones(matrices))
end

function MOI.Bridges.map_set(
    bridge::ToRankOneBridge,
    set::LRO.SetDotProducts{W},
) where {W}
    vectors = _merge_rank_ones.(getindex.(Ref(set.vectors), bridge.ranges))
    return LRO.SetDotProducts{W}(set.set, vectors)
end

function MOI.Bridges.inverse_map_set(
    ::Type{<:ToRankOneBridge},
    set::LRO.SetDotProducts{W},
) where {W}
    vectors = reduce(vcat, _split_into_rank_ones.(set.vectors))
    return LRO.SetDotProducts{W}(set.set, vectors)
end

function _sum(::Type{T}, v::MOI.VectorOfVariables) where {T}
    return MOI.ScalarAffineFunction(
        MOI.ScalarAffineTerm.(one(T), v.variables),
        zero(T),
    )
end

function MOI.Bridges.map_function(
    bridge::ToRankOneBridge{T},
    func,
    i::MOI.Bridges.IndexInVector,
) where {T}
    scalars = MOI.Utilities.eachscalar(func)
    if i.value in eachindex(bridge.ranges)
        return _sum(T, scalars[bridge.ranges[i.value]])
    else
        return scalars[i.value-length(bridge.ranges)+last(last(bridge.ranges))]
    end
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
