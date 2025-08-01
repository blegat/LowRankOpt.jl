# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module Variable

import SparseArrays
import MutableArithmetics as MA
import MathOptInterface as MOI
import LowRankOpt as LRO

for filename in readdir(joinpath(@__DIR__, "bridges"); join = true)
    include(filename)
end

function MOI.get(
    model::MOI.ModelLike,
    attr::LRO.InnerAttribute,
    bridge::MOI.Bridges.Variable.SetMapBridge,
)
    return MOI.get(model, attr, bridge.constraint)
end

const ConversionBridge{W,T} = SetConversionBridge{
    T,
    LRO.SetDotProducts{
        W,
        MOI.PositiveSemidefiniteConeTriangle,
        LRO.TriangleVectorization{T,LRO.Factorization{T,Matrix{T},Vector{T}}},
    },
}

function add_all_bridges(model, ::Type{T}) where {T}
    MOI.Bridges.add_bridge(model, ConversionBridge{LRO.WITHOUT_SET,T})
    MOI.Bridges.add_bridge(model, ConversionBridge{LRO.WITH_SET,T})
    MOI.Bridges.add_bridge(model, ToRankOneBridge{T})
    MOI.Bridges.add_bridge(model, ToPositiveBridge{T})
    MOI.Bridges.add_bridge(model, DotProductsBridge{T})
    MOI.Bridges.add_bridge(model, AppendSetBridge{T})
    return
end

end
