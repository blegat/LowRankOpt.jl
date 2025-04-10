# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module Constraint

import MathOptInterface as MOI
import LowRankOpt as LRO

for filename in readdir(joinpath(@__DIR__, "bridges"); join = true)
    include(filename)
end

const ConversionBridge{W,T} = MOI.Bridges.Constraint.SetConversionBridge{
    T,
    LRO.LinearCombinationInSet{
        W,
        MOI.PositiveSemidefiniteConeTriangle,
        LRO.TriangleVectorization{T,LRO.Factorization{T,Matrix{T},Vector{T}}},
    },
}

function add_all_bridges(model, ::Type{T}) where {T}
    MOI.Bridges.add_bridge(model, ConversionBridge{LRO.WITHOUT_SET,T})
    MOI.Bridges.add_bridge(model, ConversionBridge{LRO.WITH_SET,T})
    MOI.Bridges.add_bridge(model, LinearCombinationBridge{T})
    MOI.Bridges.add_bridge(model, AppendZeroBridge{T})
    return
end

end
