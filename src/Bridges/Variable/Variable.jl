module Variable

import MathOptInterface as MOI
import LowRankOpt as LRO

for filename in readdir(joinpath(@__DIR__, "bridges"); join = true)
    include(filename)
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
    MOI.Bridges.add_bridge(model, DotProductsBridge{T})
    MOI.Bridges.add_bridge(model, AppendSetBridge{T})
    return
end

end
