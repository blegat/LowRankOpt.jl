module Variable

import MathOptInterface as MOI

for filename in readdir(joinpath(@__DIR__, "bridges"); join = true)
    include(filename)
end

function add_all_bridges(model, ::Type{T}) where {T}
    MOI.Bridges.add_bridge(model, DotProductsBridge{T})
end

end
