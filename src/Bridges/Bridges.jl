module Bridges

import MathOptInterface as MOI

include("Variable/Variable.jl")
include("Constraint/Constraint.jl")

function add_all_bridges(model, ::Type{T}) where {T}
    Variable.add_all_bridges(model, T)
    Constraint.add_all_bridges(model, T)
    return
end

end
