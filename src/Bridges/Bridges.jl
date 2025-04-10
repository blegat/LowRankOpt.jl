# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

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
