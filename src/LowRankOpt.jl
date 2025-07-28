# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module LowRankOpt

import LinearAlgebra
import SparseArrays
import KrylovKit
import MathOptInterface as MOI

include("sets.jl")
include("factorization.jl")
include("attributes.jl")
include("distance_to_set.jl")
include("Test/Test.jl")
include("Bridges/Bridges.jl")

include("model.jl")
include("schur.jl")
include("MOI_wrapper.jl")
include("BurerMonteiro.jl")
include("errors.jl")

end # module LowRankOpt
