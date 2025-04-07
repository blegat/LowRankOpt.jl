# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestVariableAppendSet

using Test

import MathOptInterface as MOI
import LowRankOpt as LRO

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name) $T" for T in [Int, Float64]
                getfield(@__MODULE__, name)(T)
            end
        end
    end
    return
end

function test_psd(T::Type)
    MOI.Bridges.runtests(
        LRO.Bridges.Variable.AppendSetBridge,
        model -> begin
            x, _ = MOI.add_constrained_variables(
                model,
                LRO.SetDotProducts{LRO.WITHOUT_SET}(
                    MOI.PositiveSemidefiniteConeTriangle(2),
                    LRO.TriangleVectorization.([
                        T[
                            1 2
                            2 3
                        ],
                        T[
                            4 5
                            5 6
                        ],
                    ]),
                ),
            )
            MOI.add_constraint(model, one(T) * x[1], MOI.EqualTo(zero(T)))
            MOI.add_constraint(model, one(T) * x[2], MOI.LessThan(zero(T)))
        end,
        model -> begin
            x, _ = MOI.add_constrained_variables(
                model,
                LRO.SetDotProducts{LRO.WITH_SET}(
                    MOI.PositiveSemidefiniteConeTriangle(2),
                    LRO.TriangleVectorization.([
                        T[
                            1 2
                            2 3
                        ],
                        T[
                            4 5
                            5 6
                        ],
                    ]),
                ),
            )
            MOI.add_constraint(model, one(T) * x[1], MOI.EqualTo(zero(T)))
            MOI.add_constraint(model, one(T) * x[2], MOI.LessThan(zero(T)))
        end;
        cannot_unbridge = true,
        eltype = T,
    )
    return
end

end  # module

TestVariableAppendSet.runtests()
