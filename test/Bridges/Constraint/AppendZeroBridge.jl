# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestConstraintAppendZero

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
        LRO.Bridges.Constraint.AppendZeroBridge,
        model -> begin
            _, _ = MOI.add_constrained_variables(
                model,
                LRO.LinearCombinationInSet{LRO.WITHOUT_SET}(
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
        end,
        model -> begin
            x = MOI.add_variables(model, 2)
            MOI.add_constraint(
                model,
                MOI.Utilities.operate(
                    vcat,
                    T,
                    MOI.VectorOfVariables(x),
                    zeros(T, 3),
                ),
                LRO.LinearCombinationInSet{LRO.WITH_SET}(
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
        end;
        eltype = T,
        constraint_start = T[1, 2, 3],
    )
    return
end

end  # module

TestConstraintAppendZero.runtests()
