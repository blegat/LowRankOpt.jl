# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestVariableToRankOne

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

function _test_psd(T::Type, W)
    MOI.Bridges.runtests(
        LRO.Bridges.Variable.ToRankOneBridge,
        model -> begin
            x, _ = MOI.add_constrained_variables(
                model,
                LRO.SetDotProducts{LRO.WITHOUT_SET}(
                    MOI.PositiveSemidefiniteConeTriangle(2),
                    LRO.TriangleVectorization.(
                        LRO.positive_semidefinite_factorization.([
                            T[
                                1 3
                                2 4
                            ],
                            T[5; 6;;],
                        ]),
                    ),
                ),
            )
            MOI.add_constraint(model, one(T) * x[1], MOI.EqualTo(zero(T)))
            MOI.add_constraint(model, one(T) * x[2], MOI.LessThan(zero(T)))
        end,
        model -> begin
            x, _ = MOI.add_constrained_variables(
                model,
                LRO.SetDotProducts{LRO.WITHOUT_SET}(
                    MOI.PositiveSemidefiniteConeTriangle(2),
                    LRO.TriangleVectorization.(
                        LRO.positive_semidefinite_factorization.([
                            T[1, 2],
                            T[3, 4],
                            T[5, 6],
                        ]),
                    ),
                ),
            )
            MOI.add_constraint(
                model,
                one(T) * x[1] + one(T) * x[2],
                MOI.EqualTo(zero(T)),
            )
            MOI.add_constraint(model, one(T) * x[3], MOI.LessThan(zero(T)))
        end;
        cannot_unbridge = true,
        eltype = T,
    )
    return
end

function test_psd(T::Type)
    _test_psd(T, LRO.WITH_SET)
    _test_psd(T, LRO.WITHOUT_SET)
    return
end

end  # module

TestVariableToRankOne.runtests()
