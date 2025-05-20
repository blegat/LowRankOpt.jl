# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestVariableToPositive

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

function _test(T::Type, W)
    MOI.Bridges.runtests(
        LRO.Bridges.Variable.ToPositiveBridge,
        model -> begin
            x, _ = MOI.add_constrained_variables(
                model,
                LRO.SetDotProducts{W}(
                    MOI.PositiveSemidefiniteConeTriangle(2),
                    LRO.TriangleVectorization.([
                        LRO.Factorization(T[1, 2], T(3)),
                        LRO.Factorization(T[4, 5], T(6)),
                    ]),
                ),
            )
            MOI.add_constraint(model, one(T) * x[1], MOI.EqualTo(zero(T)))
            MOI.add_constraint(model, one(T) * x[2], MOI.LessThan(zero(T)))
            if W == LRO.WITH_SET
                MOI.add_constraint(model, one(T) * x[3], MOI.GreaterThan(zero(T)))
            end
        end,
        model -> begin
            x, _ = MOI.add_constrained_variables(
                model,
                LRO.SetDotProducts{W}(
                    MOI.PositiveSemidefiniteConeTriangle(2),
                    LRO.TriangleVectorization.(
                        LRO.positive_semidefinite_factorization.([
                            T[1, 2],
                            T[4, 5],
                        ]),
                    ),
                ),
            )
            MOI.add_constraint(model, T(3) * x[1], MOI.EqualTo(zero(T)))
            MOI.add_constraint(model, T(6) * x[2], MOI.LessThan(zero(T)))
            if W == LRO.WITH_SET
                MOI.add_constraint(model, one(T) * x[3], MOI.GreaterThan(zero(T)))
            end
        end;
        cannot_unbridge = true,
        eltype = T,
    )
    return
end

function test_runtests(T::Type)
    _test(T, LRO.WITH_SET)
    _test(T, LRO.WITHOUT_SET)
    return
end

end  # module

TestVariableToPositive.runtests()
