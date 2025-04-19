# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestConstraintLinearCombination

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

function _model(T, model)
    _, cx = MOI.add_constrained_variables(
        model,
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
    return cx
end

function test_psd(T::Type)
    MOI.Bridges.runtests(
        LRO.Bridges.Constraint.LinearCombinationBridge,
        Base.Fix1(_model, T),
        model -> begin
            x = MOI.add_variables(model, 5)
            MOI.add_constraint(
                model,
                MOI.Utilities.vectorize([
                    one(T) * x[1] + T(4) * x[2] + x[3],
                    T(2) * x[1] + T(5) * x[2] + x[4],
                    T(3) * x[1] + T(6) * x[2] + x[5],
                ]),
                MOI.PositiveSemidefiniteConeTriangle(2),
            )
        end;
        cannot_unbridge = true,
        eltype = T,
        constraint_start = [18, 42, 1, 2, 3],
    )
    return
end

struct Custom <: MOI.AbstractConstraintAttribute end
MOI.is_set_by_optimize(::Custom) = false

function test_attribute(T::Type)
    inner = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{T}())
    model = MOI.Bridges._bridged_model(LRO.Bridges.Constraint.LinearCombinationBridge{T}, inner)
    cx = _model(T, model)
    F = MOI.VectorAffineFunction{T}
    S = MOI.PositiveSemidefiniteConeTriangle
    ci = only(MOI.get(inner, MOI.ListOfConstraintIndices{F,S}()))
    MOI.set(inner, Custom(), ci, "test")
    @test MOI.get(inner, Custom(), ci) == "test"
    @test MOI.get(inner, LRO.InnerAttribute(Custom()), ci) == "test"
    @test MOI.get(model, LRO.InnerAttribute(Custom()), cx) == "test"
end

end  # module

TestConstraintLinearCombination.runtests()
