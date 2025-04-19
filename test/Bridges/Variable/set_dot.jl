# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestVariableDotProducts

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
    x, cx = MOI.add_constrained_variables(
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
    return cx
end

function test_psd(T::Type)
    MOI.Bridges.runtests(
        LRO.Bridges.Variable.DotProductsBridge,
        Base.Fix1(_model, T),
        model -> begin
            Q, _ = MOI.add_constrained_variables(
                model,
                MOI.PositiveSemidefiniteConeTriangle(2),
            )
            MOI.add_constraint(
                model,
                T(1) * Q[1] + T(4) * Q[2] + T(3) * Q[3],
                MOI.EqualTo(zero(T)),
            )
            MOI.add_constraint(
                model,
                T(4) * Q[1] + T(10) * Q[2] + T(6) * Q[3],
                MOI.LessThan(zero(T)),
            )
        end;
        cannot_unbridge = true,
        eltype = T,
    )
    return
end

struct Custom <: MOI.AbstractConstraintAttribute
    is_copyable::Bool
    is_set_by_optimize::Bool
end
MOI.is_copyable(c::Custom) = c.is_copyable
MOI.is_set_by_optimize(c::Custom) = c.is_set_by_optimize

function test_attribute(T::Type)
    inner = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{T}())
    model = MOI.Bridges._bridged_model(
        LRO.Bridges.Variable.DotProductsBridge{T},
        inner,
    )
    cx = _model(T, model)
    F = MOI.VectorOfVariables
    S = MOI.PositiveSemidefiniteConeTriangle
    ci = only(MOI.get(inner, MOI.ListOfConstraintIndices{F,S}()))
    attr = Custom(true, false)
    MOI.set(inner, attr, ci, "test")
    @test MOI.get(inner, attr, ci) == "test"
    attr = LRO.InnerAttribute(attr)
    @test MOI.get(inner, attr, ci) == "test"
    @test MOI.get(model, attr, cx) == "test"
    for is_copyable in [false, true]
        for is_set_by_optimize in [false, true]
            attr = LRO.InnerAttribute(Custom(is_copyable, is_set_by_optimize))
            @test MOI.is_copyable(attr) == is_copyable
            @test MOI.is_set_by_optimize(attr) == is_set_by_optimize
        end
    end
    @test_throws MOI.GetAttributeNotAllowed{Custom} MOI.get(
        inner.model,
        attr,
        ci,
    )
end

end  # module

TestVariableDotProducts.runtests()
