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

# `SetDotProducts.vectors` is typed `Vs<:AbstractVector{V}`, so passing
# `eachrow(U)::RowSlices` (or any non-`Vector{V}` `AbstractVector{V}`)
# should round-trip cleanly through the bridges. Before the `Vs`-threading
# fix in `DotProductsBridge`/`AppendSetBridge`, the bridges' supertype
# `S1` parameter erased `Vs` to its `where`-bound, so MOI couldn't convert
# the concrete `ConstraintIndex{F, SetDotProducts{...Vector{V}}}` to the
# abstract `ConstraintIndex{F, SetDotProducts{...Vs} where Vs<:...}`.
function _rowslices_model(T, model)
    U = T[1 2; 4 5]                                          # 2 rows of length 2
    vectors_rs = eachrow(U)                                  # ::Base.RowSlices
    vectors = [
        LRO.TriangleVectorization(
            LRO.Factorization(view(U, j, :), reshape(T[one(T)], ())),
        )
        for j in eachindex(vectors_rs)
    ]                                                        # `Vector{V}`
    @assert vectors isa AbstractVector
    x, cx = MOI.add_constrained_variables(
        model,
        LRO.SetDotProducts{LRO.WITH_SET}(
            MOI.PositiveSemidefiniteConeTriangle(2),
            vectors,
        ),
    )
    MOI.add_constraint(model, one(T) * x[1], MOI.EqualTo(zero(T)))
    MOI.add_constraint(model, one(T) * x[2], MOI.LessThan(zero(T)))
    return cx
end
function test_added_constrained_variable_types_with_rowslices(T::Type)
    # The bug the explicit `added_constrained_variable_types` method pins
    # down: looking it up on the `Vs`-erased UnionAll `DotProductsBridge{T}`
    # used to throw `MethodError` because the default `SetMapBridge`
    # version couldn't extract `S1` through the 4-parameter UnionAll.
    @test MOI.Bridges.added_constrained_variable_types(
        LRO.Bridges.Variable.DotProductsBridge{T},
    ) isa Vector{Tuple{Type}}
    @test MOI.Bridges.added_constrained_variable_types(
        LRO.Bridges.Variable.AppendSetBridge{T},
    ) isa Vector{Tuple{Type}}
    # Concrete form should also work and report the concrete `S1` (the
    # `SetDotProducts` with both `V` and `Vs` filled in).
    V = LRO.TriangleVectorization{T,LRO.Factorization{T,Vector{T},Array{T,0}}}
    S = MOI.PositiveSemidefiniteConeTriangle
    S1 = LRO.SetDotProducts{LRO.WITH_SET,S,V,Vector{V}}
    @test (S1,) in MOI.Bridges.added_constrained_variable_types(
        LRO.Bridges.Variable.DotProductsBridge{T,S,V,Vector{V}},
    )
    # `AppendSetBridge` should report the matching `WITH_SET` form so
    # `add_bridge` knows to chain to `DotProductsBridge`.
    @test (S1,) in MOI.Bridges.added_constrained_variable_types(
        LRO.Bridges.Variable.AppendSetBridge{T,S,V,Vector{V}},
    )
    return
end

end  # module

TestVariableDotProducts.runtests()
