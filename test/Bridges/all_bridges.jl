# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestAllBridges

using Test
import MathOptInterface as MOI
import LowRankOpt as LRO

abstract type TestModel{T} <: MOI.ModelLike end
MOI.is_empty(::TestModel) = true

function set_type(W, T, N; primal::Bool, psd::Bool)
    F = Array{T,N}
    if psd
        if N == 2
            V = LRO.Ones{T}
        else
            V = LRO.One{T}
        end
    else
        V = Array{T,N - 1}
    end
    if primal
        return LRO.SetDotProducts{
            W,
            MOI.PositiveSemidefiniteConeTriangle,
            LRO.TriangleVectorization{T,LRO.Factorization{T,F,V}},
        }
    else
        return LRO.LinearCombinationInSet{
            W,
            MOI.PositiveSemidefiniteConeTriangle,
            LRO.TriangleVectorization{T,LRO.Factorization{T,F,V}},
        }
    end
end

struct LinearCombiInSetModel{T} <: TestModel{T} end

const _LinearCombiInSet{T,F<:AbstractMatrix{T},D<:AbstractVector{T}} =
    LRO.LinearCombinationInSet{
        LRO.WITH_SET,
        MOI.PositiveSemidefiniteConeTriangle,
        LRO.TriangleVectorization{T,LRO.Factorization{T,F,D}},
    }

function MOI.supports_constraint(
    ::LinearCombiInSetModel{T},
    ::Type{MOI.VectorAffineFunction{T}},
    ::Type{<:_LinearCombiInSet},
) where {T}
    return true
end

function test_LinearCombiInSetModel(T)
    model = MOI.instantiate(LinearCombiInSetModel{T}, with_bridge_type = T)
    LRO.Bridges.add_all_bridges(model, T)
    F = MOI.VectorAffineFunction{T}
    @testset "$W" for W in [LRO.WITH_SET, LRO.WITHOUT_SET]
        @testset "psd = $psd" for psd in [false, true]
            @testset "N = $N" for N in 1:2
                S = set_type(W, T, N; primal = false, psd)
                @test MOI.supports_constraint(model, F, S)
            end
        end
    end
    for psd in [false, true]
        @test MOI.Bridges.bridge_type(
            model,
            F,
            set_type(LRO.WITHOUT_SET, T, 2; primal = false, psd),
        ) <: LRO.Bridges.Constraint.AppendZeroBridge{T}
        @test MOI.Bridges.bridge_type(
            model,
            F,
            set_type(LRO.WITH_SET, T, 1; primal = false, psd),
        ) <: LRO.Bridges.Constraint.ConversionBridge
    end
    @test MOI.Bridges.bridge_type(
        model,
        F,
        set_type(LRO.WITHOUT_SET, T, 1; primal = false, psd = true),
    ) <: LRO.Bridges.Constraint.ConversionBridge
    @test MOI.Bridges.bridge_type(
        model,
        F,
        set_type(LRO.WITHOUT_SET, T, 1; primal = false, psd = false),
    ) <: LRO.Bridges.Constraint.AppendZeroBridge{T}
end

# Like SDPLR
struct FactDotProdWithSetModel{T} <: TestModel{T} end

const _SetDotProd{T,F<:AbstractMatrix{T},D<:AbstractVector{T}} =
    LRO.SetDotProducts{
        LRO.WITH_SET,
        MOI.PositiveSemidefiniteConeTriangle,
        LRO.TriangleVectorization{T,LRO.Factorization{T,F,D}},
    }

function MOI.supports_add_constrained_variables(
    ::FactDotProdWithSetModel{T},
    ::Type{<:_SetDotProd},
) where {T}
    return true
end

function test_FactDotProdWithSet(T)
    model = MOI.instantiate(FactDotProdWithSetModel{T}, with_bridge_type = T)
    LRO.Bridges.add_all_bridges(model, T)
    @testset "$W" for W in [LRO.WITH_SET, LRO.WITHOUT_SET]
        @testset "psd = $psd" for psd in [false, true]
            @testset "N = $N" for N in 1:2
                S = set_type(W, T, N; primal = true, psd)
                @test MOI.supports_add_constrained_variables(model, S)
            end
        end
    end
    for psd in [false, true]
        @test MOI.Bridges.bridge_type(
            model,
            set_type(LRO.WITHOUT_SET, T, 2; primal = true, psd),
        ) <: LRO.Bridges.Variable.AppendSetBridge{T}
        @test MOI.Bridges.bridge_type(
            model,
            set_type(LRO.WITH_SET, T, 1; primal = true, psd),
        ) <: LRO.Bridges.Variable.ConversionBridge
    end
    @test MOI.Bridges.bridge_type(
        model,
        set_type(LRO.WITHOUT_SET, T, 1; primal = true, psd = true),
    ) <: LRO.Bridges.Variable.ConversionBridge
    @test MOI.Bridges.bridge_type(
        model,
        set_type(LRO.WITHOUT_SET, T, 1; primal = true, psd = false),
    ) <: LRO.Bridges.Variable.AppendSetBridge{T}
end

# Like Hypatia
struct FactDotRankOnePSDModel{T,W} <: TestModel{T} end

const _SetDotProdRankOnePSD{T,W,F<:AbstractVector{T}} = LRO.SetDotProducts{
    W,
    MOI.PositiveSemidefiniteConeTriangle,
    LRO.TriangleVectorization{T,LRO.Factorization{T,F,LRO.One{T}}},
}

function MOI.supports_add_constrained_variables(
    ::FactDotRankOnePSDModel,
    ::Type{MOI.PositiveSemidefiniteConeTriangle},
)
    return true
end

function MOI.supports_add_constrained_variables(
    ::FactDotRankOnePSDModel{T,W},
    ::Type{<:_SetDotProdRankOnePSD{T,W}},
) where {T,W}
    return true
end

function _test_FactDotRankOnePSDModel(T, W)
    model = MOI.instantiate(FactDotRankOnePSDModel{T,W}, with_bridge_type = T)
    LRO.Bridges.add_all_bridges(model, T)
    S = set_type(W, T, 2; primal = true, psd = true)
    @test MOI.supports_add_constrained_variables(model, S)
    @test MOI.Bridges.bridge_type(model, S) <:
          LRO.Bridges.Variable.ToRankOneBridge{T,W}
    @test MOI.Bridges.bridging_cost(model, S) == 1
    S = set_type(W, T, 2; primal = true, psd = false)
    @test MOI.supports_add_constrained_variables(model, S)
    @test MOI.Bridges.bridge_type(model, S) <:
          LRO.Bridges.Variable.ToRankOneBridge{T,W}
    @test MOI.Bridges.bridging_cost(model, S) == 2
end

function test_FactDotRankOnePSDModel(T)
    _test_FactDotRankOnePSDModel(T, LRO.WITHOUT_SET)
    return _test_FactDotRankOnePSDModel(T, LRO.WITH_SET)
end

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            for T in [Int, Float64]
                @testset "$(name) $T" begin
                    getfield(@__MODULE__, name)(T)
                end
            end
        end
    end
    return
end

end

TestAllBridges.runtests()
