module TestAllBridges

using Test
import MathOptInterface as MOI
import LowRankOpt as LRO
import FillArrays

abstract type TestModel{T} <: MOI.ModelLike end
MOI.is_empty(::TestModel) = true

function set_type(W, T, N; primal::Bool, psd::Bool)
    F = Array{T,N}
    if psd
        if N == 2
            V = FillArrays.Ones{T,N - 1,Tuple{Base.OneTo{Int}}}
        else
            V = FillArrays.Ones{T,0,Tuple{}}
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
        @test MOI.Bridges.bridge_type(model, F, set_type(LRO.WITHOUT_SET, T, 2; primal = false, psd)) <: LRO.Bridges.Constraint.AppendZeroBridge{T}
        @test MOI.Bridges.bridge_type(model, F, set_type(LRO.WITH_SET, T, 1; primal = false, psd)) <: LRO.Bridges.Constraint.ConversionBridge
    end
    @test MOI.Bridges.bridge_type(model, F, set_type(LRO.WITHOUT_SET, T, 1; primal = false, psd = true)) <: LRO.Bridges.Constraint.ConversionBridge
    @test MOI.Bridges.bridge_type(model, F, set_type(LRO.WITHOUT_SET, T, 1; primal = false, psd = false)) <: LRO.Bridges.Constraint.AppendZeroBridge{T}
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
        @test MOI.Bridges.bridge_type(model, set_type(LRO.WITHOUT_SET, T, 2; primal = true, psd)) <: LRO.Bridges.Variable.AppendSetBridge{T}
        @test MOI.Bridges.bridge_type(model, set_type(LRO.WITH_SET, T, 1; primal = true, psd)) <: LRO.Bridges.Variable.ConversionBridge
    end
    @test MOI.Bridges.bridge_type(model, set_type(LRO.WITHOUT_SET, T, 1; primal = true, psd = true)) <: LRO.Bridges.Variable.ConversionBridge
    @test MOI.Bridges.bridge_type(model, set_type(LRO.WITHOUT_SET, T, 1; primal = true, psd = false)) <: LRO.Bridges.Variable.AppendSetBridge{T}
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
