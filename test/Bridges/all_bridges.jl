module TestAllBridges

using Test
import MathOptInterface as MOI
import LowRankOpt as LRO
import FillArrays

abstract type TestModel{T} <: MOI.ModelLike end
MOI.is_empty(::TestModel) = true

# Like SDPLR
struct FactDotProdWithSetModel{T} <: TestModel{T} end

const FactDotProd{W,T,F<:AbstractMatrix{T},D<:AbstractVector{T}} =
    LRO.SetDotProducts{
        W,
        MOI.PositiveSemidefiniteConeTriangle,
        LRO.TriangleVectorization{T,LRO.Factorization{T,F,D}},
    }

const OneVec{T} = FillArrays.Ones{T,1,Tuple{Base.OneTo{Int}}}

const PosDefFactDotProd{W,T,F<:AbstractMatrix{T}} = LRO.SetDotProducts{
    W,
    MOI.PositiveSemidefiniteConeTriangle,
    LRO.TriangleVectorization{T,LRO.Factorization{T,F,OneVec{T}}},
}

function MOI.supports_add_constrained_variables(
    ::FactDotProdWithSetModel{T},
    ::Type{<:FactDotProd{LRO.WITH_SET,T}},
) where {T}
    return true
end

function test_FactDotProdWithSet(T)
    model = MOI.instantiate(FactDotProdWithSetModel{T}, with_bridge_type = T)
    LRO.Bridges.add_all_bridges(model, T)
    F = Matrix{T}
    D = Vector{T}
    @testset "W" for W in [LRO.WITH_SET, LRO.WITHOUT_SET]
        @test MOI.supports_add_constrained_variables(
            model,
            FactDotProd{W,T,F,D},
        )
        @test MOI.supports_add_constrained_variables(
            model,
            PosDefFactDotProd{W,T,F},
        )
    end
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
