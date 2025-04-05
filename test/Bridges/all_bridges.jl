module TestAllBridges

using Test
import MathOptInterface as MOI
import LowRankOpt as LRO

abstract type TestModel{T} <: MOI.ModelLike end
MOI.is_empty(::TestModel) = true

# Like SDPLR
struct FactDotProdWithSetModel{T} <: TestModel{T} end

const FactDotProdWithSet{
    T,
    F<:AbstractMatrix{T},
    D<:AbstractVector{T},
    V<:AbstractVector{
        LRO.TriangleVectorization{T,LRO.Factorization{T,F,D}},
    },
} =
    LRO.SetDotProducts{
        LRO.WITH_SET,
        MOI.PositiveSemidefiniteConeTriangle,
        LRO.TriangleVectorization{T,LRO.Factorization{T,F,D}},
        V,
    }

MOI.supports_add_constrained_variables(::FactDotProdWithSetModel{T}, ::Type{<:FactDotProdWithSet{T}}) where {T} = true

function test_FactDotProdWithSet(T)
    model = MOI.instantiate(FactDotProdWithSetModel{T}, with_bridge_type = T)
    F = Matrix{T}
    D = Vector{T}
    V = Vector{LRO.TriangleVectorization{T,LRO.Factorization{T,F,D}}}
    @test MOI.supports_add_constrained_variables(
        model,
        FactDotProdWithSet{T,F,D,V},
    )
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
