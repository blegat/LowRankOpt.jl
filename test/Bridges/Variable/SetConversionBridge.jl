module TestVariableSetConversion

using Test

import MathOptInterface as MOI
import LowRankOpt as LRO

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name) $T $W" for T in [Int], W in [LRO.WITHOUT_SET]
                #@testset "$(name) $T $W" for T in [Int, Float64], W in [LRO.WITHOUT_SET, LRO.WITH_SET]
                getfield(@__MODULE__, name)(T, W)
            end
        end
    end
    return
end

function test_psd(T::Type, W)
    MOI.Bridges.runtests(
        LRO.Bridges.Variable.ConversionBridge{W},
        model -> begin
            _, _ = MOI.add_constrained_variables(
                model,
                LRO.SetDotProducts{W}(
                    MOI.PositiveSemidefiniteConeTriangle(2),
                    LRO.TriangleVectorization.([
                        LRO.Factorization(T[1, 2], fill(T(-1), tuple())),
                    ]),
                ),
            )
        end,
        model -> begin
            _, _ = MOI.add_constrained_variables(
                model,
                LRO.SetDotProducts{W}(
                    MOI.PositiveSemidefiniteConeTriangle(2),
                    LRO.TriangleVectorization.([
                        LRO.Factorization(reshape(T[1, 2], 2, 1), T[-1]),
                    ]),
                ),
            )
        end;
        eltype = T,
    )
    return
end

end  # module

TestVariableSetConversion.runtests()
