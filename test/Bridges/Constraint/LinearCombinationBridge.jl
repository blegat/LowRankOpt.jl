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

function test_psd(T::Type)
    MOI.Bridges.runtests(
        LRO.Bridges.Constraint.LinearCombinationBridge,
        model -> begin
            x, _ = MOI.add_constrained_variables(
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
        end,
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

end  # module

TestConstraintLinearCombination.runtests()
