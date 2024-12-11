module Test

using Test
import MathOptInterface as MOI
const MOIU = MOI.Utilities

"""
The goal is to find the maximum lower bound `γ` for the polynomial `x^2 - 2x`.
Using samples `-1` and `1`, the polynomial `x^2 - 2x - γ` evaluates at `-γ`
and `2 - γ` respectively.
The dot product with the gram matrix is the evaluation of `[1; x] * [1 x]` hence
`[1; -1] * [1 -1]` and `[1; 1] * [1 1]` respectively.

The polynomial version is:
max γ
s.t. [-γ, 2 - γ] in SetDotProducts(
    PSD(2),
    [[1; -1] * [1 -1], [1; 1] * [1 1]],
)
Its dual (moment version) is:
min -y[1] - y[2]
s.t. [-γ, 2 - γ] in LinearCombinationInSet(
    PSD(2),
    [[1; -1] * [1 -1], [1; 1] * [1 1]],
)
"""
function test_conic_PositiveSemidefinite_RankOne_polynomial(
    model::MOI.ModelLike,
    config::MOI.Test.Config{T},
) where {T}
    set = LRO.SetDotProducts(
        MOI.PositiveSemidefiniteConeTriangle(2),
        MOI.TriangleVectorization.([
            MOI.PositiveSemidefiniteFactorization(T[1, -1]),
            MOI.PositiveSemidefiniteFactorization(T[1, 1]),
        ]),
    )
    MOI.Test.@requires MOI.supports_constraint(
        model,
        MOI.VectorAffineFunction{T},
        typeof(set),
    )
    MOI.Test.@requires MOI.supports_incremental_interface(model)
    MOI.Test.@requires MOI.supports(model, MOI.ObjectiveSense())
    MOI.Test.@requires MOI.supports(model, MOI.ObjectiveFunction{MOI.VariableIndex}())
    γ = MOI.add_variable(model)
    c = MOI.add_constraint(
        model,
        MOI.Utilities.operate(vcat, T, T(3) - T(1) * γ, T(-1) - T(1) * γ),
        set,
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), γ)
    if _supports(config, MOI.optimize!)
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        MOI.optimize!(model)
        @test MOI.get(model, MOI.TerminationStatus()) == config.optimal_status
        @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        if _supports(config, MOI.ConstraintDual)
            @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
        end
        @test ≈(MOI.get(model, MOI.ObjectiveValue()), T(-1), config)
        if _supports(config, MOI.DualObjectiveValue)
            @test ≈(MOI.get(model, MOI.DualObjectiveValue()), T(-1), config)
        end
        @test ≈(MOI.get(model, MOI.VariablePrimal(), γ), T(-1), config)
        @test ≈(MOI.get(model, MOI.ConstraintPrimal(), c), T[4, 0], config)
        if _supports(config, MOI.ConstraintDual)
            @test ≈(MOI.get(model, MOI.ConstraintDual(), c), T[0, 1], config)
        end
    end
    return
end

function MOI.Test.setup_test(
    ::typeof(test_conic_PositiveSemidefinite_RankOne_polynomial),
    model::MOIU.MockOptimizer,
    ::MOI.Test.Config{T},
) where {T<:Real}
    A = MOI.TriangleVectorization{
        T,
        MOI.PositiveSemidefiniteFactorization{T,Vector{T}},
    }
    MOIU.set_mock_optimize!(
        model,
        (mock::MOIU.MockOptimizer) -> MOIU.mock_optimize!(
            mock,
            T[-1],
            (
                MOI.VectorAffineFunction{T},
                LRO.SetDotProducts{
                    MOI.PositiveSemidefiniteConeTriangle,
                    A,
                    Vector{A},
                },
            ) => [T[0, 1]],
        ),
    )
    return
end

"""
The moment version of `test_conic_PositiveSemidefinite_RankOne_polynomial`

We look for a measure `μ = y1 * δ_{-1} + y2 * δ_{1}` where `δ_{c}` is the Dirac
measure centered at `c`. The objective is
`⟨μ, x^2 - 2x⟩ = y1 * ⟨δ_{-1}, x^2 - 2x⟩ + y2 * ⟨δ_{1}, x^2 - 2x⟩ = 3y1 - y2`.
We want `μ` to be a probability measure so `1 = ⟨μ, 1⟩ = y1 + y2`.
"""
function test_conic_PositiveSemidefinite_RankOne_moment(
    model::MOI.ModelLike,
    config::MOI.Test.Config{T},
) where {T}
    set = LRO.LinearCombinationInSet(
        MOI.PositiveSemidefiniteConeTriangle(2),
        MOI.TriangleVectorization.([
            MOI.PositiveSemidefiniteFactorization(T[1, -1]),
            MOI.PositiveSemidefiniteFactorization(T[1, 1]),
        ]),
    )
    MOI.Test.@requires MOI.supports_add_constrained_variables(model, typeof(set))
    MOI.Test.@requires MOI.supports_incremental_interface(model)
    MOI.Test.@requires MOI.supports(model, MOI.ObjectiveSense())
    MOI.Test.@requires MOI.supports(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
    )
    y, cy = MOI.add_constrained_variables(model, set)
    c = MOI.add_constraint(model, T(1) * y[1] + T(1) * y[2], MOI.EqualTo(T(1)))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
        T(3) * y[1] - T(1) * y[2],
    )
    if _supports(config, MOI.optimize!)
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        MOI.optimize!(model)
        @test MOI.get(model, MOI.TerminationStatus()) == config.optimal_status
        @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        if _supports(config, MOI.ConstraintDual)
            @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
        end
        @test ≈(MOI.get(model, MOI.ObjectiveValue()), T(-1), config)
        if _supports(config, MOI.DualObjectiveValue)
            @test ≈(MOI.get(model, MOI.DualObjectiveValue()), T(-1), config)
        end
        @test ≈(MOI.get(model, MOI.VariablePrimal(), y), T[0, 1], config)
        @test ≈(MOI.get(model, MOI.ConstraintPrimal(), c), T(1), config)
        if _supports(config, MOI.ConstraintDual)
            @test ≈(MOI.get(model, MOI.ConstraintDual(), cy), T[4, 0], config)
            @test ≈(MOI.get(model, MOI.ConstraintDual(), c), T(-1), config)
        end
    end
    return
end

function MOI.Test.setup_test(
    ::typeof(test_conic_PositiveSemidefinite_RankOne_moment),
    model::MOIU.MockOptimizer,
    ::MOI.Test.Config{T},
) where {T<:Real}
    A = MOI.TriangleVectorization{
        T,
        MOI.PositiveSemidefiniteFactorization{T,Vector{T}},
    }
    MOIU.set_mock_optimize!(
        model,
        (mock::MOIU.MockOptimizer) -> MOIU.mock_optimize!(
            mock,
            T[0, 1],
            (MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}) => [T(-1)],
            (
                MOI.VectorOfVariables,
                LRO.LinearCombinationInSet{
                    MOI.PositiveSemidefiniteConeTriangle,
                    A,
                    Vector{A},
                },
            ) => [T[4, 0]],
        ),
    )
    return
end

end
