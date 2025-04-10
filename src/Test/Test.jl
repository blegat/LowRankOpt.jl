# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module Test

using Test
import MathOptInterface as MOI
const MOIU = MOI.Utilities
import LowRankOpt as LRO

import FillArrays

const One{T} = FillArrays.Ones{T,0,Tuple{}}

"""
The goal is to find the maximum lower bound `γ` for the polynomial `x^2 - 2x`.
Using samples `-1` and `1`, the polynomial `x^2 - 2x - γ` evaluates at `-γ`
and `2 - γ` respectively.
The dot product with the gram matrix is the evaluation of `[1; x] * [1 x]` hence
`[1; -1] * [1 -1]` and `[1; 1] * [1 1]` respectively.

The polynomial version is:
max γ
s.t. [-γ, 2 - γ] in SetDotProducts(
    LRO.WITHOUT_SET,
    PSD(2),
    [[1; -1] * [1 -1], [1; 1] * [1 1]],
)
Its dual (moment version) is:
min -y[1] - y[2]
s.t. [-γ, 2 - γ] in LinearCombinationInSet(
    LRO.WITHOUT_SET,
    PSD(2),
    [[1; -1] * [1 -1], [1; 1] * [1 1]],
)
"""
function test_conic_PositiveSemidefinite_RankOne_polynomial(
    model::MOI.ModelLike,
    config::MOI.Test.Config{T},
) where {T}
    set = LRO.SetDotProducts{LRO.WITHOUT_SET}(
        MOI.PositiveSemidefiniteConeTriangle(2),
        LRO.TriangleVectorization.([
            LRO.positive_semidefinite_factorization(T[1, -1]),
            LRO.positive_semidefinite_factorization(T[1, 1]),
        ]),
    )
    MOI.Test.@requires MOI.supports_constraint(
        model,
        MOI.VectorAffineFunction{T},
        typeof(set),
    )
    MOI.Test.@requires MOI.supports_incremental_interface(model)
    MOI.Test.@requires MOI.supports(model, MOI.ObjectiveSense())
    MOI.Test.@requires MOI.supports(
        model,
        MOI.ObjectiveFunction{MOI.VariableIndex}(),
    )
    γ = MOI.add_variable(model)
    c = MOI.add_constraint(
        model,
        MOI.Utilities.operate(vcat, T, T(3) - T(1) * γ, T(-1) - T(1) * γ),
        set,
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), γ)
    if MOI.Test._supports(config, MOI.optimize!)
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        MOI.optimize!(model)
        @test MOI.get(model, MOI.TerminationStatus()) == config.optimal_status
        @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        if MOI.Test._supports(config, MOI.ConstraintDual)
            @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
        end
        @test ≈(MOI.get(model, MOI.ObjectiveValue()), T(-1), config)
        if MOI.Test._supports(config, MOI.DualObjectiveValue)
            @test ≈(MOI.get(model, MOI.DualObjectiveValue()), T(-1), config)
        end
        @test ≈(MOI.get(model, MOI.VariablePrimal(), γ), T(-1), config)
        @test ≈(MOI.get(model, MOI.ConstraintPrimal(), c), T[4, 0], config)
        if MOI.Test._supports(config, MOI.ConstraintDual)
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
    MOIU.set_mock_optimize!(
        model,
        (mock::MOIU.MockOptimizer) -> MOIU.mock_optimize!(
            mock,
            T[-1],
            (
                MOI.VectorAffineFunction{T},
                LRO.SetDotProducts{
                    LRO.WITHOUT_SET,
                    MOI.PositiveSemidefiniteConeTriangle,
                    LRO.TriangleVectorization{
                        T,
                        LRO.Factorization{T,Vector{T},One{T}},
                    },
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
    set = LRO.LinearCombinationInSet{LRO.WITHOUT_SET}(
        MOI.PositiveSemidefiniteConeTriangle(2),
        LRO.TriangleVectorization.([
            LRO.positive_semidefinite_factorization(T[1, -1]),
            LRO.positive_semidefinite_factorization(T[1, 1]),
        ]),
    )
    MOI.Test.@requires MOI.supports_add_constrained_variables(
        model,
        typeof(set),
    )
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
    if MOI.Test._supports(config, MOI.optimize!)
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        MOI.optimize!(model)
        @test MOI.get(model, MOI.TerminationStatus()) == config.optimal_status
        @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        if MOI.Test._supports(config, MOI.ConstraintDual)
            @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
        end
        @test ≈(MOI.get(model, MOI.ObjectiveValue()), T(-1), config)
        if MOI.Test._supports(config, MOI.DualObjectiveValue)
            @test ≈(MOI.get(model, MOI.DualObjectiveValue()), T(-1), config)
        end
        @test ≈(MOI.get(model, MOI.VariablePrimal(), y), T[0, 1], config)
        @test ≈(MOI.get(model, MOI.ConstraintPrimal(), c), T(1), config)
        if MOI.Test._supports(config, MOI.ConstraintDual)
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
    MOIU.set_mock_optimize!(
        model,
        (mock::MOIU.MockOptimizer) -> MOIU.mock_optimize!(
            mock,
            T[0, 1],
            (MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}) => [T(-1)],
            (
                MOI.VectorOfVariables,
                LRO.LinearCombinationInSet{
                    LRO.WITHOUT_SET,
                    MOI.PositiveSemidefiniteConeTriangle,
                    LRO.Factorization{T,Vector{T},One{T}},
                },
            ) => [T[4, 0]],
        ),
    )
    return
end

function runtests(
    model::MOI.ModelLike,
    config::MOI.Test.Config;
    include::Vector = String[],
    exclude::Vector = String[],
    warn_unsupported::Bool = false,
    verbose::Bool = false,
    exclude_tests_after::VersionNumber = v"999.0.0",
)
    tests = filter(names(@__MODULE__; all = true)) do name
        return startswith("$name", "test_")
    end
    tests = string.(tests)
    for ex in exclude
        if ex in tests && any(t -> ex != t && occursin(ex, t), tests)
            @warn(
                "The exclude string \"$ex\" is ambiguous because it exactly " *
                "matches a test, but it also partially matches another. Use " *
                "`r\"^$ex\$\"` to exclude the exactly matching test, or " *
                "`r\"$ex.*\"` to exclude all partially matching tests.",
            )
        end
    end
    for name_sym in names(@__MODULE__; all = true)
        name = string(name_sym)
        if !startswith(name, "test_")
            continue  # All test functions start with test_
        elseif !isempty(include) && !any(s -> occursin(s, name), include)
            continue
        elseif !isempty(exclude) && any(s -> occursin(s, name), exclude)
            continue
        end
        if verbose
            @info "Running $name"
        end
        test_function = getfield(@__MODULE__, name_sym)
        if MOI.Test.version_added(test_function) > exclude_tests_after
            if verbose
                println("  Skipping test because of `exclude_tests_after`")
            end
            continue
        end
        @testset "$(name)" begin
            c = copy(config)
            tear_down = MOI.Test.setup_test(test_function, model, c)
            # Make sure to empty the model before every test.
            MOI.empty!(model)
            try
                test_function(model, c)
            catch err
                if verbose
                    println("  Test errored with $(typeof(err))")
                end
                MOI.Test._error_handler(err, name, warn_unsupported)
            end
            if tear_down !== nothing
                tear_down()
            end
        end
    end
    return
end

end
