module TestNativeErrors

using Test, LinearAlgebra, SparseArrays
import LowRankOpt as LRO

@testset "Native conic residuals and multiplier signs" begin
    for T in (Float32, Float64)
        model = LRO.Model(
            [spdiagm(T[4, 5])],
            reshape([spdiagm(T[1, 2])], 1, 1),
            T[-2],
            sparse(T[3]),
            sparse(reshape(T[1], 1, 1)),
            [2];
            objective_constant = 7,
        )
        x = T[1]
        slack_flat = T[-1, -2, 0, 0, 4]
        multiplier_flat = T[1, 3, 0, 0, -2]
        flat(v::Vector) = v
        flat(v::LRO.ShapedConeVector) = vcat(v.scalars, vec.(v.matrices)...)
        shaped(v) = LRO.ShapedConeVector(v[1:1], [reshape(v[2:end], 2, 2)])
        work = [LRO.EigenWorkspace(zeros(T, 2, 2))]
        # c(x) = [2, 3, 0, 0, 3], so ||c(x)-s|| = sqrt(35).
        # grad f + J'λ = 2, f(x) = 9, and the dual objective is 12.
        # The worst primal/dual cone violations are 2 and 3, respectively.
        for representation in (identity, shaped),
            workspace in (nothing, work),
            p in (1, 2, Inf)

            slack, multipliers = representation.((slack_flat, multiplier_flat))
            primal_den = 1 + norm(T[3, 4, 0, 0, 5], p)
            expected = (
                sqrt(T(35)) / primal_den,
                T(2) / primal_den,
                T(2) / 3,
                one(T),
                -T(3) / 22,
                T(15) / 22,
            )
            actual = LRO.errors(model, x; slack, multipliers, workspace, p)
            @test collect(actual) ≈ collect(expected)
            precomputed = LRO.errors(
                model,
                x;
                slack,
                multipliers,
                workspace,
                p,
                primal_err = T[3, 5, 0, 0, -1],
                dual_err = T[2],
                pobj = T(9),
                dobj = T(12),
            )
            @test collect(precomputed) ≈ collect(actual)
            @test flat(slack) == T[-1, -2, 0, 0, 4]
            @test flat(multipliers) == T[1, 3, 0, 0, -2]
        end
        # Negative cone multipliers are feasible in the native convention.
        multipliers = T[-1, -2, 0, 0, -2]
        plain = LRO.errors(model, x; multipliers)
        buffered = LRO.errors(model, x; multipliers, workspace = work)
        @test collect(buffered) ≈ collect(plain)
        @test plain[1] == plain[2] == plain[4] == 0
        @test_throws ArgumentError LRO.errors(model, x; multipliers, p = 0.5)
    end
end

@testset "Empty native PSD blocks" begin
    model = LRO.Model(
        [zeros(0, 0)],
        reshape([zeros(0, 0)], 1, 1),
        [0.0],
        spzeros(0),
        spzeros(1, 0),
        [0],
    )
    @test LRO.errors(model, [1.0]; multipliers = Float64[]) ==
          ntuple(_ -> 0.0, 6)
end

end
