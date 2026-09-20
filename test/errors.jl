module TestErrors

using Test, LinearAlgebra, SparseArrays
import LowRankOpt as LRO

@testset "Dual cone feasibility is independent of the equation residual" begin
    model = LRO.Model(
        [spdiagm([3.0, 0.0]), spzeros(1, 1)],
        reshape([spzeros(2, 2), spzeros(1, 1)], 2, 1),
        [0.0],
        sparsevec([1], [4.0], 1),
        spzeros(1, 1),
        [2, 1],
    )
    x = LRO.ShapedSolution([1.0], [Matrix{Float64}(I, 2, 2), ones(1, 1)])
    slack =
        LRO.ShapedSolution([1.0], [Matrix(Diagonal([-2.0, 3.0])), ones(1, 1)])
    residual = LRO.ShapedSolution([0.0], [zeros(2, 2), zeros(1, 1)])
    vectorized(s) = LRO.VectorizedSolution(
        [s.scalars; reduce(vcat, vec.(s.matrices))],
        model.dim,
    )
    for convert_solution in (identity, vectorized)
        xx, ss, rr = convert_solution.((x, slack, residual))
        # The cost normalization is 1 + 4 + 3 = 8; the smallest slack
        # eigenvalue is -2 even though the dual equation residual is zero.
        err = LRO.errors(model, xx; y = [0.0], dual_slack = ss, dual_err = rr)
        @test err[3] == 0
        @test err[4] == 1 / 4
        err = LRO.errors(model, xx; y = [0.0], dual_slack = ss)
        @test err[3] == 0
        @test err[4] == 1 / 4

        # A nonzero equation residual does not make a feasible slack infeasible.
        rr[LRO.ScalarIndex] .= 6
        err = LRO.errors(model, xx; y = [0.0], dual_slack = xx, dual_err = rr)
        @test err[3] == 3 / 4
        @test err[4] == 0
        # `identity` aliases the original residual; restore it for the next case.
        rr[LRO.ScalarIndex] .= 0
    end
end

end
