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
        # The first PSD block uses 1 + norm(C[1]) = 4; its smallest
        # slack eigenvalue is -2 even though the equation residual is zero.
        err = LRO.errors(model, xx; y = [0.0], dual_slack = ss, dual_err = rr)
        @test err[3] == 0
        @test err[4] == 1 / 2
        err = LRO.errors(model, xx; y = [0.0], dual_slack = ss)
        @test err[3] == 0
        @test err[4] == 1 / 2

        # A nonzero equation residual does not make a feasible slack infeasible.
        rr[LRO.ScalarIndex] .= 6
        err = LRO.errors(model, xx; y = [0.0], dual_slack = xx, dual_err = rr)
        @test err[3] ≈ 6 / 5
        @test err[4] == 0
        # `identity` aliases the original residual; restore it for the next case.
        rr[LRO.ScalarIndex] .= 0
    end
end

@testset "Loraine main normalization" begin
    # Different block scales, negative eigenvalues in every block, and
    # cancelling objective contributions distinguish global from blockwise
    # normalization. Expected values are computed by hand from Loraine main.
    cases = (
        (true, true, [2, 6 / 5, 23 / 6, 45 / 13, -27 / 22, 163 / 36]),
        (true, false, [2, 3 / 5, 11 / 6, 44 / 39, -21 / 22, 79 / 36]),
        (false, true, [2, 3 / 5, 2, 7 / 3, -8 / 3, 7 / 3]),
    )
    for (psd, scalar, expected) in cases
        dims = psd ? [2, 1] : Int[]
        C = [
            spdiagm(v) for v in (psd ? [[3.0, 4.0], [12.0]] : Vector{Float64}[])
        ]
        A = reshape([spzeros(d, d) for d in dims], length(dims), 1)
        model = LRO.Model(
            C,
            A,
            [4.0],
            sparsevec(scalar ? [2.0] : Float64[]),
            spzeros(1, Int(scalar)),
            dims,
        )
        solution(v, blocks) = LRO.ShapedSolution(
            scalar ? [v] : Float64[],
            [Matrix(Diagonal(b)) for b in (psd ? blocks : Vector{Float64}[])],
        )
        x = solution(-3.0, [[-1.0, 2.0], [-2.0]])
        slack = solution(-7.0, [[-4.0, 5.0], [-6.0]])
        residual = solution(6.0, [[3.0, 4.0], [13.0]])
        vectorized(s) = LRO.VectorizedSolution(
            vcat(s.scalars, vec.(s.matrices)...),
            model.dim,
        )
        for convert_solution in (identity, vectorized)
            xx, ss, rr = convert_solution.((x, slack, residual))
            err = LRO.errors(
                model,
                xx;
                y = [0.5],
                primal_err = [10.0],
                dual_slack = ss,
                dual_err = rr,
            )
            for i in 1:6
                @test err[i] ≈ expected[i]
            end
        end
    end
end

end
