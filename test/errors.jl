module TestErrors

using Test, LinearAlgebra, SparseArrays
import LowRankOpt as LRO

# Explicit residuals and objectives isolate the convergence check from setup.
function buffered_errors(model, x, slack, residual, work, primal_err; p = 2)
    return LRO.errors(
        model,
        x;
        primal_err,
        dual_slack = slack,
        dual_err = residual,
        pobj = one(eltype(primal_err)),
        dobj = 2 * one(eltype(primal_err)),
        workspace = work,
        p,
    )
end

function buffered_errors_allocations(
    model,
    x,
    slack,
    residual,
    work,
    primal_err,
    p = 2,
)
    buffered_errors(model, x, slack, residual, work, primal_err; p)
    return @allocated buffered_errors(
        model,
        x,
        slack,
        residual,
        work,
        primal_err;
        p,
    )
end

@testset "Dual cone feasibility is independent of the equation residual" begin
    model = LRO.Model(
        [spdiagm([3.0, 0.0]), spzeros(1, 1)],
        reshape([spzeros(2, 2), spzeros(1, 1)], 2, 1),
        [0.0],
        sparsevec(Int64[1], [4.0], 1),
        spzeros(Float64, Int64, 1, 1),
        Int64[2, 1],
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
        # The default global C scale is 1 + sqrt(3^2 + 4^2) = 6. The smallest
        # slack eigenvalue is -2 even though the equation residual is zero.
        err = LRO.errors(model, xx; y = [0.0], dual_slack = ss, dual_err = rr)
        @test err[3] == 0
        @test err[4] == 1 / 3
        err = LRO.errors(model, xx; y = [0.0], dual_slack = ss)
        @test err[3] == 0
        @test err[4] == 1 / 3

        # A nonzero equation residual does not make a feasible slack infeasible.
        rr[LRO.ScalarIndex] .= 6
        err = LRO.errors(model, xx; y = [0.0], dual_slack = xx, dual_err = rr)
        @test err[3] ≈ 1
        @test err[4] == 0
        # `identity` aliases the original residual; restore it for the next case.
        rr[LRO.ScalarIndex] .= 0
    end
end

@testset "Global DIMACS normalization" begin
    # Different block scales, negative eigenvalues in every block, and
    # cancelling objective contributions distinguish global from blockwise
    # normalization. Expected values follow the DIMACS definition by hand.
    cases = (
        (true, true, (3, sqrt(230), 7, -27 / 28, 47 / 28)),
        (true, false, (2, sqrt(194), 6, -21 / 22, 13 / 11)),
        (false, true, (3, 6, 7, -8 / 9, 7 / 3)),
    )
    for T in (Float32, Float64),
        (psd, scalar, numerators) in cases,
        p in (1, 2, Inf)

        C_entries = psd ? T[3, 4, 12] : T[]
        scalar && push!(C_entries, T(2))
        b_den = 1 + norm(T[3, 4], p)
        C_den = 1 + norm(C_entries, p)
        primal_cone, dual_residual, dual_cone, gap, complementarity = numerators
        expected = (
            10 / b_den,
            primal_cone / b_den,
            dual_residual / C_den,
            dual_cone / C_den,
            gap,
            complementarity,
        )
        dims = psd ? Int64[2, 1] : Int64[]
        C = SparseMatrixCSC{T,Int64}[
            spdiagm(T.(v)) for v in (psd ? [[3, 4], [12]] : Vector{Int}[])
        ]
        A = SparseMatrixCSC{T,Int64}[
            spzeros(T, Int64, d, d) for d in dims, _ in 1:2
        ]
        model = LRO.Model(
            C,
            A,
            T[3, 4],
            SparseVector{T,Int64}(sparsevec(scalar ? T[2] : T[])),
            spzeros(T, Int64, 2, Int(scalar)),
            dims,
        )
        solution(v, blocks) = LRO.ShapedSolution(
            scalar ? T[v] : T[],
            Matrix{T}[
                Matrix(Diagonal(T.(b))) for
                b in (psd ? blocks : Vector{Float64}[])
            ],
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
                y = T[0, 0.5],
                primal_err = T[6, 8],
                dual_slack = ss,
                dual_err = rr,
                p,
            )
            for i in 1:6
                @test err[i] ≈ expected[i]
            end
            work = LRO.EigenWorkspace{T}[
                LRO.EigenWorkspace(zeros(T, d, d)) for d in dims
            ]
            original_x = [copy(xx[i]) for i in LRO.matrix_indices(model)]
            original_s = [copy(ss[i]) for i in LRO.matrix_indices(model)]
            buffered = LRO.errors(
                model,
                xx;
                y = T[0, 0.5],
                primal_err = T[6, 8],
                dual_slack = ss,
                dual_err = rr,
                p,
                workspace = work,
            )
            @test all(isapprox.(buffered, err))
            if p == 2
                default = LRO.errors(
                    model,
                    xx;
                    y = T[0, 0.5],
                    primal_err = T[6, 8],
                    dual_slack = ss,
                    dual_err = rr,
                    workspace = work,
                )
                @test default == buffered
                @test_throws ArgumentError LRO.errors(
                    model,
                    xx;
                    y = T[0, 0.5],
                    p = 0,
                )
            end
            primal_err = T[6, 8]
            complete = buffered_errors(model, xx, ss, rr, work, primal_err; p)
            for residual in (rr, nothing), slack in (ss, nothing)
                result = buffered_errors(
                    model,
                    xx,
                    slack,
                    residual,
                    work,
                    primal_err;
                    p,
                )
                @test result[1:2] == complete[1:2]
                @test result[3] == (isnothing(residual) ? 0 : complete[3])
                @test result[4] == (isnothing(slack) ? 0 : complete[4])
                @test result[5] == complete[5]
                @test result[6] == (isnothing(slack) ? 0 : complete[6])
                @test buffered_errors_allocations(
                    model,
                    xx,
                    slack,
                    residual,
                    work,
                    primal_err,
                    p,
                ) == 0
            end
            for i in LRO.matrix_indices(model)
                @test xx[i] == original_x[i.value]
                @test ss[i] == original_s[i.value]
            end
        end
    end
end

end
