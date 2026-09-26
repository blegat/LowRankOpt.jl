module TestSolution

using Test, LinearAlgebra
import LowRankOpt as LRO

@testset "Solution block structure and inner products" begin
    # Distinct operands and negative entries exercise the bilinear product,
    # not just its special case used by norm. Include scalar-only and PSD-only
    # solutions so empty parts of the product cone are covered too.
    for (scalars, matrices, other_scalars, other_matrices) in (
        (
            [3.0, -4.0],
            [[1.0 2.0; 2.0 -1.0], fill(5.0, 1, 1)],
            [-2.0, 1.0],
            [[2.0 -1.0; -1.0 3.0], fill(-2.0, 1, 1)],
        ),
        ([3.0, -4.0], Matrix{Float64}[], [-2.0, 1.0], Matrix{Float64}[]),
        (Float64[], [[1.0 2.0; 2.0 -1.0]], Float64[], [[2.0 -1.0; -1.0 3.0]]),
    )
        flat = vcat(scalars, vec.(matrices)...)
        other_flat = vcat(other_scalars, vec.(other_matrices)...)
        dims = LRO.Dimensions(
            length(scalars),
            size.(matrices, 1),
            cumsum([length(scalars); length.(matrices)]),
        )
        shaped = LRO.ShapedSolution(scalars, matrices)
        other_shaped = LRO.ShapedSolution(other_scalars, other_matrices)
        vectorized = LRO.VectorizedSolution(flat, dims)
        other_vectorized = LRO.VectorizedSolution(other_flat, dims)
        @test LRO.num_matrices(dims) == length(matrices)
        for (x, y) in ((shaped, other_shaped), (vectorized, other_vectorized))
            @test LRO.num_matrices(x) == length(matrices)
            @test length(collect(LRO.matrix_indices(x))) == length(matrices)
            @test size(x) == size(flat)
            @test dot(x, y) ≈ dot(flat, other_flat)
            @test norm(x) ≈ norm(flat)
        end
    end
end

end
