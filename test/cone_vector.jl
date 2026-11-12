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
        dims = LRO.ConeDimensions(
            length(scalars),
            size.(matrices, 1),
            cumsum([length(scalars); length.(matrices)]),
        )
        shaped = LRO.ShapedConeVector(scalars, matrices)
        other_shaped = LRO.ShapedConeVector(other_scalars, other_matrices)
        vectorized = LRO.VectorizedConeVector(flat, dims)
        other_vectorized = LRO.VectorizedConeVector(other_flat, dims)
        @test collect(shaped) == flat
        @test -shaped == -flat
        for i in (0, length(flat) + 1)
            @test_throws BoundsError(shaped, (i,)) shaped[i]
        end
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

@testset "Cone vectors retain view-backed storage" begin
    storage = fill(-99.0, 13)
    slice = view(storage, 2:2:12)
    slice .= 1:6
    dimensions = LRO.ConeDimensions(2, [2], [2, 6])
    cone = LRO.VectorizedConeVector(slice, dimensions)
    @test cone.x === slice
    @test Vector(cone) == collect(1.0:6.0)
    @test cone[LRO.ScalarIndex] == [1.0, 2.0]
    @test cone[LRO.MatrixIndex(1)] == [3.0 5.0; 4.0 6.0]
    cone[LRO.MatrixIndex(1)] .= [7.0 8.0; 8.0 9.0]
    @test slice == [1.0, 2.0, 7.0, 8.0, 8.0, 9.0]
    @test all(==(-99.0), storage[1:2:end])
    copied = similar(cone)
    copied .= cone
    @test copied == cone
    @test copied.x !== cone.x
end

end
