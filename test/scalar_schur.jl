module TestScalarSchur

using Test, LinearAlgebra, SparseArrays
import LowRankOpt as LRO

function scalar_model(C::SparseMatrixCSC{T}) where {T}
    m, n = size(C)
    # NLPModels requires at least one variable, even for an empty scalar block.
    dims = iszero(n) ? Int64[1] : Int64[]
    model = LRO.Model(
        [spzeros(T, d, d) for d in dims],
        [spzeros(T, d, d) for d in dims, _ in 1:m],
        zeros(T, m),
        SparseVector{T,Int64}(spzeros(T, n)),
        SparseMatrixCSC{T,Int64}(C),
        dims,
    )
    return LRO.BufferedModelForSchur(model, 1)
end

function scalar_schur_allocations(model, w, H)
    return @allocated LRO.add_schur_complement!(model, w, LRO.ScalarIndex, H)
end

@testset "Scalar Schur assembly reuses its destination" begin
    for T in (Float32, Float64)
        # Overlapping columns, an empty column/row, and an explicit zero.
        C = sparse(
            [1, 3, 2, 3, 1, 2, 4],
            [1, 1, 2, 2, 4, 4, 4],
            T[2, -1, 3, 2, 1, -4, 0],
            4,
            6,
        )
        model = scalar_model(C)
        original = copy(model.model.C_lin)
        H0 = reshape(T.(1:16), 4, 4) ./ T(10)
        for destination in (copy, H -> view(copy(H), :, :))
            H = destination(H0)
            expected = copy(H0)
            for w in (T[1, 2, 0, 3, 4, -1], zeros(T, 6), -ones(T, 6))
                # Exercise the scalar-slice view used by VectorizedSolution too.
                weights = view(vcat(w, zero(T)), 1:6)
                contribution = C * spdiagm(w) * C'
                expected .+= contribution
                @test LRO.add_schur_complement!(
                    model,
                    weights,
                    LRO.ScalarIndex,
                    H,
                ) === H
                @test H ≈ expected
                allocated = scalar_schur_allocations(model, weights, H)
                @test allocated == 0
                expected .+= contribution
                @test H ≈ expected
                @test model.model.C_lin == original
            end
        end
    end
end

@testset "Empty scalar Schur blocks" begin
    for T in (Float32, Float64), (m, n) in ((0, 0), (0, 3), (3, 0), (3, 4))
        model = scalar_model(spzeros(T, m, n))
        w, H = ones(T, n), ones(T, m, m)
        @test LRO.add_schur_complement!(model, w, LRO.ScalarIndex, H) === H
        @test H == ones(T, m, m)
        @test scalar_schur_allocations(model, w, H) == 0
    end
end

end
