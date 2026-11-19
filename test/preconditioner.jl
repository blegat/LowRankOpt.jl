using Test
using LinearAlgebra
using SparseArrays
using Random
import LowRankOpt as LRO
import NLPModels
import JSOSolvers

const BM = LRO.BurerMonteiro

function gram_test_model(sizes, ranks; scalars = 0)
    src = LRO.Model(
        [zeros(n, n) for n in sizes],
        Matrix{Matrix{Float64}}(undef, length(sizes), 0),
        Float64[],
        spzeros(scalars),
        spzeros(0, scalars),
        sizes,
    )
    return BM.Model{false}(src, ranks)
end

@testset "Gram preconditioner" begin
    rng = MersenneTwister(42)
    model = gram_test_model([4, 3], [2, 2]; scalars = 1)
    x = randn(rng, model.meta.nvar)
    # Exercise an exactly rank-deficient block.
    X = BM.Solution(x, model.dim)
    X[LRO.MatrixIndex(2)].factor[:, 2] .= 0
    M = BM.GramPreconditioner(model; regularization = 0.01)
    BM.update!(M, x)
    @test size(M) == (length(x), length(x))
    @test size(M, 1) == length(x)
    @test eltype(M) == Float64
    v = randn(rng, length(x))
    z = similar(v)
    mul!(z, M, v)
    @test z[1] == v[1]
    Z, V = BM.Solution(z, model.dim), BM.Solution(v, model.dim)
    for i in LRO.matrix_indices(model.model)
        U = X[i].factor
        @test Z[i].factor ≈ V[i].factor / (U'U + 0.01I)
    end
    @test dot(v, z) > 0
    w = similar(v)
    ldiv!(w, M, z)
    @test w ≈ v
    mul!(w, M, w)
    @test w ≈ z
    ldiv!(w, M, w)
    @test w ≈ v
    # Orthogonal changes of factor must rotate the result in the same way.
    Q = Matrix(qr(randn(rng, 2, 2)).Q)
    xq, vq = copy(x), copy(v)
    Xq, Vq = BM.Solution(xq, model.dim), BM.Solution(vq, model.dim)
    for i in LRO.matrix_indices(model.model)
        Xq[i].factor .= X[i].factor * Q
        Vq[i].factor .= V[i].factor * Q
    end
    BM.update!(M, xq)
    mul!(w, M, vq)
    W = BM.Solution(w, model.dim)
    for i in LRO.matrix_indices(model.model)
        @test W[i].factor ≈ Z[i].factor * Q
    end
    for δ in (0.0, -1.0, Inf, NaN)
        @test_throws ArgumentError BM.GramPreconditioner(
            model;
            regularization = δ,
        )
    end
    @test_throws DimensionMismatch BM.update!(M, x[2:end])
    @test_throws DimensionMismatch mul!(z, M, v[2:end])
    @test_throws DimensionMismatch ldiv!(z, M, v[2:end])
    scalar_model = gram_test_model(Int[], Int[]; scalars = 2)
    scalar_M = BM.GramPreconditioner(scalar_model)
    scalar_z = zeros(2)
    mul!(scalar_z, scalar_M, v[1:2])
    @test scalar_z == v[1:2]
end

# A squared constraint residual, using BM's negative-multiplier Hessian
# convention explicitly rather than relying on an outer solver's convention.
struct GramPenaltyModel{M} <:
       NLPModels.AbstractNLPModel{Float64,Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64,Vector{Float64}}
    counters::NLPModels.Counters
    bm::M
end

function NLPModels.obj(model::GramPenaltyModel, x::AbstractVector)
    NLPModels.increment!(model, :neval_obj)
    c = NLPModels.cons(model.bm, x) - model.bm.meta.lcon
    return sum(abs2, c) / 2
end

function NLPModels.grad!(
    model::GramPenaltyModel,
    x::AbstractVector,
    g::AbstractVector,
)
    NLPModels.increment!(model, :neval_grad)
    c = NLPModels.cons(model.bm, x) - model.bm.meta.lcon
    return NLPModels.jtprod!(model.bm, x, c, g)
end

function NLPModels.hprod!(
    model::GramPenaltyModel,
    x::AbstractVector,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight = 1.0,
)
    NLPModels.increment!(model, :neval_hprod)
    c = NLPModels.cons(model.bm, x) - model.bm.meta.lcon
    NLPModels.hprod!(model.bm, x, -c, v, Hv; obj_weight = 0.0)
    Hv .+= NLPModels.jtprod(model.bm, x, NLPModels.jprod(model.bm, x, v))
    Hv .*= obj_weight
    return Hv
end

@testset "Trunk CG with a lower-rank PSD target" begin
    n, p = 8, 3
    target = Diagonal([1.0, 0.01, zeros(n - 2)...])
    # The penalty for all symmetric entries is 1/2 * norm(UU' - target)^2.
    constraints = Matrix{Float64}[]
    rhs = Float64[]
    for j in 1:n, i in 1:j
        A = zeros(n, n)
        A[i, j] = A[j, i] = i == j ? 1.0 : inv(sqrt(2.0))
        push!(constraints, A)
        push!(rhs, dot(A, target))
    end
    src = LRO.Model(
        [zeros(n, n)],
        reshape(constraints, 1, :),
        rhs,
        spzeros(0),
        spzeros(length(rhs), 0),
        [n],
    )
    bm = BM.Model{false}(src, [p])
    x0 = vec(randn(MersenneTwister(12), n, p))
    results = []
    for preconditioned in (false, true)
        al = GramPenaltyModel(
            NLPModels.NLPModelMeta(length(x0); x0 = copy(x0)),
            NLPModels.Counters(),
            bm,
        )
        v = randn(MersenneTwister(23), length(x0))
        step = 1e-5
        @test NLPModels.hprod(al, x0, v) ≈
              (
            NLPModels.grad(al, x0 + step * v) -
            NLPModels.grad(al, x0 - step * v)
        ) / (2step) rtol = 1e-7
        M = BM.GramPreconditioner(bm; regularization = 1e-6)
        BM.update!(M, x0)
        callback = (nlp, solver, stats) -> BM.update!(M, solver.x)
        stats = JSOSolvers.trunk(
            al;
            x = copy(x0),
            M = preconditioned ? M : I,
            callback = preconditioned ? callback : (args...) -> nothing,
            atol = 1e-9,
            rtol = 0.0,
            max_iter = 1000,
            max_time = 60.0,
        )
        U = reshape(stats.solution, n, p)
        err = norm(U * U' - target)
        @test stats.status == :first_order
        @test err < 1e-5
        push!(
            results,
            (preconditioned, stats.iter, al.counters.neval_hprod, err),
        )
        NLPModels.reset!(bm)
    end
    @info "Trunk CG comparison (preconditioned, iterations, Hessian products, error)" results
end
