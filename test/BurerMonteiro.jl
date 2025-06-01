using Test
using LowRankOpt
import Percival

include(joinpath(dirname(@__DIR__), "examples", "maxcut.jl"))
weights = [0 5 7 6; 5 0 0 1; 7 0 0 1; 6 1 1 0];
model = maxcut(weights, LowRankOpt.Optimizer)

set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
set_attribute(model, "sub_solver", Percival.PercivalSolver)
set_attribute(model, "ranks", [1])
optimize!(model)
solution_summary(model)

import NLPModels, FiniteDiff
function jac_check(model, x; tol = 1e-6)
    f(x) = NLPModels.cons(model, x)
    J = FiniteDiff.finite_difference_jacobian(f, x)
    v = rand(model.meta.nvar)
    @test NLPModels.jprod(model, x, v) ≈ J * v rtol = tol atol = tol
    v = rand(model.meta.ncon)
    @test NLPModels.jtprod(model, x, v) ≈ J' * v rtol = tol atol = tol
end


@testset "Diff check" begin
    b = unsafe_backend(model)
    using NLPModelsTest
    bm = b.solver.model
    x = rand(bm.meta.nvar)
    @test isempty(NLPModelsTest.gradient_check(bm; x))
    jac_check(bm, x)
end
