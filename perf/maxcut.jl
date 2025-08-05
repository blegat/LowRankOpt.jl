using LowRankOpt
using Dualization
include(joinpath(dirname(@__DIR__), "examples", "maxcut.jl"))

using SDPLRPlus
import Random
using BenchmarkTools

function A_sym_bench(aux, var)
    println("𝒜 sym")
    @btime SDPLRPlus.𝒜!($var.primal_vio, $aux, $var.Rt)
end

function A_not_sym_bench(aux, var)
    println("𝒜 not sym")
    @btime SDPLRPlus.𝒜!($var.A_RD, $aux, $var.Rt, $var.Gt)
end

function At_bench(aux, var)
    println("𝒜t")
    @btime SDPLRPlus.𝒜t!($var.Gt, $var.Rt, $aux, $var)
end

function At_bench2(aux, var)
    println("𝒜t rank-1")
    n = SDPLRPlus.side_dimension(aux)
    x = rand(n)
    y = similar(x)
    @btime SDPLRPlus.𝒜t!($y, $aux, $x, $var)
end

function bench_lmul(A)
    n = LinearAlgebra.checksquare(A)
    x = rand(1, n)
    y = similar(x)
    println("lmul")
    @btime LinearAlgebra.mul!($y, $x, $A, 2.0, 1.0)
end

function bench_rmul(A)
    n = LinearAlgebra.checksquare(A)
    x = rand(n, 1)
    y = similar(x)
    println("rmul")
    if A.factor isa AbstractVector
        buffer = zeros(1)
    else
        buffer = zeros(1, LRO.max_rank(A))
    end
    @btime LRO.buffered_mul!($y, $A, $x, 2.0, 1.0, $buffer)
end

function bench(aux, var)
    A_sym_bench(aux, var)
    A_not_sym_bench(aux, var)
    At_bench(aux, var)
    At_bench2(aux, var)
    return
end

function weights(n; p = 0.1)
    Random.seed!(0)
    W = sprand(n, n, p)
    return W + W'
end

function bench_plus(args...; kws...)
    C = maxcut_objective(weights(args...; kws...))
    As = [SymLowRankMatrix(Diagonal(ones(1)), e_i(Float64, i, n, sparse = false, vector = false)) for i in 1:n]
    b = ones(n)
    d = SDPLRPlus.SDPData(C, As, b)
    var = SDPLRPlus.SolverVars(d, 1)
    aux = SDPLRPlus.SolverAuxiliary(d)
    bench(aux, var)
    bench_lmul(aux.symlowrank_As[1]);
end

function bench_lro(args...; vector, kws...)
    println("vector ? $vector")
    model = maxcut(weights(args...; kws...), dual_optimizer(LRO.Optimizer); vector)
    set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
    set_attribute(model, "sub_solver", SDPLRPlus.Solver)
    set_attribute(model, "ranks", [1])
    set_attribute(model, "maxmajoriter", 0)
    set_attribute(model, "printlevel", 3)
    @time optimize!(model)
    solver = unsafe_backend(model).dual_problem.dual_model.model.optimizer.solver
    aux = solver.model
    var = solver.solver.var
    bench(aux, var)
    bench_rmul(aux.model.A[1])
end

n = 500
p = 0.1
bench_plus(n; p)
bench_lro(n; p, vector = true)
bench_lro(n; p, vector = false)
