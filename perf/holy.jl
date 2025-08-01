using Dualization
import SDPLRPlus
include(joinpath(dirname(@__DIR__), "examples", "holy_model.jl"))
n = 30
A = data(n)
lr = holy_lowrank(A)
set_optimizer(lr, dual_optimizer(LRO.Optimizer))
set_attribute(lr, "solver", LRO.BurerMonteiro.Solver)
set_attribute(lr, "sub_solver", SDPLRPlus.Solver)
set_attribute(lr, "ranks", [15])
set_attribute(lr, "maxmajoriter", 0)
set_attribute(lr, "square_scalars", true)
optimize!(lr)

solver = unsafe_backend(lr).dual_problem.dual_model.model.optimizer.solver;
aux = solver.model;
var = solver.solver.var;

using BenchmarkTools
import NLPModels
const BM = LRO.BurerMonteiro
function jtprod(model, var)
    x = var.Rt
    y = view(var.y, 1:model.meta.ncon)
    Jtv = var.Gt
    @btime NLPModels.jtprod!($model, $x, $y, $Jtv)

    X = BM.Solution(x, model.dim)
    JtV = BM.Solution(Jtv, model.dim)
    @btime BM.jtprod!($model, $X, $y, LRO.left_factor($JtV, LRO.ScalarIndex), LRO.ScalarIndex)

    i = LRO.MatrixIndex(1)
    @btime BM.add_jtprod!($model, $X[$i], $y, $JtV[$i], $i)

    j = 1
    res = JtV[i].factor
    A = LRO.jac(model.model, j, i)
    B = X[i].factor
    α = 2y[j]
    buffer = model.jtprod_buffer[]
    @btime LRO.buffered_mul!($res, $A, $B, $α, true, $buffer)

    C = LRO._mul_to!(buffer, B', LRO.right_factor(A))
    C = LRO._rmul_diag!!(C, A.scaling)
    lA = LRO.left_factor(A)
    @btime LRO._add_mul!($res, $lA, $C', $α)
end

jtprod(aux, var)

function jtprod_matrix(model, var)
    x = var.Rt
    y = view(var.y, 1:model.meta.ncon)
    Jtv = var.Gt
    #@time NLPModels.jtprod!(model, x, y, Jtv)
    X = @time BM.Solution(x, model.dim)
    JtV = @time BM.Solution(Jtv, model.dim)
    #@time BM.jtprod!(model, X, y, LRO.left_factor(JtV, LRO.ScalarIndex), LRO.ScalarIndex)
    i = LRO.MatrixIndex(1)
    Xi = @time X[i]
    JtVi = @time JtV[i]
    @show length(y)
    i = LRO.MatrixIndex(1)
    #@btime BM.add_jtprod!($model, $X[i], $y, $JtV[i], $i)
    j = 1
    res = JtVi.factor
    A = @time LRO.jac(model.model, j, i)
    B = Xi.factor
    α = 2y[j]
    buffer = model.jtprod_buffer[]
    #@edit LinearAlgebra.mul!(res, A, B, α, true)
    #@profview for i in 1:1000_000
    #    LRO.buffered_mul!(res, A, B, α, true, buffer)
    #end
    @btime LRO.buffered_mul!($res, $A, $B, $α, true, $buffer)
    #@btime LRO.buffered_mul!($res, $A, $B, $α, true, $buffer)

    C = LRO._mul_to!(buffer, B', LRO.right_factor(A))
    C = LRO._rmul_diag!!(C, A.scaling)
    lA = LRO.left_factor(A)
    #@code_native LRO._add_mul!(res, lA, C', α)
    @btime LRO._add_mul!($res, $lA, $C', $α)
end
jtprod_matrix(aux, var);
