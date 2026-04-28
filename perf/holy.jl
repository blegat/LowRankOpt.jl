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
set_attribute(lr, "maxmajoriter", 20)
set_attribute(lr, "square_scalars", true)
@profview optimize!(lr)
#function g(lr)
    solver = unsafe_backend(lr).dual_problem.dual_model.model.optimizer.solver;
    model = solver.model;
    var = solver.solver.var;
    A = LRO.jac(model.model, LRO.ScalarIndex)
    y = view(var.y, 1:model.meta.ncon)
    JtV = view(var.Gt, 1:size(A, 2))
    C = JtV
    @edit LinearAlgebra.mul!(JtV, A', y, 1.0, 0.0)
    MKLSparse.cscmv!('T', 1.0, "GUUF", A, y, 0.0, C)
    for i in 1:10
        y = view(var.y, 1:model.meta.ncon)
        NLPModels.jtprod!(model, var.Rt, y, var.Gt)
        var.y .* 2
    end
#end
g(lr)

using BenchmarkTools
import NLPModels
const BM = LRO.BurerMonteiro

function jtprod(model, var)
    x = var.Rt
    y = view(var.y, 1:model.meta.ncon)
    Jtv = var.Gt
    println("jtprod!")
    @profview for _ in 1:10000
        SDPLRPlus.𝒜t!(
            Jtv,
            x,
            model,
            var,
        )
        #NLPModels.jtprod!(model, x, y, Jtv)
    end
    return
    @btime NLPModels.jtprod!($model, $x, $y, $Jtv)

    X = BM.Solution(x, model.dim)
    JtV = BM.Solution(Jtv, model.dim)
    println("Scalar jtprod!")
    @btime BM.jtprod!(
        $model,
        $X,
        $y,
        LRO.left_factor($JtV, LRO.ScalarIndex),
        LRO.ScalarIndex,
    )

    i = LRO.MatrixIndex(1)
    println("Matrix jtprod!")
    @btime BM.add_jtprod!($model, $X[$i], $y, $JtV[$i], $i)

    j = 1
    res = JtV[i].factor
    A = LRO.jac(model.model, j, i)
    B = X[i].factor
    α = 2y[j]
    buffer = model.jtprod_buffer[]
    println("buffered_mul!")
    @btime LRO.buffered_mul!($res, $A, $B, $α, true, $buffer)
end

jtprod(aux, var)
