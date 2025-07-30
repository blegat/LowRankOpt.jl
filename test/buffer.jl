module TestBuffer

import FillArrays, SparseArrays
using JuMP, Dualization
include("diff_check.jl")

# Test with zero Ai matrices
function test_zero_Ai()
    model = Model(dual_optimizer(LRO.Optimizer))
    @variable(model, x[1:2] in MOI.Nonnegatives(2))
    @variable(model, X[1:2, 1:2] in PSDCone())
    @constraint(model, sum(x) == 1)
    @constraint(model, 2sum(x) == 2)
    @constraint(model, sum(X) == 2)
    @constraint(model, x[1] - x[2] == 1)
    @objective(model, Max, x[1])
    set_attribute(model, "solver", ConvexSolver)
    optimize!(model)
    b = _backend(model)
    T = Float64
    Z = FillArrays.Zeros{T,2,Tuple{Base.OneTo{Int},Base.OneTo{Int}}}
    S = SparseArrays.SparseMatrixCSC{T,Int}
    @test b.model.A isa Matrix{Union{Z,S}}
    @test b.model.A[1] isa Z
    @test b.model.A[2] isa Z
    @test b.model.A[3] isa S
    @test b.model.A[4] isa Z
    buf = LRO.BufferedModelForSchur(b.model, 1)
    for A in b.model.A
        @test buf.jtprod_buffer[] !== A
    end
end

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
end

end

TestBuffer.runtests()
