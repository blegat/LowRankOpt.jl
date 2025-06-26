using LinearAlgebra, SparseArrays
using JuMP, Dualization
import LowRankOpt as LRO

# This is a model of a self-avoiding chain (e.g., a polymer) in arbitrary dimensions
# See https://github.com/jump-dev/JuMP.jl/issues/4011

function data(i, j, n)
    A = zeros(n, n)
    A[i, j] = A[j, i] = 1
    return A
end

function classic(A)
    n = LinearAlgebra.checksquare(A)
    model = Model()
    @variable(model, XX[1:n, 1:n], PSD)    # modeling the Gram matrix, XX[i, j] = X[:, i]'X[:, j]
    # Impose a penalty on distance between the paired nodes
    @objective(model, Min, sum(A[i, j] * (XX[i, i] + XX[j, j] - XX[i, j] - XX[j, i]) for i = 1:n-1, j = i+1:n))
    # Link successive nodes together, with an edge length of 1
    @constraint(model, [i = 1:n-1], XX[i, i] + XX[i+1, i+1] - XX[i, i+1] - XX[i+1, i] == 1)
    # Avoid self-intersections
    @constraint(model, [i = 1:n-2, j = i+2:n], XX[i, i] + XX[j, j] - XX[i, j] - XX[j, i] >= 1)
    return model
end


n = 30
# pick one random pairs to be close
i = rand(1:n-1)
j = rand(i+1:n)
A = data(i, j, n)

cl = classic(A)

# Let's start with SCS:

set_optimizer(cl, SCS.Optimizer)
optimize!(cl)
objective_value(cl)
solve_time(cl)

# SCS needs to creates 29 `free` variables, what are they?
# SCS expects a problem in the geometric/image form with
# free variables and PSD constraints.
# Here we rather have PSD variables and linear constraints
# so we are closer to the standard/kernel form so let's dualize:

import SCS
set_optimizer(cl, dual_optimizer(SCS.Optimizer))
optimize!(cl)
objective_value(cl)
solve_time(cl)

# We can see that the problem stats decreased slightly, good news.
# Our constraints are low-rank. The matrix `[1 -1; -1 1]` is equal
# to `[1, -1] * [1, -1]'` so it is rank-1.
# Do we gain anything by formulating the constraints matrices
# as rank-1 with sparse factors of 2 entries instead of generic sparse
# matrices of 4 entries ?

function _factor(i, j, n)
    F = sparsevec([i, j], Float64[1, -1], n)
    return LRO.positive_semidefinite_factorization(F)
end

function lowrank(A)
    n = LinearAlgebra.checksquare(A)
    model = Model()
    set = LRO.SetDotProducts{LRO.WITHOUT_SET}(
        MOI.PositiveSemidefiniteConeTriangle(n),
        LRO.TriangleVectorization.([
            _factor(i, j, n) for i = 1:n-1 for j = i+1:n
        ]),
    )
    @variable(model, x[1:MOI.dimension(set)] in set)
    Δ = Matrix{VariableRef}(undef, n, n)
    k = 0
    for i = 1:n-1
        for j = i+1:n
            k += 1
            Δ[i, j] = x[k]
        end
    end
    # Impose a penalty on distance between the paired nodes
    @objective(model, Min, sum(A[i, j] * Δ[i, j] for i = 1:n-1, j = i+1:n))
    # Link successive nodes together, with an edge length of 1
    @constraint(model, [i = 1:n-1], Δ[i, i+1] == 1)
    # Avoid self-intersections
    @constraint(model, [i = 1:n-2, j = i+2:n], Δ[i, j] >= 1)
    return model
end
lr = lowrank(A)

# If we try with SCS, we see no difference.
# Since SCS does not support it, it is simply reformulated into the exact same problem.

set_optimizer(lr, dual_optimizer(SCS.Optimizer))
LRO.Bridges.add_all_bridges(backend(lr).optimizer, Float64)
optimize!(lr)
objective_value(lr)
solve_time(lr)

# Let's try with SDPLR with rank-15 PSD matrices:

import SDPLR
set_optimizer(cl, SDPLR.Optimizer)
set_attribute(cl, "maxrank", (m, n) -> 15)
optimize!(cl)
objective_value(cl)
solve_time(cl)

# We can see that the solution is indeed of rank-15:

MOI.get(cl, SDPLR.Factor(), VariableInSetRef(cl[:XX]))

# We have some negative eigenvalues so we don't really have an optimality certificate
# but we saw the objective value is 1.0.

using LinearAlgebra
eigvals(dual(VariableInSetRef(cl[:XX])))

# SDPLR supports low-rank constraints as well so let's try with the low-rank model.

set_optimizer(lr, SDPLR.Optimizer)
LRO.Bridges.add_all_bridges(backend(lr).optimizer, Float64)
set_attribute(lr, "maxrank", (m, n) -> 15)
optimize!(lr)
objective_value(lr)
solve_time(lr)

# It is slower! This is because SDPLR does not support sparse factors so it
# desified our factors of 2 entries!

# Let's try with Percival:

import Percival

set_optimizer(cl, dual_optimizer(LRO.Optimizer))
set_attribute(cl, "solver", LRO.BurerMonteiro.Solver)
set_attribute(cl, "sub_solver", Percival.PercivalSolver)
set_attribute(cl, "ranks", [15])
set_attribute(cl, "verbose", 2)
optimize!(cl)
objective_value(cl)
solve_time(cl)

# We would like to try now with `LowRankOpt` with sparse low-rank factors.
# The feature is coming soon...
