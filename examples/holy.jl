# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using Dualization
include("holy_model.jl")

n = 30
A = data(n)
cl = holy_classic(A)

# Let's start with SCS:

import SCS
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

lr = holy_lowrank(A)

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

# Let's try with SDPLRPlus:

import SDPLRPlus
set_optimizer(cl, dual_optimizer(LRO.Optimizer))
set_attribute(cl, "solver", LRO.BurerMonteiro.Solver)
set_attribute(cl, "sub_solver", SDPLRPlus.Solver)
set_attribute(cl, "ranks", [15])
set_attribute(cl, "maxmajoriter", 5)
optimize!(cl)

set_optimizer(lr, dual_optimizer(LRO.Optimizer))
set_attribute(lr, "solver", LRO.BurerMonteiro.Solver)
set_attribute(lr, "sub_solver", SDPLRPlus.Solver)
set_attribute(lr, "ranks", [15])
set_attribute(lr, "maxmajoriter", 5)
optimize!(lr)

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
