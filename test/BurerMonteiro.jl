using Test
using LowRankOpt
include(joinpath(dirname(@__DIR__), "examples", "maxcut.jl"))
weights = [0 5 7 6; 5 0 0 1; 7 0 0 1; 6 1 1 0];
model = maxcut(weights, LowRankOpt.Optimizer)

import Percival
set_attribute(model, "solver", LRO.BurerMonteiro.Solver)
set_attribute(model, "sub_solver", Percival.PercivalSolver)
set_attribute(model, "ranks", [1])
optimize!(model)
