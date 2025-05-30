using Test
using LowRankOpt
include(joinpath(dirname(@__DIR__), "examples", "maxcut.jl"))
weights = [0 5 7 6; 5 0 0 1; 7 0 0 1; 6 1 1 0];
model = maxcut(weights, LowRankOpt.Optimizer)
optimize!(model)
