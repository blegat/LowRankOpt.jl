# LowRankOpt

| **Build Status** |
|:----------------:|
| [![Build Status][build-img]][build-url] [![Codecov branch][codecov-img]][codecov-url] |

Extends MathOptInterface (MOI) to low-rank constraints.

[build-img]: https://github.com/blegat/LowRankOpt.jl/actions/workflows/ci.yml/badge.svg?branch=main
[build-url]: https://github.com/blegat/LowRankOpt.jl/actions?query=workflow%3ACI
[codecov-img]: https://codecov.io/gh/blegat/LowRankOpt.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/blegat/LowRankOpt.jl/branch/main

See "Why you should stop using the monomial basis" at [JuMP-dev 2024](https://jump.dev/meetings/jumpdev2024/) : [slides](https://jump.dev/assets/jump-dev-workshops/2024/legat.html) [video](https://youtu.be/CGPHaHxCG2w)

Started as an [MOI issue](https://github.com/jump-dev/MathOptInterface.jl/issues/2197) and [MOI PR](https://github.com/jump-dev/MathOptInterface.jl/pull/2198).

## Use with JuMP

To use LowRankOpt with [JuMP](https://github.com/jump-dev/JuMP.jl), use
add its bridge to your model with:
```julia
model = Model()
LRO.add_all_bridges(model, Float64)
```
Then, either use the `LRO.SetDotProducts` or `LRO.LinearCombinationInSet`.
[Check with `print_active_bridges(model)`](https://jump.dev/JuMP.jl/stable/tutorials/conic/ellipse_approx/)
to see if the solver receives the low-rank constraint or if it is transformed to classical constraints.

The solvers that support `LRO.SetDotProducts` are:

* [DSDP.jl](https://github.com/jump-dev/DSDP.jl) : [⚠ WIP](https://github.com/jump-dev/DSDP.jl/pull/37)
* [Hypatia.jl](https://github.com/jump-dev/Hypatia.jl) : [⚠ WIP](https://github.com/jump-dev/Hypatia.jl/pull/844)
* [SDPLR.jl](https://github.com/jump-dev/SDPLR.jl) : [⚠ WIP](https://github.com/jump-dev/SDPLR.jl/pull/26)

The solvers that support `LRO.LinearCombinationInSet` are:

* [Hypatia.jl](https://github.com/jump-dev/Hypatia.jl) : [⚠ WIP](https://github.com/jump-dev/Hypatia.jl/pull/844)

If you use `LRO.LinearCombinationInSet` while the solvers supports `LRO.SetDotProducts` or vice versa, simply [use a `Dualization.jl` layer](https://jump.dev/JuMP.jl/stable/tutorials/conic/dualization/).

Note that `Hypatia.jl` only supports `LRO.SetDotProducts{LRO.WITHOUT_SET}` or `LRO.LinearCombinationInSet{LRO.WITHOUT_SET}` and not the `LRO.WITH_SET` version.

## Example

Below is [this example](https://github.com/jump-dev/SDPLR.jl?tab=readme-ov-file#example-modifying-the-rank-and-checking-optimality)
adapted to exploit the low-rank constraints.

```julia-repl
julia> include(joinpath(dirname(dirname(pathof(LowRankOpt))), "examples", "maxcut.jl"))
maxcut (generic function with 2 methods)

julia> weights = [0 5 7 6; 5 0 0 1; 7 0 0 1; 6 1 1 0];

julia> model = maxcut(weights, SDPLR.Optimizer);

julia> optimize!(model)

            ***   SDPLR 1.03-beta   ***

===================================================
 major   minor        val        infeas      time
---------------------------------------------------
    1        9   1.77916821e+02  2.0e+01       0
    2       12  -1.78345610e+01  4.7e-01       0
    3       14  -1.80546137e+01  2.5e-01       0
    4       16  -1.80170234e+01  9.7e-02       0
    5       18  -1.79967453e+01  4.0e-02       0
    6       19  -1.79987069e+01  1.1e-02       0
    7       20  -1.79999405e+01  1.7e-03       0
    8       21  -1.79999841e+01  3.3e-04       0
    9       22  -1.79999912e+01  5.9e-06       0
===================================================

DIMACS error measures: 5.86e-06 0.00e+00 0.00e+00 0.00e+00 2.25e-05 9.84e-06


julia> objective_value(model)
17.99998881724702

julia> con_ref = VariableInSetRef(model[:dot_prod_set]);
```

Use `LRO.InnerAttribute` to request the `SDPLR.Factor` attribute.
```julia-repl
julia> MOI.get(model, LRO.InnerAttribute(SDPLR.Factor()), VariableInSetRef(model[:dot_prod_set]))
4×3 Matrix{Float64}:
  0.949505   0.31101    0.0414433
 -0.950269  -0.308646  -0.0415714
 -0.949503  -0.311095  -0.0406689
 -0.948855  -0.312898  -0.0420402
```
We can see that SDPLR decided to search for a solution of rank at most 3.
To check if SDPLR found an optimal solution, we need to check whether the dual solution is feasible.
This can be achieved as follows:
```julia-repl
julia> dual_set = MOI.dual_set(constraint_object(con_ref).set);

julia> MOI.Utilities.distance_to_set(dual(con_ref), dual_set)
1.602881211577939e-8
```

For the MAX-CUT problem, we know there exists a rank-1 solution where
the entries of the factor are `-1` or `1` depending on the side of the cut
the nodes are on. Let's now be greedy and search for a solution of rank-1.
```julia
julia> set_attribute(model, "maxrank", (m, n) -> 1)

julia> optimize!(model)

            ***   SDPLR 1.03-beta   ***

===================================================
 major   minor        val        infeas      time  
---------------------------------------------------
    1        0   2.68869170e-01  8.7e-01       0
    2        7   8.10081717e+01  1.0e+01       0
    3       10  -1.81587378e+01  2.3e-01       0
    4       11  -1.80062673e+01  1.3e-01       0
    5       13  -1.79951904e+01  5.0e-02       0
    6       15  -1.79981240e+01  1.4e-02       0
    7       16  -1.79998759e+01  2.2e-03       0
    8       17  -1.79999980e+01  1.9e-04       0
    9       18  -1.80000000e+01  1.4e-05       0
   10       19  -1.80000000e+01  7.1e-07       0
===================================================

DIMACS error measures: 7.12e-07 0.00e+00 0.00e+00 1.31e-05 1.59e-06 -7.81e-06


julia> MOI.get(model, LRO.InnerAttribute(SDPLR.Factor()), VariableInSetRef(model[:dot_prod_set]))
4×1 Matrix{Float64}:
 -0.9999998713637199
  0.9999995531365233
  0.9999997036204064
  0.9999995498147483
```

Note that even though a solution of rank-1 exists, SDPLR is more likely to
converge to a spurious local minimum if we use a lower-rank, so we should
be careful before claiming that we found the optimal solution.
Luckily, the following shows that the dual is feasible which gives a certificate
of primal optimality.
```julia-repl
julia> MOI.Utilities.distance_to_set(dual(con_ref), dual_set)
0.0
```
