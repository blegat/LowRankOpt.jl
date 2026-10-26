@inline function block_errors(
    model,
    x,
    i,
    dual_err,
    dual_slack,
    workspace,
    p,
    errors,
)
    err2, err3, err4, err6, C_norm = errors
    X = x[i]
    isempty(X) && return (err2, err3, err4, err6, C_norm)
    C = grad(model, i)
    C_norm = LinearAlgebra.norm((C_norm, LinearAlgebra.norm(C, p)), p)
    min_x = i === ScalarIndex ? minimum(X) : matrix_eigmin(X, i, workspace)
    err2 = max(err2, -min_x)
    if !isnothing(dual_err)
        err3 = hypot(err3, LinearAlgebra.norm(dual_err[i]))
    end
    if !isnothing(dual_slack)
        S = dual_slack[i]
        min_s = i === ScalarIndex ? minimum(S) : matrix_eigmin(S, i, workspace)
        err4 = max(err4, -min_s)
        err6 += LinearAlgebra.dot(S, X)
    end
    return (err2, err3, err4, err6, C_norm)
end

function matrix_eigmin(A, ::MatrixIndex, ::Nothing)
    return LinearAlgebra.eigmin(LinearAlgebra.Symmetric(A))
end
function matrix_eigmin(A, i::MatrixIndex, workspace::Vector{<:EigenWorkspace})
    return eigmin!(workspace[i.value], A)
end

"""
    errors(model::AbstractModel, x; y = nothing, p = 2, ...)

Return [the six DIMACS-style errors](https://plato.asu.edu/dimacs/node3.html).

For a collection `v = (v_scalar, V_1, ..., V_k)` of scalar variables and
matrix blocks, distinguish these entrywise norms:

- `F(v) = sqrt(norm(v_scalar, 2)^2 + sum(norm(V_i)^2))`: the global
  Euclidean/Frobenius norm, called `Fnorm` in CSDP.
- `K(v) = norm(v_scalar, 2) + sum(norm(V_i))`: the sum of the block
  Euclidean/Frobenius norms, called `Knorm` in CSDP.
- `L1(v)`: the sum of the absolute values of all scalar and matrix entries.
- `Linf(v)`: the largest absolute value among those entries.

For a single vector, `F` and `K` both reduce to its Euclidean norm. For
matrix blocks `[3]` and `[4]`, `F = 5` while `K = 7`. Matrix norms here are
entrywise norms, not induced operator norms.

This function uses `Fnorm` (`F` above) for both equation-residual
numerators. Cone violations use the worst eigenvalue or scalar across all
blocks. `p` selects only the data
denominators `1 + norm(b, p)` and `1 + norm(C, p)`, combining all objective
entries into one norm. Thus `p = 2` (the default) uses `F`, `p = 1` uses
`L1`, and `p = Inf` uses `Linf` for the data.

The linked DIMACS description labels the data norm as a 1-norm but defines
it as the largest absolute component, which is `Linf`. The following table
separates internal feasibility measures from reported equation residuals.
It does not describe cone errors, gap tests, or infeasibility certificates.

In the table, `r` is the equation residual and `d` is its data: `b` on the
primal side and `C` (or `c`/`q`) on the dual side. A single norm applies on
both sides; `P / D` distinguishes primal and dual choices. `1` denotes an
unnormalized reported residual. The less common scales are:

- Clarabel: `Dcl_p = max(1, Linf(b) + F(x) + F(s))` and
  `Dcl_d = max(1, Linf(q) + F(x) + F(z))`.
- SCS: `T_p = max(Linf(b), Linf(A*x), Linf(s))` and
  `T_d = max(Linf(c), Linf(P*x), Linf(A'*y))`.
  `Dsc = eps_abs + eps_rel * T` is the full residual bound; SCS compares
  `Linf(r)` directly against this bound, rather than a separate tolerance.

| Solver | Internal numerator | Internal denominator | Report numerator | Report denominator |
| :--- | :--- | :--- | :--- | :--- |
| Loraine (`p = 2`) | `F(r)` | `1 + F(d)` | `F(r)` | `1 + F(d)` |
| [SeDuMi](https://github.com/sqlp/sedumi/blob/master/sedumi.m) | `weighted Linf` | `1 + Linf(d)` | `F(r) / 0` | `1 + L1(b) / —` |
| [SDPT3](https://github.com/sqlp/sdpt3/blob/master/Solver/sqlpmain.m) | `F(r)` | `1 + F(d)` | `F(r)` | `1 + Linf(d)` |
| [CSDP](https://github.com/coin-or/Csdp/blob/master/lib/psd_feas.c) | `F(r)` | `1 + F(d)` | `F(r) / K(r)` | `1 + Linf(d)` |
| [MOSEK](https://docs.mosek.com/latest/capi/solving-conic.html) | `Linf(r)` | `1 + Linf(d)` | `Linf(r_h)` | `1` |
| [Clarabel](https://github.com/oxfordcontrol/Clarabel.jl/blob/main/src/info.jl) | `F(r)` | `Dcl_p / Dcl_d` | `F(r)` | `Dcl_p / Dcl_d` |
| [SCS](https://www.cvxgrp.org/scs/algorithm/index.html) (default) | `Linf(r)` | `Dsc` | `Linf(r)` | `1` |
| [COPT](https://guide.coap.online/copt/en-doc/logging.html) | unknown | unknown | unknown | unknown |

"Report" means the optional DIMACS report for SeDuMi, SDPT3, and CSDP;
Clarabel's printed residuals; SCS's returned residuals; and MOSEK's
`PFEAS`/`DFEAS` log fields. These are different outputs:

- SeDuMi's internal row describes the weighted residual terms used in its
  homogeneous-embedding precision tests, not the complete stopping test.
  Its DIMACS `err[3]` is always zero because it is not maintained explicitly.
  Its cone-error denominators still use `1 + L1(b)` and `1 + L1(c)`.
- MOSEK's log uses homogeneous residuals `r_h`, such as `A*x - b*tau`.
  Its internal tests use the recovered solution, such as `A*(x/tau) - b`.
- COPT's public documentation describes absolute/relative infeasibility
  output and tolerances but does not establish these normalization formulas.

Additional norm definitions are in SeDuMi's
[sdinit.m](https://github.com/sqlp/sedumi/blob/master/sdinit.m), CSDP's
[Fnorm.c](https://github.com/coin-or/Csdp/blob/master/lib/Fnorm.c), and
Clarabel's `problemdata.jl` and `utils/mathutils.jl`:
<https://github.com/oxfordcontrol/Clarabel.jl/tree/main/src>.

The default `p = 2` matches the Euclidean/Frobenius residual/data scaling
used by SDPT3 and CSDP internally. Selecting `p = Inf` only changes our data
denominators; it does not reproduce another solver's full stopping test.
For fixed residuals, larger data norms yield smaller errors, so tolerances
should be compared using the same normalization.

The signed duality gap and complementarity share the denominator
`1 + abs(pobj) + abs(dobj)`, including the scalar objective.

`workspace` may be a vector of `EigenWorkspace`s, one per PSD block, to
reuse eigensolver storage. The primal and dual solutions are preserved.
"""
function errors(
    model::AbstractModel,
    x;
    y = nothing,
    primal_err = NLPModels.cons(model, x),
    dual_slack = nothing,
    dual_err = nothing,
    pobj = NLPModels.obj(model, x),
    dobj = dual_obj(model, y),
    workspace::Union{Nothing,Vector{<:EigenWorkspace}} = nothing,
    p = 2,
)
    p >= 1 || throw(ArgumentError("the denominator norm requires p >= 1"))
    b_den = 1 + LinearAlgebra.norm(cons_constant(model), p)
    block_err =
        (zero(b_den), zero(b_den), zero(b_den), zero(b_den), zero(b_den))
    # Keep matrix and scalar indices in separate calls so each block's
    # arithmetic specializes on its solution and objective storage.
    for i in matrix_indices(model)
        block_err = block_errors(
            model,
            x,
            i,
            dual_err,
            dual_slack,
            workspace,
            p,
            block_err,
        )
    end
    err2, err3, err4, err6, C_norm = block_errors(
        model,
        x,
        ScalarIndex,
        dual_err,
        dual_slack,
        workspace,
        p,
        block_err,
    )
    C_den = 1 + C_norm
    obj_den = 1 + abs(pobj) + abs(dobj)
    return (
        LinearAlgebra.norm(primal_err) / b_den,
        err2 / b_den,
        err3 / C_den,
        err4 / C_den,
        (pobj - dobj) / obj_den,
        err6 / obj_den,
    )
end
