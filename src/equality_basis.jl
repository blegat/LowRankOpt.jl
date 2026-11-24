"""
    equality_basis(model::Model; tolerance=0)

Return scalar equality rows after removing verified dependencies. Sparse
QR proposes dependencies on the cyclic part of the row/variable graph;
exact rational arithmetic verifies their coefficients. A removed row's
right-hand side must agree within `tolerance` in the original units.
The public model is unchanged; callers must still check its original rows.
"""
function equality_basis(model::Model{T}; tolerance = zero(T)) where {T}
    if !isfinite(tolerance) || tolerance < 0
        throw(
            ArgumentError(
                "The equality tolerance must be finite and nonnegative.",
            ),
        )
    end
    exact_tolerance=Rational{BigInt}(tolerance)
    rows=filter(i->model.meta.lcon[i]==model.meta.ucon[i], 1:num_scalars(model))
    if isempty(rows) || !(T <: Union{Float32,Float64})
        return rows
    end
    E=model.jacobian[rows, :]
    SparseArrays.dropzeros!(E)
    target=model.meta.lcon[rows]-model.constant[rows]
    if !all(isfinite, E.nzval) || !all(isfinite, target)
        return rows
    end
    Et=SparseArrays.sparse(transpose(E))
    degree=diff(E.colptr)
    queue=findall(==(1), degree)
    active=trues(length(rows))
    # A private variable proves its equality independent of the remaining
    # rows. Peeling these rows keeps the subsequent QR problem small.
    while !isempty(queue)
        column=pop!(queue)
        if degree[column]!=1
            continue
        end
        row=0
        for p in SparseArrays.nzrange(E, column)
            if active[E.rowval[p]]
                row=E.rowval[p]
                break
            end
        end
        active[row]=false
        for p in SparseArrays.nzrange(Et, row)
            j=Et.rowval[p]
            degree[j]-=1
            if degree[j]==1
                push!(queue, j)
            end
        end
    end
    core=findall(active)
    if isempty(core)
        return rows
    end
    A=E[core, :]
    columns=sort!(unique(SparseArrays.findnz(A)[2]))
    A=A[:, columns]
    keep=trues(length(rows))
    if isempty(columns)
        for i in core
            if abs(Rational{BigInt}(target[i]))<=exact_tolerance
                keep[i]=false
            end
        end
        return rows[keep]
    end
    factor=LinearAlgebra.qr(SparseArrays.sparse(transpose(A)))
    r=LinearAlgebra.rank(factor)
    if r==length(core)
        return rows
    end
    permutation=isempty(factor.pcol) ? collect(1:length(core)) : factor.pcol
    independent=permutation[1:r]
    dependent=permutation[(r+1):end]
    coefficients=LinearAlgebra.UpperTriangular(factor.R[1:r, 1:r])\Matrix(
        factor.R[1:r, (r+1):end],
    )
    transpose_A=SparseArrays.sparse(transpose(A))
    for (j, dependency) in enumerate(dependent)
        candidate=view(coefficients, :, j)
        if !all(isfinite, candidate)
            continue
        end
        scale=maximum(abs, candidate; init = 0.0)
        rational=map(candidate) do value
            if abs(value)<=100eps(Float64)*scale
                return zero(Rational{BigInt})
            end
            return rationalize(BigInt, value; tol = 100eps(Float64)*abs(value))
        end
        residual=fill(zero(Rational{BigInt}), size(A, 2))
        rhs=Rational{BigInt}(target[core[dependency]])
        for (k, column) in enumerate(independent)
            coefficient=rational[k]
            if iszero(coefficient)
                continue
            end
            rhs-=coefficient*Rational{BigInt}(target[core[column]])
            for p in SparseArrays.nzrange(transpose_A, column)
                residual[transpose_A.rowval[p]]+=coefficient*Rational{BigInt}(
                    transpose_A.nzval[p],
                )
            end
        end
        for p in SparseArrays.nzrange(transpose_A, dependency)
            residual[transpose_A.rowval[p]]-=Rational{BigInt}(
                transpose_A.nzval[p],
            )
        end
        if all(iszero, residual) && abs(rhs)<=exact_tolerance
            keep[core[dependency]]=false
        end
    end
    return rows[keep]
end
