"Orthogonal, self-inverse conversion between rotated and standard SOC coordinates."
function rotate_soc(vector::AbstractVector{T}) where {T}
    result = collect(vector)
    result[1] = (vector[1]+vector[2])/sqrt(T(2))
    result[2] = (vector[1]-vector[2])/sqrt(T(2))
    return result
end

function append_second_order_constraints!(
    model::Model{T},
    J,
    c,
    dimensions,
) where {T}
    before = num_scalars(model) + sum(model.dim.second_order_dimensions)
    model.jacobian =
        vcat(model.jacobian[1:before, :], J, model.jacobian[(before+1):end, :])
    model.constant =
        vcat(model.constant[1:before], c, model.constant[(before+1):end])
    lower = vcat(
        model.meta.lcon[1:before],
        fill(T(-Inf), length(c)),
        model.meta.lcon[(before+1):end],
    )
    upper = vcat(
        model.meta.ucon[1:before],
        fill(T(Inf), length(c)),
        model.meta.ucon[(before+1):end],
    )
    soc = vcat(model.dim.second_order_dimensions, dimensions)
    model.dim = ConeDimensions(
        num_scalars(model),
        soc,
        model.dim.side_dimensions,
        cumsum([num_scalars(model)+sum(soc); model.dim.side_dimensions .^ 2]),
    )
    model.jac_rows, model.jac_cols, _ = SparseArrays.findnz(model.jacobian)
    m = length(model.constant)
    model.meta = NLPModels.NLPModelMeta{T,Vector{T}}(
        model.meta.nvar;
        x0 = copy(model.meta.x0),
        ncon = m,
        lcon = lower,
        ucon = upper,
        nnzj = length(model.jac_rows),
        nnzh = 0,
        lin_nnzj = length(model.jac_rows),
        nln_nnzj = 0,
        lin = collect(1:m),
    )
    return model
end

function _soc_arrow(vector::AbstractVector{T}) where {T}
    n = length(vector)
    rows, cols, values = Int[], Int[], T[]
    for i in 1:n
        if !iszero(vector[1])
            push!(rows, i)
            push!(cols, i)
            push!(values, vector[1])
        end
        if i > 1 && !iszero(vector[i])
            push!(rows, i)
            push!(cols, 1)
            push!(values, vector[i])
            push!(rows, 1)
            push!(cols, i)
            push!(values, vector[i])
        end
    end
    return SparseArrays.sparse(rows, cols, values, n, n)
end

# Explicit dual solvers can factorize an equivalent PSD arrow. The public
# model retains its native SOC coordinates, independently of this adapter.
function lifted_soc_data(model::Model{T}) where {T}
    if iszero(num_second_order_cones(model))
        return model.C, model.A, model.msizes
    end
    cone = VectorizedConeVector(model.constant, model.dim)
    constants = [_soc_arrow(cone[idx]) for idx in second_order_indices(model)]
    costs = vcat(model.C, constants)
    types = Union{unique(typeof.(costs))...}
    costs = convert(Vector{types}, costs)
    count = length(costs)
    components = map(
        component ->
            BlockColumns(component.block_ids, component.values, count),
        model.A,
    )
    coefficients = Matrix{SparseArrays.SparseMatrixCSC{T,Int}}(
        undef,
        model.meta.nvar,
        length(constants),
    )
    for idx in second_order_indices(model)
        rows = Base.to_index(cone, idx)
        for j in 1:model.meta.nvar
            coefficients[j, idx.value] =
                _soc_arrow(-Vector(model.jacobian[rows, j]))
        end
    end
    blocks = BlockColumns((length(model.C)+1):count, coefficients, count)
    return costs,
    (components..., blocks),
    vcat(model.msizes, model.dim.second_order_dimensions)
end
