import LowRankOpt as LRO

# Read the positive coefficient independently from the native Jacobian.
function _coefficient(model::LRO.Model, j, i::LRO.MatrixIndex)
    rows = (model.dim.offsets[i.value]+1):model.dim.offsets[i.value+1]
    d = model.msizes[i.value]
    return -reshape(model.jacobian[rows, j], d, d)
end

function _flat_cone(v, model)
    return [
        Vector(v[LRO.ScalarIndex]);
        reduce(
            vcat,
            (vec(Matrix(v[i])) for i in LRO.matrix_indices(model));
            init = eltype(model.b)[],
        )
    ]
end
