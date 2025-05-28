import NLPModels

struct BurerMonteiro{T} <: NLPModels.AbstractNLPModel{T,Vector{T}}
    model::Model{T}
    meta::NLPModels.NLPModelMeta{T,Vector{T}}
    counters::NLPModels.Counters
    function BurerMonteiro(model::Model{T}) where {T}
        n = num_scalars(model) + sum(side_dimension(model, i) for i in matrix_indices(model); init = 0)
        ncon = num_constraints(model)
        return new(
            ad,
            NLPModels.NLPModelMeta(
                n,     #nvar
                ncon = ncon,
                nnzj = 0,
                nnzh = 0,
                x0 = rand(n),
                y0 = rand(ncon),
                lvar = fill(-Inf, n),
                uvar = fill(Inf, n),
                lcon = cons_constant(model),
                ucon = cons_constant(model),
                minimize = true,
            ),
            NLPModels.Counters(),
        )
    end
end

function NLPModels.obj(model::BurerMonteiro, x::AbstractVector)
    return obj(model.model, x)
end

function NLPModels.grad!(model::BurerMonteiro, x::AbstractVector, g::AbstractVector)
    grad!(model.model, x, g)
    return g
end
