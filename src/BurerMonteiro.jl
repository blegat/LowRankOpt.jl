module BurerMonteiro

import NLPModels
import LowRankOpt as LRO

struct Dimensions
    num_scalars::Int64
    side_dimensions::Vector{Int64}
    ranks::Vector{Int64}
    offsets::Vector{Int64}
end

function Dimensions(model::LRO.Model, ranks)
    side_dimensions = [LRO.side_dimension(model, i) for i in LRO.matrix_indices(model)]
    return Dimensions(
        LRO.num_scalars(model),
        side_dimensions,
        ranks,
        [0; cumsum(side_dimensions .* ranks)],
    )
end

Base.length(d::Dimensions) = d.side_dimensions[end]

struct Model{T} <: NLPModels.AbstractNLPModel{T,Vector{T}}
    model::Model{T}
    d::Dimension
    meta::NLPModels.NLPModelMeta{T,Vector{T}}
    counters::NLPModels.Counters
    function Model(model::Model{T}, ranks) where {T}
        d = Dimensions(model, ranks)
        n = length(d)
        ncon = num_constraints(model)
        return new(
            ad,
            d,
            NLPModels.NLPModelMeta(
                n,     #nvar
                ncon = ncon,
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

struct Solution{T,VT<:AbstractVector{T}}
    x::VT
    d::Dimensions
end

function Base.getindex(s::Solution, ::Type{ScalarIndex})
    return view(s.x, Base.OneTo(s.d.num_scalars))
end

function Base.getindex(s::Solution, mi::MatrixIndex)
    i = mi.value
    U = reshape(
        view(s.x, s.d.offsets[i]:(s.d.offsets[i+1] - 1)),
        s.d.side_dimensions[i],
        s.d.ranks[i],
    )
    return LRO.positive_semidefinite_factorization(U)
end

function NLPModels.obj(model::Model, x::AbstractVector)
    return obj(model.model, Solution(x, model.d))
end

function NLPModels.grad!(model::Model, x::AbstractVector, g::AbstractVector)
    grad!(model.model, Solution(x, model.d), Solution(g, model.d))
    return g
end

end
