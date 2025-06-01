module BurerMonteiro

import LinearAlgebra
import FillArrays
import SolverCore
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
    num_scalars = LRO.num_scalars(model)
    offsets = num_scalars .+ [0; cumsum(side_dimensions .* ranks)]
    return Dimensions(num_scalars, side_dimensions, ranks, offsets)
end

Base.length(d::Dimensions) = d.offsets[end]

struct Model{T,AT} <: NLPModels.AbstractNLPModel{T,Vector{T}}
    model::LRO.Model{T,AT}
    dim::Dimensions
    meta::NLPModels.NLPModelMeta{T,Vector{T}}
    counters::NLPModels.Counters
    function Model(model::LRO.Model{T,AT}, ranks) where {T,AT}
        dim = Dimensions(model, ranks)
        n = length(dim)
        ncon = LRO.num_constraints(model)
        return new{T,AT}(
            model,
            dim,
            NLPModels.NLPModelMeta(
                n,     #nvar
                ncon = ncon,
                x0 = rand(n),
                y0 = rand(ncon),
                lvar = fill(-Inf, n),
                uvar = fill(Inf, n),
                lcon = LRO.cons_constant(model),
                ucon = LRO.cons_constant(model),
                minimize = true,
            ),
            NLPModels.Counters(),
        )
    end
end

struct Solution{T,VT<:AbstractVector{T}}
    x::VT
    dim::Dimensions
end

struct _OuterProduct{T,UT<:AbstractVector{T},VT<:AbstractVector{T}}
    x::Solution{T,VT}
    v::Solution{T,UT}
end

Base.eltype(::Type{<:Union{Solution{T},_OuterProduct{T}}}) where {T} = T
Base.eltype(x::Union{Solution,_OuterProduct}) = eltype(typeof(x))

function Base.getindex(s::Solution, ::Type{LRO.ScalarIndex})
    return view(s.x, Base.OneTo(s.dim.num_scalars))
end

function Base.getindex(s::_OuterProduct, ::Type{LRO.ScalarIndex})
    return getindex(s.v, LRO.ScalarIndex)
end

function Base.getindex(s::Solution, mi::LRO.MatrixIndex)
    i = mi.value
    U = reshape(
        view(s.x, (1 + s.dim.offsets[i]):s.dim.offsets[i+1]),
        s.dim.side_dimensions[i],
        s.dim.ranks[i],
    )
    return LRO.positive_semidefinite_factorization(U)
end

function Base.getindex(s::_OuterProduct{T}, i::LRO.MatrixIndex) where {T}
    U = s.x[i]
    V = s.v[i]
    return LRO.AsymmetricFactorization(U, V, FillArrays.Fill(T(2), size(U, 2)))
end

function NLPModels.obj(model::Model, x::AbstractVector)
    return NLPModels.obj(model.model, Solution(x, model.dim))
end

function NLPModels.grad!(model::Model, x::AbstractVector, g::AbstractVector)
    X = Solution(x, model.dim)
    G = Solution(g, model.dim)
    copyto!(G[LRO.ScalarIndex], NLPModels.grad(model.model, LRO.ScalarIndex))
    for i in LRO.matrix_indices(model.model)
        C = NLPModels.grad(model.model, i)
        LinearAlgebra.mul!(G[i].factor, C, X[i].factor)
        G[i].factor ./= 2
    end
    return g
end

function NLPModels.cons!(model::Model, x::AbstractVector, cx::AbstractVector)
    NLPModels.cons!(model.model, Solution(x, model.dim), cx)
end

function NLPModels.jprod!(model::Model, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
    X = Solution(x, model.dim)
    V = Solution(v, model.dim)
    # The second argument is ignored as it is linear so it does
    # not matter that we give `x`
    NLPModels.jprod!(model.model, X, _OuterProduct(X, V), Jv)
end

function NLPModels.jtprod!(model::Model, x::AbstractVector, y::AbstractVector, Jtv::AbstractVector)
    X = Solution(x, model.dim)
    JtV = Solution(Jtv, model.dim)
    LinearAlgebra.mul!(
        JtV[LRO.ScalarIndex],
        NLPModels.jac(model.model, LRO.ScalarIndex)',
        y,
    )
    for i in LRO.matrix_indices(model.model)
        U = JtV[i].factor
        for j in eachindex(y)
            A = NLPModels.jac(model.model, LRO.ConstraintIndex(j), i)
            LinearAlgebra.mul!(U, A, X[i].factor)
            U .*= 2
        end
    end
    return Jtv
end

struct Solver{T,ST} <: SolverCore.AbstractOptimizationSolver
    model::Model{T}
    solver::ST
    stats::SolverCore.GenericExecutionStats{T,Vector{T},Vector{T},Any}
end

function Solver(src::LRO.Model; sub_solver, ranks, kws...)
    model = Model(src, ranks)
    solver = sub_solver(model; kws...)
    stats = SolverCore.GenericExecutionStats(model)
    return Solver(model, solver, stats)
end

function SolverCore.solve!(
    solver::Solver,
    model::NLPModels.AbstractNLPModel, # Same as `solver.model.model`
    stats::SolverCore.GenericExecutionStats;
    kws...,
)
    SolverCore.solve!(solver.solver, solver.model, solver.stats; kws...)
end


end
