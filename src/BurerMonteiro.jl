module BurerMonteiro

import LinearAlgebra
import FillArrays
import SolverCore
import NLPModels
import MathOptInterface as MOI
import NLPModelsJuMP
import LowRankOpt as LRO

struct Dimensions
    num_scalars::Int64
    side_dimensions::Vector{Int64}
    ranks::Vector{Int64}
    offsets::Vector{Int64}
end

function Dimensions(model::LRO.Model, ranks)
    side_dimensions =
        [LRO.side_dimension(model, i) for i in LRO.matrix_indices(model)]
    num_scalars = LRO.num_scalars(model)
    offsets = num_scalars .+ [0; cumsum(side_dimensions .* ranks)]
    return Dimensions(num_scalars, side_dimensions, ranks, offsets)
end

Base.length(d::Dimensions) = d.offsets[end]

function set_rank!(d::Dimensions, i::LRO.MatrixIndex, rank)
    d.ranks[i.value] = rank
    for j in (i.value+1):length(d.offsets)
        d.offsets[j] = d.offsets[j-1] + d.side_dimensions[j-1] * d.ranks[j-1]
    end
    return
end

mutable struct Model{T,AT} <: NLPModels.AbstractNLPModel{T,Vector{T}}
    model::LRO.Model{T,AT}
    dim::Dimensions
    meta::NLPModels.NLPModelMeta{T,Vector{T}}
    counters::NLPModels.Counters
    function Model(model::LRO.Model{T,AT}, ranks) where {T,AT}
        dim = Dimensions(model, ranks)
        return new{T,AT}(
            model,
            dim,
            meta(dim, LRO.cons_constant(model)),
            NLPModels.Counters(),
        )
    end
end

function meta(dim, con::AbstractVector{T}) where {T}
    n = length(dim)
    ncon = length(con)
    return NLPModels.NLPModelMeta(
        n;     #nvar
        ncon,
        x0 = rand(n),
        y0 = rand(ncon),
        lvar = [
            fill(zero(T), dim.num_scalars);
            fill(typemin(T), n - dim.num_scalars)
        ],
        uvar = fill(typemax(T), n),
        lcon = con,
        ucon = con,
        minimize = true,
    )
end

function set_rank!(model::Model, i::LRO.MatrixIndex, r)
    set_rank!(model.dim, i, r)
    # `nvar` has changed so we need to reset `model.meta`
    model.meta = meta(model.dim, model.meta.lcon)
    return
end

struct Solution{T,VT<:AbstractVector{T}} <: AbstractVector{T}
    x::VT
    dim::Dimensions
end

struct _OuterProduct{T,UT<:AbstractVector{T},VT<:AbstractVector{T}} <:
       AbstractVector{T}
    x::Solution{T,VT}
    v::Solution{T,UT}
end

Base.eltype(::Type{<:Union{Solution{T},_OuterProduct{T}}}) where {T} = T
Base.eltype(x::Union{Solution,_OuterProduct}) = eltype(typeof(x))

Base.size(s::Solution) = size(s.x)
Base.getindex(s::Solution, i::Integer) = getindex(s.x, i)

Base.size(s::_OuterProduct) = size(s.x)
function Base.show(io::IO, s::_OuterProduct)
    print(io, "_OuterProduct(")
    print(io, s.x)
    print(io, ", ")
    print(io, s.v)
    print(io, ")")
    return
end

function Base.getindex(s::Solution, ::Type{LRO.ScalarIndex})
    return view(s.x, Base.OneTo(s.dim.num_scalars))
end

function Base.getindex(s::_OuterProduct, ::Type{LRO.ScalarIndex})
    return getindex(s.v, LRO.ScalarIndex)
end

function Base.getindex(s::Solution, mi::LRO.MatrixIndex)
    i = mi.value
    U = reshape(
        view(s.x, (1+s.dim.offsets[i]):s.dim.offsets[i+1]),
        s.dim.side_dimensions[i],
        s.dim.ranks[i],
    )
    return LRO.positive_semidefinite_factorization(U)
end

function Base.getindex(s::_OuterProduct{T}, i::LRO.MatrixIndex) where {T}
    U = s.x[i].factor
    V = s.v[i].factor
    return LRO.AsymmetricFactorization(U, V, FillArrays.Fill(T(2), size(U, 2)))
end

function NLPModels.obj(model::Model, x::AbstractVector)
    return NLPModels.obj(model.model, Solution(x, model.dim))
end

function NLPModels.grad!(model::Model, X::LRO.Factorization, G::LRO.Factorization, i::LRO.MatrixIndex)
    C = NLPModels.grad(model.model, i)
    LinearAlgebra.mul!(G.factor, C, X.factor)
    G.factor .*= 2
    return
end

function NLPModels.grad!(model::Model, x::AbstractVector, g::AbstractVector)
    X = Solution(x, model.dim)
    G = Solution(g, model.dim)
    copyto!(G[LRO.ScalarIndex], NLPModels.grad(model.model, LRO.ScalarIndex))
    for i in LRO.matrix_indices(model.model)
        NLPModels.grad!(model, X[i], G[i], i)
    end
    return g
end

# This is used by `SDPLRPlus.jl` in its linesearch.
# It could just take the dot product with the gradient that it already has but
# SDPLRPlus does not treat the objective and constraints differently.
# So since it needs Jacobian-vector product, we also need to implement
# gradient-vector product.
function gprod(model::Model, x::AbstractVector, v::AbstractVector)
    X = Solution(x, model.dim)
    V = Solution(v, model.dim)
    return NLPModels.obj(model.model, _OuterProduct(X, V))
end

function NLPModels.cons!(model::Model, x::AbstractVector, cx::AbstractVector)
    X = Solution(x, model.dim)
    # We don't call `cons!` as we don't want to include `-b` since the constraint
    # is encoded as `b <= c(x) <= b` and we just need to specify `c(x)` here.
    # We don't use the version with buffers because that destroys the low-rank structure of `x`
    return NLPModels.jprod!(model.model, X, X, cx)
end

function NLPModels.jprod!(
    model::Model,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    X = Solution(x, model.dim)
    V = Solution(v, model.dim)
    # The second argument is ignored as it is linear so it does
    # not matter that we give `x`
    return NLPModels.jprod!(model.model, X, _OuterProduct(X, V), Jv)
end

function add_jtprod!(
    model::Model,
    X::LRO.Factorization,
    y::AbstractVector,
    JtV::LRO.Factorization,
    i::LRO.MatrixIndex,
)
    for j in eachindex(y)
        A = NLPModels.jac(model.model, j, i)
        JtV.factor .+= A * X.factor .* (2y[j])
    end
end

function NLPModels.jtprod!(
    model::Model,
    X,
    y::AbstractVector,
    JtV::LRO.Factorization{T},
    i::LRO.MatrixIndex,
) where {T}
    fill!(JtV.factor, zero(T))
    add_jtprod!(model, X, y, JtV, i)
end

function NLPModels.jtprod!(
    model::Model,
    x::AbstractVector,
    y::AbstractVector,
    Jtv::AbstractVector,
)
    X = Solution(x, model.dim)
    JtV = Solution(Jtv, model.dim)
    LinearAlgebra.mul!(
        JtV[LRO.ScalarIndex],
        NLPModels.jac(model.model, LRO.ScalarIndex)',
        y,
    )
    for i in LRO.matrix_indices(model.model)
        NLPModels.jtprod!(model, X[i], y, JtV[i], i)
    end
    return Jtv
end

function NLPModels.hprod!(
    model::Model{T},
    ::AbstractVector,
    y,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight = one(T),
) where {T}
    V = Solution(v, model.dim)
    HV = Solution(Hv, model.dim)
    fill!(Hv, zero(eltype(Hv)))
    for i in LRO.matrix_indices(model.model)
        Vi = V[i].factor
        C = NLPModels.grad(model.model, i)
        Hvi = HV[i].factor
        Hvi .+= C * Vi
        Hvi .*= 2obj_weight
        for j in 1:model.meta.ncon
            A = NLPModels.jac(model.model, j, i)
            Hvi .-= A * Vi .* (2y[j])
        end
    end
    return Hv
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
    model::NLPModels.AbstractNLPModel; # Same as `solver.model.model`
    kws...,
)
    return SolverCore.solve!(solver.solver, solver.model, solver.stats; kws...)
end

function MOI.get(solver::Solver, attr::MOI.SolverName)
    return "BurerMonteiro with " * MOI.get(solver.solver, attr)
end

function MOI.get(solver::Solver, ::LRO.ConvexTerminationStatus)
    return NLPModelsJuMP.TERMINATION_STATUS[solver.stats.status]
    # TODO if the dual is feasible, we can still claim that we found the optimal
    #      and turn `LOCALLY_SOLVED` into `OPTIMAL`
end

function MOI.get(solver::Solver, ::LRO.Solution)
    return Solution(solver.stats.solution, solver.model.dim)
end

end
