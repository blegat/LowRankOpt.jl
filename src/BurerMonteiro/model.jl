mutable struct Model{S,T,CT,AT,JTB} <: NLPModels.AbstractNLPModel{T,Vector{T}}
    model::LRO.Model{T,CT,AT}
    dim::Dimensions{S}
    meta::NLPModels.NLPModelMeta{T,Vector{T}}
    counters::NLPModels.Counters
    jtprod_buffer::JTB
    function Model{S}(model::LRO.Model{T,CT,AT}, ranks) where {S,T,CT,AT}
        dim = Dimensions{S}(model, ranks)
        jtprod_buffer = buffer_for_jtprod(model, dim)
        return new{S,T,CT,AT,typeof(jtprod_buffer)}(
            model,
            dim,
            meta(dim, LRO.cons_constant(model)),
            NLPModels.Counters(),
            jtprod_buffer,
        )
    end
end

function meta(dim::Dimensions{S}, con::AbstractVector{T}) where {S,T}
    n = length(dim)
    ncon = length(con)
    if S
        lvar = fill(typemin(T), n)
    else
        lvar = [
            fill(zero(T), dim.num_scalars);
            fill(typemin(T), n - dim.num_scalars)
        ]
    end
    return NLPModels.NLPModelMeta(
        n;     #nvar
        ncon,
        x0 = rand(n),
        y0 = rand(ncon),
        lvar,
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
    model.jtprod_buffer = buffer_for_jtprod(model.model, model.dim)
    return
end

#######################
###### Objective ######
#######################

function NLPModels.obj(model::Model, x::AbstractVector)
    return NLPModels.obj(model.model, Solution(x, model.dim))
end

function grad!(model::Model{false}, _, g, ::Type{LRO.ScalarIndex})
    return copyto!(g, LRO.grad(model.model, LRO.ScalarIndex))
end

function grad!(model::Model{true}, x, g, ::Type{LRO.ScalarIndex})
    g .=
        2 .* LRO.grad(model.model, LRO.ScalarIndex) .*
        LRO.left_factor(x, LRO.ScalarIndex)
    return g
end

function grad!(
    model::Model,
    X::LRO.Factorization,
    G::LRO.Factorization,
    i::LRO.MatrixIndex,
)
    C = LRO.grad(model.model, i)
    buffer = _buffer(model.jtprod_buffer[i.value], C, X.factor)
    LRO.buffered_mul!(G.factor, C, X.factor, true, false, buffer)
    G.factor .*= 2
    return
end

function NLPModels.grad!(model::Model, x::AbstractVector, g::AbstractVector)
    X = Solution(x, model.dim)
    G = Solution(g, model.dim)
    grad!(model, X, LRO.left_factor(G, LRO.ScalarIndex), LRO.ScalarIndex)
    for i in LRO.matrix_indices(model.model)
        grad!(model, X[i], G[i], i)
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

#########################
###### Constraints ######
#########################

function NLPModels.cons!(model::Model, x::AbstractVector, cx::AbstractVector)
    X = Solution(x, model.dim)
    # We don't call `cons!` as we don't want to include `-b` since the constraint
    # is encoded as `b <= c(x) <= b` and we just need to specify `c(x)` here.
    # We don't use the version with buffers because that destroys the low-rank structure of `x`
    return NLPModels.jprod!(model.model, X, X, cx)
end

#######################
###### J product ######
#######################

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

########################
###### Jᵀ product ######
########################

function jtprod!(
    model::Model{false},
    _,
    y::AbstractVector,
    JtV::AbstractVector,
    ::Type{LRO.ScalarIndex},
)
    return LinearAlgebra.mul!(JtV, LRO.jac(model.model, LRO.ScalarIndex)', y)
end

function jtprod!(
    model::Model{true},
    X,
    y::AbstractVector,
    JtV::AbstractVector,
    ::Type{LRO.ScalarIndex},
)
    LinearAlgebra.mul!(JtV, LRO.jac(model.model, LRO.ScalarIndex)', y)
    JtV .*= 2 .* LRO.left_factor(X, LRO.ScalarIndex)
    return JtV
end

const _RankOne{T} = LRO.AbstractFactorization{T,<:AbstractVector{T}}
const _LowRank{T} = LRO.AbstractFactorization{T,<:AbstractMatrix{T}}

function buffer_for_jtprod(
    model::LRO.Model{T},
    dim::Dimensions,
    i::LRO.MatrixIndex,
) where {T}
    row = view(model.A, i.value, :)
    C = model.C[i.value]
    if any(A -> A isa _LowRank, row) || C isa _LowRank
        ncols = maximum(row; init = 0) do A
            if A isa _LowRank
                return LRO.max_rank(A)
            else
                return 0
            end
        end
        if C isa _LowRank
            ncols = max(ncols, LRO.max_rank(C))
        end
        return zeros(T, dim.ranks[i.value], ncols)
    elseif any(A -> A isa _RankOne, row) || C isa _RankOne
        return zeros(T, dim.ranks[i.value])
    end
    return
end

function buffer_for_jtprod(model::LRO.Model, dim::Dimensions)
    return buffer_for_jtprod.(model, dim, LRO.matrix_indices(model))
end

_buffer(_, ::AbstractMatrix, _) = nothing
_buffer(buffer::AbstractVector, ::_RankOne, ::AbstractMatrix) = buffer
# TODO check that the size matches, it may not be the highest rank matrix
_buffer(buffer::AbstractMatrix, ::_LowRank, ::AbstractMatrix) = buffer

function add_jtprod!(
    model::Model,
    X::LRO.Factorization,
    y::AbstractVector,
    JtV::LRO.Factorization,
    i::LRO.MatrixIndex,
    α = 2,
)
    for j in eachindex(y)
        A = LRO.jac(model.model, j, i)
        buffer = _buffer(model.jtprod_buffer[i.value], A, X.factor)
        LRO.buffered_mul!(JtV.factor, A, X.factor, α * y[j], true, buffer)
    end
end

function jtprod!(
    model::Model,
    X,
    y::AbstractVector,
    JtV::LRO.Factorization{T},
    i::LRO.MatrixIndex,
) where {T}
    fill!(JtV.factor, zero(T))
    return add_jtprod!(model, X, y, JtV, i)
end

function NLPModels.jtprod!(
    model::Model,
    x::AbstractVector,
    y::AbstractVector,
    Jtv::AbstractVector,
)
    X = Solution(x, model.dim)
    JtV = Solution(Jtv, model.dim)
    jtprod!(model, X, y, LRO.left_factor(JtV, LRO.ScalarIndex), LRO.ScalarIndex)
    for i::LRO.MatrixIndex in LRO.matrix_indices(model.model)
        Xi = X[i]
        JtVi = JtV[i]
        jtprod!(model, Xi, y, JtVi, i)
    end
    return Jtv
end

#######################
###### H product ######
#######################

function NLPModels.hprod!(
    ::Model{false},
    ::AbstractVector,
    y,
    ::AbstractVector,
    Hv::AbstractVector{T},
    ::Type{LRO.ScalarIndex};
    obj_weight,
) where {T}
    return fill!(Hv, zero(T))
end

function NLPModels.hprod!(
    model::Model{true},
    ::AbstractVector,
    y,
    v::AbstractVector,
    Hv::AbstractVector{T},
    ::Type{LRO.ScalarIndex};
    obj_weight,
) where {T}
    Hv .= obj_weight .* LRO.grad(model.model, LRO.ScalarIndex)
    LinearAlgebra.mul!(
        Hv,
        LRO.jac(model.model, LRO.ScalarIndex)',
        y,
        true,
        true,
    )
    Hv .*= -2 .* LRO.left_factor(v, LRO.ScalarIndex)
    return Hv
end

function NLPModels.hprod!(
    model::Model{S,T},
    x::AbstractVector,
    y,
    v::AbstractVector,
    Hv::AbstractVector;
    obj_weight = one(T),
) where {S,T}
    V = Solution(v, model.dim)
    HV = Solution(Hv, model.dim)
    NLPModels.hprod!(
        model,
        x,
        y,
        V,
        LRO.left_factor(HV, LRO.ScalarIndex),
        LRO.ScalarIndex;
        obj_weight,
    )
    for i in LRO.matrix_indices(model.model)
        Vi = V[i].factor
        C = LRO.grad(model.model, i)
        Hvi = HV[i].factor
        LinearAlgebra.mul!(Hvi, C, Vi, 2obj_weight, false)
        for j in 1:model.meta.ncon
            A = LRO.jac(model.model, j, i)
            LinearAlgebra.mul!(Hvi, A, Vi, -2y[j], true)
        end
    end
    return Hv
end
