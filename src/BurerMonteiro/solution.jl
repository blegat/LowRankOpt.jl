# `Dimensions{false}` means that nonnegative scalars have a zero lower bound
# `Dimensions{true}` means that nonnegative scalars are the square of a free variable
struct Dimensions{S}
    num_scalars::Int64
    side_dimensions::Vector{Int64}
    ranks::Vector{Int64}
    offsets::Vector{Int64}
end

function Dimensions{S}(model::LRO.Model, ranks) where {S}
    side_dimensions =
        [LRO.side_dimension(model, i) for i in LRO.matrix_indices(model)]
    num_scalars = LRO.num_scalars(model)
    offsets = num_scalars .+ [0; cumsum(side_dimensions .* ranks)]
    return Dimensions{S}(num_scalars, side_dimensions, ranks, offsets)
end

Base.broadcastable(d::Dimensions) = Ref(d)

Base.length(d::Dimensions) = d.offsets[end]

function set_rank!(d::Dimensions, i::LRO.MatrixIndex, rank)
    d.ranks[i.value] = rank
    for j in (i.value+1):length(d.offsets)
        d.offsets[j] = d.offsets[j-1] + d.side_dimensions[j-1] * d.ranks[j-1]
    end
    return
end

struct Solution{S,T,VT<:AbstractVector{T}} <: AbstractVector{T}
    x::VT
    dim::Dimensions{S}
end

struct _OuterProduct{S,T,UT<:AbstractVector{T},VT<:AbstractVector{T}} <:
       AbstractVector{T}
    x::Solution{S,T,VT}
    v::Solution{S,T,UT}
end

Base.eltype(::Type{<:Union{Solution{S,T},_OuterProduct{S,T}}}) where {S,T} = T
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

function LRO.left_factor(s::Solution, ::Type{LRO.ScalarIndex})
    return view(s.x, Base.OneTo(s.dim.num_scalars))
end

function Base.getindex(s::Solution{false}, ::Type{LRO.ScalarIndex})
    return LRO.left_factor(s::Solution, LRO.ScalarIndex)
end

function Base.getindex(s::Solution{true,T}, ::Type{LRO.ScalarIndex}) where {T}
    s = LRO.left_factor(s::Solution, LRO.ScalarIndex)
    return MOI.Utilities.VectorLazyMap{T}(abs2, s)
end

function Base.getindex(s::_OuterProduct{false}, ::Type{LRO.ScalarIndex})
    return getindex(s.v, LRO.ScalarIndex)
end

function Base.getindex(s::_OuterProduct{true}, ::Type{LRO.ScalarIndex})
    # TODO Lazy
    return 2 .* LRO.left_factor(s.x, LRO.ScalarIndex) .*
           LRO.left_factor(s.v, LRO.ScalarIndex)
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

function Base.getindex(s::_OuterProduct{S,T}, i::LRO.MatrixIndex) where {S,T}
    U = s.x[i].factor
    V = s.v[i].factor
    return LRO.AsymmetricFactorization(U, V, FillArrays.Fill(T(2), size(U, 2)))
end
