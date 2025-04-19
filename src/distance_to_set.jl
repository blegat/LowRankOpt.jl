function MOI.Utilities.distance_to_set(
    d::MOI.Utilities.ProjectionUpperBoundDistance,
    x::AbstractVector{T},
    set::LinearCombinationInSet{W},
) where {T,W}
    MOI.Utilities._check_dimension(x, set)
    if W == WITH_SET
        init = x[(length(set.vectors)+1):end]
    else
        init = zeros(T, MOI.dimension(set.set))
    end
    y = sum(x[i] * set.vectors[i] for i in eachindex(set.vectors); init)
    return MOI.Utilities.distance_to_set(d, y, set.set)
end
