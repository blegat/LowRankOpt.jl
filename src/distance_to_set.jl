function MOI.Utilities.distance_to_set(
    d::MOI.Utilities.ProjectionUpperBoundDistance,
    x::AbstractVector{T},
    set::SetDotProducts{WITH_SET},
) where {T,W}
    MOI.Utilities._check_dimension(x, set)
    n = length(set.vectors)
    vec = x[(n+1):end]
    init = MOI.Utilities.distance_to_set(d, vec, set.set)^2
    return √sum(1:n; init) do i
        return (x[i] - MOI.Utilities.set_dot(set.vectors[i], vec, set.set))^2
    end
end

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
