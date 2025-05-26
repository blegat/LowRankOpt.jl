# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function MOI.Utilities.distance_to_set(
    d::MOI.Utilities.ProjectionUpperBoundDistance,
    x::AbstractVector{T},
    set::SetDotProducts{WITH_SET},
) where {T}
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
