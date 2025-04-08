# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    SetConversionBridge{T,S1,S2,F} <:
        MOI.Bridges.Variable.SetMapBridge{T,S1,S2}

`SetConversionBridge` implements the following reformulations:

  * ``x \\in S2`` into ``x \\in S1``

In order to add this bridge, you need to create a bridge specific
for a given type `T` and set `S1`:
```julia
MOI.Bridges.add_bridge(model, MOI.Bridges.Constraint.SetConversionBridge{T,S1})
```
In order to define a bridge with `S1` specified but `T` unspecified, for example
for `JuMP.add_bridge`, you can use
```julia
const MyBridge{T,S2} = MOI.Bridges.Constraint.SetConversionBridge{T,S1,S2}
```

## Source node

`SetConversionBridge` supports:

  * `F` in `S2`

## Target nodes

`SetConversionBridge` creates:

  * `F` in `S1`
"""
struct SetConversionBridge{T,S1,S2,F} <:
       MOI.Bridges.Variable.SetMapBridge{T,S1,S2}
    variables::Vector{MOI.VariableIndex}
    constraint::MOI.ConstraintIndex{F,S1}
end

function MOI.Bridges.Variable.supports_constrained_variable(
    ::Type{SetConversionBridge{T,S1}},
    ::Type{S2},
) where {T,S1,S2<:MOI.AbstractSet}
    return isfinite(MOI.Bridges.Constraint.conversion_cost(S1, S2))
end

function MOI.Bridges.Variable.concrete_bridge_type(
    ::Type{<:SetConversionBridge{T,S1}},
    ::Type{S2},
) where {T,S1,S2<:MOI.AbstractVectorSet}
    return SetConversionBridge{T,S1,S2,MOI.Utilities.variable_function_type(S1)}
end

function MOI.Bridges.bridging_cost(
    ::Type{<:SetConversionBridge{T,S2,S1}},
) where {T,S2,S1}
    return MOI.Bridges.Constraint.conversion_cost(S2, S1)
end

function MOI.Bridges.map_set(
    ::Type{<:SetConversionBridge{T,S1,S2}},
    set::S1,
) where {T,S1,S2}
    return convert(S2, set)
end

function MOI.Bridges.inverse_map_set(
    ::Type{<:SetConversionBridge{T,S1,S2}},
    set::S2,
) where {T,S1,S2}
    return convert(S1, set)
end

function MOI.Bridges.map_function(::Type{<:SetConversionBridge}, func)
    return func
end

function MOI.Bridges.inverse_map_function(::Type{<:SetConversionBridge}, func)
    return func
end

function MOI.Bridges.adjoint_map_function(::Type{<:SetConversionBridge}, func)
    return func
end

function MOI.Bridges.inverse_adjoint_map_function(
    ::Type{<:SetConversionBridge},
    func,
)
    return func
end
