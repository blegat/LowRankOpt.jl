include(joinpath(dirname(@__DIR__), "examples", "maxcut.jl"))

# Important for Dualization
function bench_set_dot(n)
    T = Float64
    set = maxcut_set(n; T)
    d = MOI.dimension(set)
    x = MOI.Utilities.CanonicalVector{T}(div(n, 2), d)
    y = MOI.Utilities.CanonicalVector{T}(2n, d)
    @btime MOI.Utilities.set_dot($x, $x, $set)
    @btime MOI.Utilities.set_dot($y, $y, $set)
end

# Expected:
#   3.437 ns (0 allocations: 0 bytes)
#   6.182 ns (0 allocations: 0 bytes)
bench_set_dot(1000)
