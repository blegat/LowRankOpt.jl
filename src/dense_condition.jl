"""
    struct DenseCandidate
        nnz::Int
        side::Int
        remaining::Int
    end

Sparse constraint matrix `Aᵢ` with `nnz` nonzeros of a PSD block of side
`side`. A dense condition, see [`default_dense_condition`](@ref), decides
whether its contribution to the Schur complement is computed by forming the
dense matrix `L Aᵢ R` or from the inner products `⟨Aᵢ, L Aⱼ R⟩` computed from
the nonzeros of `Aᵢ` and `Aⱼ`. The constraint matrices of the block are
visited by decreasing number of nonzeros and `remaining` is the sum of the
number of nonzeros of `Aᵢ` and of the matrices `Aⱼ` visited after `Aᵢ`.
"""
struct DenseCandidate
    nnz::Int
    side::Int
    remaining::Int
end

"""
    struct MinNonzeros
        value::Int
    end

Dense condition that holds if `Aᵢ` has more than `value` nonzeros, see
[`DenseCandidate`](@ref). This is the parameter `κ` of [FKN97].

[FKN97] Fujisawa, Katsuki, Masakazu Kojima, and Kazuhide Nakata. "Exploiting
sparsity in primal-dual interior-point methods for semidefinite programming."
Mathematical Programming 79 (1997): 235-253.
"""
struct MinNonzeros
    value::Int
end

(c::MinNonzeros)(x::DenseCandidate) = x.nnz > c.value

"""
    struct DenseCost
        factor::Float64
    end

Dense condition that holds if `factor * nnz * remaining ≥ side^3`, see
[`DenseCandidate`](@ref). The dense matrix `L Aᵢ R` needs `O(side^3)`
operations while the inner products with the sparse matrices need
`O(nnz * remaining)` operations. The `factor` is the ratio between the time
of an operation of the latter, which accesses `L` and `R` at random
positions, and the time of a flop of the former, which is done by BLAS.
"""
struct DenseCost
    factor::Float64
end

function (c::DenseCost)(x::DenseCandidate)
    return c.factor * float(x.nnz) * x.remaining >= float(x.side)^3
end

"""
    struct And{A,B}
        a::A
        b::B
    end

Dense condition that holds if both `a` and `b` hold.
"""
struct And{A,B}
    a::A
    b::B
end

(c::And)(x::DenseCandidate) = c.a(x) && c.b(x)

"""
    struct Or{A,B}
        a::A
        b::B
    end

Dense condition that holds if `a` or `b` holds.
"""
struct Or{A,B}
    a::A
    b::B
end

(c::Or)(x::DenseCandidate) = c.a(x) || c.b(x)

"""
    struct Not{A}
        a::A
    end

Dense condition that holds if `a` does not hold.
"""
struct Not{A}
    a::A
end

(c::Not)(x::DenseCandidate) = !c.a(x)

"""
    default_dense_condition()

Default dense condition of [`BufferedDualModel`](@ref). A dense condition
is any callable that returns a `Bool` for a [`DenseCandidate`](@ref), for
instance a combination of [`MinNonzeros`](@ref) and [`DenseCost`](@ref) with
[`And`](@ref), [`Or`](@ref) and [`Not`](@ref), or a Julia function.

The factor of [`DenseCost`](@ref) was estimated to be about 170 on
`thetaG51` with a multithreaded BLAS, so this is only a rough estimate.
"""
default_dense_condition() = And(MinNonzeros(8), DenseCost(100))
