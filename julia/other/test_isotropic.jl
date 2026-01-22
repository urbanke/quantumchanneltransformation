include("../env_utils/Isotropic.jl")

using Random, .Isotropic


# -----------------------------
# Symplectic dot product over GF(2)
# <u,v> = u_x·v_z + u_z·v_x  (mod 2)
# u and v are BitVectors of length 2n
# -----------------------------
function symplectic_dot(u::BitVector, v::BitVector, n::Int)::Bool
    acc = false
    @inbounds for i in 1:n
        # u_x[i] & v_z[i]
        if u[i] && v[n+i]
            acc = !acc
        end
        # u_z[i] & v_x[i]
        if u[n+i] && v[i]
            acc = !acc
        end
    end
    return acc  # true means 1 mod 2 (anti-commute)
end

# -----------------------------
# Check isotropy: all pairs commute
# -----------------------------
function assert_isotropic(S::BitMatrix, n::Int)
    s = size(S, 1)
    for i in 1:s, j in i+1:s
        u = BitVector(S[i, :])
        v = BitVector(S[j, :])
        @assert !symplectic_dot(u, v, n) "Non-isotropic: rows $i and $j anticommute"
    end
    return true
end


"""
Return the index of the first `true` bit in v, or 0 if all false.
(We use the leftmost pivot convention.)
"""
function first_one(v::BitVector)
    for i in 1:length(v)
        if v[i]
            return i
        end
    end
    return 0
end


# -----------------------------
# GF(2) rank of a BitMatrix (row-rank)
# Gaussian elimination over GF(2)
# -----------------------------
function gf2_rank(M::BitMatrix)::Int
    A = copy(M)
    m, ncols = size(A)
    rank = 0
    col = 1

    @inbounds while rank < m && col <= ncols
        # find pivot row at or below rank+1 with A[row, col] = 1
        pivot = 0
        for r in (rank+1):m
            if A[r, col]
                pivot = r
                break
            end
        end

        if pivot == 0
            col += 1
            continue
        end

        # swap pivot row into position rank+1
        if pivot != rank+1
            tmp = copy(view(A, rank+1, :))
            A[rank+1, :] .= view(A, pivot, :)
            A[pivot, :] .= tmp
        end

        # eliminate this column from all other rows
        for r in 1:m
            if r != rank+1 && A[r, col]
                A[r, :] .⊻= view(A, rank+1, :)
            end
        end

        rank += 1
        col += 1
    end

    return rank
end

# -----------------------------
# Verify your intended block-structure
# - first r rows: X has identity at (i,i); X[1:r] has only that 1
# - remaining rows (r+1:s): X-part is all zeros (pure-Z)
# -----------------------------
function assert_structure(S::BitMatrix, n::Int, s::Int, r::Int)
    @assert size(S, 1) == s
    @assert size(S, 2) == 2n

    # first r rows: identity in X[1:r]
    for i in 1:r
        @assert S[i, i] "Row $i should have identity 1 at X position $i"
        for j in 1:r
            if j != i
                @assert !S[i, j] "Row $i should have 0 at X position $j (within I block)"
            end
        end
    end

    # remaining rows: pure-Z (X part = 0)
    for i in (r+1):s
        xcount = count(S[i, 1:n])
        @assert xcount == 0 "Row $i should be pure-Z (X-part all zero), but has $xcount ones"
    end

    return true
end

# -----------------------------
# One test instance
# -----------------------------
function test_one(n::Int, k::Int, r::Int; rng = Random.default_rng())
    s = n - k
    @assert 0 <= k <= n
    @assert 1 <= s <= n  "Need s=n-k >= 1 for a nontrivial stabilizer"
    @assert 0 <= r <= s

    S = random_isotropic_basis_with_structure(n, s, r; rng=rng)

    # checks
    assert_isotropic(S, n)
    @assert gf2_rank(S) == s "Not full rank: rank=$(gf2_rank(S)) but expected s=$s"
    assert_structure(S, n, s, r)

    return true
end


@inline function symplectic_dot(u::BitVector, v::BitVector, n::Int)
    parity = false
    @inbounds for i in 1:n
        # add mod 2: (x_i & z'_i) + (z_i & x'_i)
        parity ⊻= (u[i] & v[n+i]) ⊻ (u[n+i] & v[i])
    end
    return parity
end

function assert_isotropic(S::BitMatrix, n::Int)
    m = size(S, 1)
    for i in 1:m, j in i+1:m
        u = BitVector(S[i, :])
        v = BitVector(S[j, :])
        @assert !symplectic_dot(u, v, n) "Rows $i and $j anticommute!"
    end
    return true
end

function run_tests(; trials::Int=500, seed::Int=0)
    rng = MersenneTwister(seed)

    # Some fixed edge-ish cases
    fixed = [
        (4, 1, 0), (4, 1, 1), (4, 2, 2),
        (6, 1, 3), (6, 2, 2), (8, 3, 1),
        (10, 5, 0), (10, 3, 3), (9,4,2), (8,5,0)
    ]

    println("Running fixed tests...")
    for (n,k,r) in fixed
        test_one(n,k,r; rng=rng)
        println("  OK: (n,k,r)=($n,$k,$r)  s=$(n-k)")
    end

    println("\nRunning $trials random tests...")
    for t in 1:trials
        n = rand(rng, 2:20)
        k = rand(rng, 1:(n-1))      # ensure s=n-k >= 1
        s = n - k
        r = rand(rng, 0:s)
        test_one(n,k,r; rng=rng)
        println("  OK[$t]: (n,k,r)=($n,$k,$r)  s=$s")
    end

    println("\nAll tests passed ✅")
end

# Run if executed as a script
run_tests(trials=5000, seed=1234)
