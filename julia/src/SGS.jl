module SGS

using ..Symplectic: symp_inner

export tableau_from_stabilizers

"""
    tableau_from_stabilizers(S::AbstractMatrix{Bool})

Construct a full stabilizer tableau (H, Lx, Lz, G) from an independent commuting
stabilizer set S ∈ 𝔽₂^{r×2n}, strictly following the SGS procedure:

1) Build an ordered list M with S first, then Σ = (Z₁, X₁, …, Zₙ, Xₙ).
2) Select exactly 2n independent rows (prioritizing S).
3) Run Symplectic Gram–Schmidt (SGS):
   while M nonempty: take a = PopFirst(M);
     - if there exists b ∈ M with ⟨a,b⟩ = 1:
         remove b; for all remaining s ∈ M, do s ← s ⊕ ⟨s,a⟩b ⊕ ⟨s,b⟩a;
         append (a,b) to Q
       else
         append a to C
   end
4) With 2n independent input rows, C = ∅ and |Q| = n.
5) First r pairs of Q span the stabilizer space → H (first of pair), G (second).
   Remaining n−r pairs → Lx (first), Lz (second).

Returns (H, Lx, Lz, G).
"""
function tableau_from_stabilizers(S::AbstractMatrix{Bool})
    r, n2 = size(S)
    @assert iseven(n2) "S must have 2n columns"
    n = n2 >>> 1
    @assert r <= n "Expect r = n - k with k ≥ 0"

    # -----------------------------
    # Step 1: Standard generators Σ
    # Σ = (Z1, X1, ..., Zn, Xn) in binary (u|v).
    # Z_i -> (u=0, v=e_i); X_i -> (u=e_i, v=0)
    # Put them in the Z, X, Z, X order per notes.
    # -----------------------------
    Sigma = falses(2n, 2n)
    row = 1
    for i in 1:n
        # Z_i row
        Sigma[row, n + i] = true
        row += 1
        # X_i row
        Sigma[row, i] = true
        row += 1
    end

    # -----------------------------
    # Step 2: Select exactly 2n independent rows, preferring S first
    # -----------------------------
    M = select2n_independent(S, Sigma)

    # -----------------------------
    # Step 3: SGS on M (list of rows) with full "clean remaining" updates
    # -----------------------------
    C, Q = sgs_pairs(M)

    # With exactly 2n independent input rows, C must be empty and |Q| = n
    @assert isempty(C) "SGS produced commuting leftovers; expected none with 2n-independent input"
    @assert length(Q) == n "SGS should produce exactly n pairs"

    # -----------------------------
    # Step 4: Build H,G,Lx,Lz from pairs; first r pairs correspond (in span) to S
    # -----------------------------
    H  = falses(r, 2n)
    G  = falses(r, 2n)
    Lx = falses(n - r, 2n)
    Lz = falses(n - r, 2n)

    # First r pairs -> stabilizers (H) and destabilizers (G)
    for i in 1:r
        a, b = Q[i]
        H[i, :] .= a
        G[i, :] .= b
    end

    # Remaining n - r pairs -> logicals (Lx, Lz)
    for (j, idx) in enumerate(r+1:n)
        a, b = Q[idx]
        Lx[j, :] .= a
        Lz[j, :] .= b
    end

    return H, Lx, Lz, G
end

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

# Select 2n independent rows from [S; Sigma], in order, preferring S.
# Returns a Matrix{Bool} with exactly 2n rows.
function select2n_independent(S::AbstractMatrix{Bool}, Sigma::AbstractMatrix{Bool})
    n2 = size(S, 2)
    n  = n2 >>> 1
    L  = vcat(S, Sigma)  # candidate pool with stabilizers first
    M  = falses(0, n2)
    rk = 0
    for i in 1:size(L,1)
        cand = vcat(M, @view L[i:i, :])
        newrk = rank_f2(cand)
        if newrk > rk
            M = cand
            rk = newrk
        end
        if size(M,1) == 2n
            break
        end
    end
    @assert size(M,1) == 2n "Select2nIndependent: failed to reach rank 2n"
    return M
end

# Symplectic Gram–Schmidt on a set of 2n independent rows (as a matrix).
# Returns (C, Q) where C is a Vector of commuting rows (should be empty),
# and Q is a Vector of anticommuting pairs (a, b).
function sgs_pairs(M::AbstractMatrix{Bool})
    # Turn rows into a mutable vector of vectors
    rows = [Vector{Bool}(@view M[i, :]) for i in 1:size(M,1)]

    C = Vector{Vector{Bool}}()
    Q = Vector{Tuple{Vector{Bool}, Vector{Bool}}}()

    while !isempty(rows)
        a = popfirst!(rows)  # "PopFirst" from notes
        found = false

        # Seek partner b among remaining rows
        for j in 1:length(rows)
            b = rows[j]
            if symp_inner(a, b)
                # Remove b from the pool
                splice!(rows, j)

                # Clean every remaining s: s ← s ⊕ <s,a>b ⊕ <s,b>a
                for t in 1:length(rows)
                    s = rows[t]
                    ja = symp_inner(s, a)
                    jb = symp_inner(s, b)
                    if jb
                        @inbounds rows[t] .= xor.(rows[t], a)
                    end
                    if ja
                        @inbounds rows[t] .= xor.(rows[t], b)
                    end
                end

                push!(Q, (a, b))
                found = true
                break
            end
        end

        if !found
            # No partner found: add to commuting set C
            push!(C, a)
        end
    end

    return C, Q
end

# --------- GF(2) helpers ---------

# Simple Gaussian elimination over F2 to compute rank.
function rank_f2(A::AbstractMatrix{Bool})
    R = copy(A)
    m, n = size(R)
    r = 0
    for c in 1:n
        # pivot search
        piv = findfirst(i -> R[i, c], r + 1:m)
        if isnothing(piv)
            continue
        end
        piv = r + piv
        # swap to row r+1
        R[r+1, :], R[piv, :] = copy(R[piv, :]), copy(R[r+1, :])
        # eliminate other rows
        for i in 1:m
            if i != r + 1 && R[i, c]
                @inbounds R[i, :] .= xor.(R[i, :], R[r+1, :])
            end
        end
        r += 1
        r == m && break
    end
    return r
end

end # module

