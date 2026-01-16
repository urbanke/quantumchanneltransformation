module SGS

using ..Symplectic: symp_inner

export tableau_from_stabilizers


function generate_likely_errors(n::Int, r::Int, channel::NTuple{4,Float64})
    pI, pX, pZ, pY = channel
    
    # Create all possible single-qubit errors with their probabilities
    # I=(0,0), X=(1,0), Z=(0,1), Y=(1,1)
    single_errors = [
        (prob=pI, x=0, z=0),
        (prob=pX, x=1, z=0),
        (prob=pZ, x=0, z=1),
        (prob=pY, x=1, z=1)
    ]
    sort!(single_errors, by=e->e.prob, rev=true)
    
    # Generate all n-qubit error patterns and compute probabilities
    # We'll enumerate smartly rather than all 4^n possibilities
    
    error_patterns = Vector{Tuple{Float64, Vector{Bool}}}()
    
    # Strategy: enumerate by Hamming weight (number of non-identity errors)
    for weight in 0:n
        if length(error_patterns) >= 10*r  # Stop when we have enough candidates
            break
        end
        
        # Generate all patterns with this weight
        for positions in combinations(1:n, weight)
            # For these positions, try different error types
            # We'll be greedy: use highest probability errors
            pattern = zeros(Bool, 2*n)
            prob = 1.0
            
            for pos in positions
                # Use the most likely non-identity error
                best_error = single_errors[2]  # Start with second (first is I)
                pattern[pos] = best_error.x
                pattern[n + pos] = best_error.z
                prob *= best_error.prob
            end
            
            # Remaining positions get identity (probability pI each)
            prob *= pI^(n - weight)
            
            push!(error_patterns, (prob, pattern))
        end
    end
    
    # Sort by probability and take top r
    sort!(error_patterns, by=e->e[1], rev=true)
    
    # Convert to matrix, ensuring linear independence
    G = falses(0, 2*n)
    candidates = error_patterns
    
    for (prob, pattern) in candidates
        if size(G, 1) >= r
            break
        end
        
        # Check if adding this pattern maintains linear independence
        candidate_G = vcat(G, reshape(pattern, 1, 2*n))
        if rank_f2(candidate_G) > size(G, 1)
            G = candidate_G
            println("  Added error with prob=$prob: $(error_pattern_to_string(pattern, n))")
        end
    end
    
    if size(G, 1) < r
        error("Could not find $r linearly independent likely errors")
    end
    
    return G
end

function error_pattern_to_string(pattern::Vector{Bool}, n::Int)
    paulis = ['I', 'X', 'Z', 'Y']
    result = Char[]
    for i in 1:n
        x = pattern[i]
        z = pattern[n + i]
        idx = 1 + x + 2*z
        push!(result, paulis[idx])
    end
    return String(result)
end

function combinations(items, k)
    n = length(items)
    if k > n || k < 0
        return []
    end
    if k == 0
        return [[]]
    end
    result = []
    function backtrack(start, current)
        if length(current) == k
            push!(result, copy(current))
            return
        end
        for i in start:n
            push!(current, items[i])
            backtrack(i + 1, current)
            pop!(current)
        end
    end
    backtrack(1, [])
    return result
end




function tableau_from_channel(n, r, pchannel)
    S = generate_likely_errors(n, r, pchannel)

    k = (n - r) # since we have lx and lz 
    # -----------------------------
    # Step 1: Standard generators Σ
    # Σ = (Z1, X1, ..., Zn, Xn) in binary (u|v).
    # Z_i -> (u=0, v=e_i); X_i -> (u=e_i, v=0)
    # Put them in the Z, X, Z, X order per notes.
    # -----------------------------
    # for now i am assuming px > pz > py for proof of concept 
    #=
    Sigma = falses(2n, 2n)
    for i in 1:r
        # Z_i row
        Sigma[i, n + i] = true
    end
    
    for i in 0:k
        # Z_i row
        Sigma[i+r+1, 1:2n] .= true
        Sigma[i+r+1, 2n-i] = false
    end
    =# 
    Sigma = trues(2n, 2n) 
    row = 1
    for i in 1:n
        # Z_i row
        Sigma[row, n + i] = false
        row += 1
        # X_i row
        Sigma[row, i] = false
        row += 1
    end
    # -----------------------------
    # Step 2: Select exactly 2n independent rows, preferring S first
    # -----------------------------
    M = select2n_independent(S, Sigma)

    # -----------------------------
    # Step 3: SGS on M (list of rows) with full "clean remaining" updates
    # -----------------------------
    C, Q = sgs_pairs(Sigma)

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

    # First k pairs -> stabilizers (H) and destabilizers (G)
    for i in 1:k
        a, b = Q[i]
        Lx[i, :] .= a
        Lz[i, :] .= b
    end

    # Remaining n - r pairs -> logicals (Lx, Lz)
    for (j, idx) in enumerate(k+1:n)
        a, b = Q[idx]
        H[j, :] .= a
        G[j, :] .= b
    end

    return G, Lx, Lz, H, M
end



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
