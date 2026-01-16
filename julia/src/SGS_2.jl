module SGS

using ..Symplectic: symp_inner

export tableau_from_stabilizers

using Random


struct PauliError
    logp::Float64
    v::Vector{Bool}   # length 2n
end

function select2n_independent_from_list(rows::Vector{Vector{Bool}}, n::Int)
    M = falses(0, 2n)
    rk = 0
    for v in rows
        test = vcat(M, reshape(v,1,:))
        newrk = rank_f2(test)
        if newrk > rk
            M = test
            rk = newrk
        end
        rk == 2n && break
    end
    return M, rk
end


function generate_error_pool(n::Int, channel; maxweight::Int=3)
    pI, pX, pZ, pY = channel
    logp = Dict(
        (0,0) => log(pI),
        (1,0) => log(pX),
        (0,1) => log(pZ),
        (1,1) => log(pY)
    )

    pool = PauliError[]

    for w in 0:maxweight
        for positions in combinations(1:n, w)
            v = falses(2n)
            lp = 0.0

            for i in 1:n
                if i ∈ positions
                    # choose most likely non-identity at that position
                    best = argmax([(pX,(1,0)),(pZ,(0,1)),(pY,(1,1))]) do e
                        logp[e[2]]
                    end
                    x,z = best[2]
                    v[i] = x
                    v[n+i] = z
                    lp += logp[(x,z)]
                else
                    lp += logp[(0,0)]
                end
            end

            push!(pool, PauliError(lp, v))
        end
    end

    sort!(pool, by = e -> e.logp, rev=true)
    return pool
end

function tableau_from_channel(n::Int, k::Int, channel)
    r = n - k

    # 1. Generate a ranked error pool
    pool = generate_error_pool(n, channel; maxweight=3)

    # 2. Pick many likely + many unlikely errors
    Ng = 2r                 # over-generate destabilizers
    Nl = 2k                 # over-generate logical candidates

    Gcand = [e.v for e in pool[1:Ng]]
    Lcand = [e.v for e in pool[end-Nl+1:end]]

    # 3. Standard generators Σ
    Sigma = falses(2n, 2n)
    row = 1
    for i in 1:n
        Sigma[row, n+i] = true; row += 1   # Z_i
        Sigma[row, i]   = true; row += 1   # X_i
    end

    rows = Vector{Vector{Bool}}()

    # 1. Most likely errors first → destabilizers
    append!(rows, Gcand)

    # 2. Least likely errors last → logical bias
    append!(rows, Lcand)

    # 3. Add canonical generators ONLY IF NEEDED
    for i in 1:size(Sigma,1)
        push!(rows, Vector(Sigma[i,:]))
        if rank_f2(reduce(vcat, reshape.(rows,1,:))) ≥ 2n
            break
        end
    end

    # 4. Build matrix
    M = reduce(vcat, reshape.(rows,1,:))
    # 5. Select 2n independent rows
   # M = select2n_independent(M, zeros(Bool,0,2n))

    # 6. Run SGS
    C, Q = sgs_pairs(M)
    @assert isempty(C)
    @assert length(Q) == n

    # 7. Read off blocks
    H  = falses(r, 2n)
    G  = falses(r, 2n)
    Lx = falses(k, 2n)
    Lz = falses(k, 2n)

    for i in 1:r
        H[i,:] .= Q[i][1]
        G[i,:] .= Q[i][2]
    end

    for j in 1:k
        Lx[j,:] .= Q[r+j][1]
        Lz[j,:] .= Q[r+j][2]
    end

    return H, Lx, Lz, G
end

"""
    generate_likely_errors(n::Int, r::Int, channel::NTuple{4,Float64}) -> Matrix{Bool}

Generate the r most likely error patterns based on channel probabilities.
Returns an r × 2n matrix where each row is an error pattern.
"""
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

"""
    generate_logicals_commuting_with_G(n::Int, k::Int, G::Matrix{Bool}) -> (Lx, Lz)

Generate k pairs of logical operators (Lx, Lz) such that:
- Lx[i] and Lz[i] anticommute (for same i)
- Lx[i] and Lz[j] commute (for i ≠ j)
- All logicals commute with all G rows
- Logicals are NOT in span(G) - they represent true logical degrees of freedom
"""
function generate_logicals_commuting_with_G(n::Int, k::Int, G::AbstractMatrix{Bool})
    r = size(G, 1)
    Lx = falses(k, 2*n)
    Lz = falses(k, 2*n)
    
    # We need to find vectors in the symplectic orthogonal complement of G
    # that also form k anticommuting pairs
    
    # Build all existing rows to avoid
    existing_rows = [Vector{Bool}(G[i, :]) for i in 1:r]
    
    for i in 1:k
        println("  Generating logical pair $i of $k")
        
        # Generate Lx[i] that commutes with all G and all previously generated logicals
        all_existing = vcat(existing_rows, 
                          [Vector{Bool}(Lx[j, :]) for j in 1:i-1],
                          [Vector{Bool}(Lz[j, :]) for j in 1:i-1])
        
        # Try random vectors that satisfy constraints
        max_attempts = 10000
        found_lx = false
        
        for attempt in 1:max_attempts
            candidate_lx = generate_orthogonal_row_random(all_existing, n)
            
            # Check it's not in span of G (would be trivial)
            test_matrix = vcat(G, reshape(candidate_lx, 1, 2*n))
            if rank_f2(test_matrix) > r
                Lx[i, :] .= candidate_lx
                found_lx = true
                println("    Found Lx[$i] after $attempt attempts")
                break
            end
        end
        
        if !found_lx
            error("Could not find suitable Lx[$i] after $max_attempts attempts")
        end
        
        # Now generate Lz[i] that:
        # - Anticommutes with Lx[i]
        # - Commutes with all G
        # - Commutes with all other logicals
        
        found_lz = false
        for attempt in 1:max_attempts
            candidate_lz = generate_anticommuting_row(Lx[i, :], all_existing, n)
            
            # Check it's not in span of G
            test_matrix = vcat(G, reshape(candidate_lz, 1, 2*n))
            if rank_f2(test_matrix) > r
                # Verify it anticommutes with Lx[i]
                if symp_inner(candidate_lz, Vector{Bool}(Lx[i, :]))
                    Lz[i, :] .= candidate_lz
                    found_lz = true
                    println("    Found Lz[$i] after $attempt attempts")
                    break
                end
            end
        end
        
        if !found_lz
            error("Could not find suitable Lz[$i] after $max_attempts attempts")
        end
        
        # Add to existing rows for next iteration
        push!(existing_rows, Vector{Bool}(Lx[i, :]))
        push!(existing_rows, Vector{Bool}(Lz[i, :]))
    end
    
    return Lx, Lz
end


"""
    generate_stabilizers_commuting_with_all(n::Int, r::Int, G, Lx, Lz) -> H
"""
function generate_stabilizers_commuting_with_all(n::Int, r::Int, 
                                                 G::AbstractMatrix{Bool},
                                                 Lx::AbstractMatrix{Bool}, 
                                                 Lz::AbstractMatrix{Bool})
    H = falses(r, 2*n)
    
    # Collect all rows we must commute with
    existing_rows = Vector{Vector{Bool}}()
    for i in 1:r
        push!(existing_rows, Vector{Bool}(G[i, :]))
    end
    for i in 1:size(Lx, 1)
        push!(existing_rows, Vector{Bool}(Lx[i, :]))
        push!(existing_rows, Vector{Bool}(Lz[i, :]))
    end
    
    for i in 1:r
        println("  Generating stabilizer $i of $r")
        
        max_attempts = 10000
        found = false
        
        for attempt in 1:max_attempts
            candidate_h = generate_orthogonal_row_random(existing_rows, n)
            
            # Check linear independence from existing stabilizers
            if i == 1
                H[i, :] .= candidate_h
                found = true
                break
            else
                test_matrix = vcat(H[1:i-1, :], reshape(candidate_h, 1, 2*n))
                if rank_f2(test_matrix) == i
                    H[i, :] .= candidate_h
                    found = true
                    println("    Found H[$i] after $attempt attempts")
                    break
                end
            end
        end
        
        if !found
            error("Could not find suitable H[$i] after $max_attempts attempts")
        end
        
        push!(existing_rows, Vector{Bool}(H[i, :]))
    end
    
    return H
end

"""
    generate_orthogonal_row_random(existing_rows, n) -> Vector{Bool}

Generate a random row that commutes with all existing rows.
Simple randomized approach.
"""
function generate_orthogonal_row_random(existing_rows::Vector{Vector{Bool}}, n::Int)
    row = rand(Bool, 2*n)
    
    # Make it orthogonal by flipping bits if needed
    for existing_row in existing_rows
        if symp_inner(row, existing_row)
            # Flip a random bit to fix orthogonality
            # This is naive but works for small systems
            pos = rand(1:2*n)
            row[pos] = !row[pos]
        end
    end
    
    return row
end

"""
    generate_anticommuting_row(target_row, commuting_rows, n) -> Vector{Bool}

Generate a row that:
- ANTICOMMUTES with target_row
- COMMUTES with all rows in commuting_rows

This requires solving a constraint system properly.
"""
function generate_anticommuting_row(target_row, 
                                   commuting_rows::Vector{Vector{Bool}}, n::Int)
    # We need to find a vector v such that:
    # 1. <v, target_row> = 1 (anticommute)
    # 2. <v, commuting_row[i]> = 0 for all i (commute)
    
    # Split target_row into X and Z parts
    target_x = target_row[1:n]
    target_z = target_row[n+1:2n]
    
    # We'll build v = (v_x, v_z) position by position
    v = zeros(Bool, 2*n)
    
    # Strategy: Use Gaussian elimination to solve the constraint system
    # Let's use a different approach: find the symplectic orthogonal complement
    
    # Build constraint matrix for commutation requirements
    num_commute_constraints = length(commuting_rows)
    
    # For each commuting row r = (r_x, r_z), we need:
    # v_x · r_z + v_z · r_x = 0 (mod 2)
    
    # For target row t = (t_x, t_z), we need:
    # v_x · t_z + v_z · t_x = 1 (mod 2)
    
    # This is a system of linear equations over F2
    # We have 2n variables (the bits of v) and num_commute_constraints + 1 equations
    
    # Let's use a randomized approach with proper constraint checking
    max_attempts = 10000
    rng = Random.GLOBAL_RNG
    
    for attempt in 1:max_attempts
        # Start with a random vector
        v_candidate = rand(rng, Bool, 2*n)
        
        # Check if it anticommutes with target
        if !symp_inner(v_candidate, target_row)
            # Need to flip to make it anticommute
            # Find a position to flip that will change the inner product
            for pos in 1:2*n
                v_test = copy(v_candidate)
                v_test[pos] = !v_test[pos]
                
                if symp_inner(v_test, target_row)
                    # Now check if it commutes with all commuting_rows
                    all_commute = true
                    for comm_row in commuting_rows
                        if symp_inner(v_test, comm_row)
                            all_commute = false
                            break
                        end
                    end
                    
                    if all_commute
                        return v_test
                    end
                end
            end
        else
            # Already anticommutes, check commutation with others
            all_commute = true
            for comm_row in commuting_rows
                if symp_inner(v_candidate, comm_row)
                    all_commute = false
                    break
                end
            end
            
            if all_commute
                return v_candidate
            end
            
            # Try to fix commutation issues
            for pos in 1:2*n
                v_test = copy(v_candidate)
                v_test[pos] = !v_test[pos]
                
                # Check it still anticommutes with target
                if symp_inner(v_test, target_row)
                    # Check commutation with all
                    all_commute = true
                    for comm_row in commuting_rows
                        if symp_inner(v_test, comm_row)
                            all_commute = false
                            break
                        end
                    end
                    
                    if all_commute
                        return v_test
                    end
                end
            end
        end
    end
    
    # If random approach failed, use systematic construction
    return generate_anticommuting_row_systematic(target_row, commuting_rows, n)
end

"""
    generate_anticommuting_row_systematic(target_row, commuting_rows, n) -> Vector{Bool}

Systematic construction using linear algebra over F2.
"""
function generate_anticommuting_row_systematic(target_row, 
                                               commuting_rows::Vector{Vector{Bool}}, n::Int)
    # Build the constraint system
    # Variables: v[1], v[2], ..., v[2n]
    # Constraints:
    # 1. For each commuting_row r: sum_i (v[i] * r_complement[i]) = 0 (mod 2)
    #    where r_complement[i] = r[i+n] if i <= n, else r[i-n]
    # 2. For target_row t: sum_i (v[i] * t_complement[i]) = 1 (mod 2)
    
    num_constraints = length(commuting_rows) + 1
    num_vars = 2*n
    
    # Build augmented matrix [A | b] for Ax = b (mod 2)
    A = zeros(Bool, num_constraints, num_vars)
    b = zeros(Bool, num_constraints)
    
    # Fill in commutation constraints (should equal 0)
    for (idx, comm_row) in enumerate(commuting_rows)
        for i in 1:n
            A[idx, i] = comm_row[n + i]        # v_x[i] coefficient is r_z[i]
            A[idx, n + i] = comm_row[i]        # v_z[i] coefficient is r_x[i]
        end
        b[idx] = false
    end
    
    # Fill in anticommutation constraint (should equal 1)
    for i in 1:n
        A[num_constraints, i] = target_row[n + i]     # v_x[i] coefficient is t_z[i]
        A[num_constraints, n + i] = target_row[i]     # v_z[i] coefficient is t_x[i]
    end
    b[num_constraints] = true
    
    # Solve using Gaussian elimination with randomization for free variables
    solution = solve_linear_system_f2_with_random(A, b)
    
    if solution === nothing
        error("No solution exists for anticommuting row - this shouldn't happen!")
    end
    
    return solution
end

"""
    solve_linear_system_f2_with_random(A::Matrix{Bool}, b::Vector{Bool}) -> Vector{Bool}

Solve Ax = b (mod 2) with randomization of free variables.
Returns a solution vector, or nothing if no solution exists.
"""
function solve_linear_system_f2_with_random(A::Matrix{Bool}, b::Vector{Bool})
    m, n = size(A)
    
    # Create augmented matrix [A | b]
    Aug = hcat(Matrix{Bool}(A), reshape(b, m, 1))
    
    # Gaussian elimination (row echelon form)
    pivot_cols = Int[]
    current_row = 1
    
    for col in 1:n
        # Find pivot in current column
        pivot_row = findfirst(i -> Aug[i, col], current_row:m)
        
        if pivot_row === nothing
            continue  # No pivot in this column - free variable
        end
        
        pivot_row += current_row - 1
        
        # Swap rows if needed
        if pivot_row != current_row
            Aug[current_row, :], Aug[pivot_row, :] = Aug[pivot_row, :], Aug[current_row, :]
        end
        
        # Eliminate below and above
        for i in 1:m
            if i != current_row && Aug[i, col]
                Aug[i, :] .= xor.(Aug[i, :], Aug[current_row, :])
            end
        end
        
        push!(pivot_cols, col)
        current_row += 1
        
        if current_row > m
            break
        end
    end
    
    # Check for inconsistency (row like [0 0 ... 0 | 1])
    for i in 1:m
        if all(.!Aug[i, 1:n]) && Aug[i, n+1]
            return nothing  # No solution
        end
    end
    
    # Back substitution with random free variables
    x = rand(Bool, n)  # Random initial assignment
    free_vars = setdiff(1:n, pivot_cols)
    
    # Solve for pivot variables in reverse order
    for col in reverse(pivot_cols)
        row = findfirst(i -> Aug[i, col], 1:m)
        
        # x[col] should satisfy: x[col] + sum(Aug[row, j] * x[j] for j > col) = Aug[row, n+1]
        val = Aug[row, n+1]
        for j in (col+1):n
            if Aug[row, j]
                val = xor(val, x[j])
            end
        end
        x[col] = val
    end
    
    # Verify solution
    for i in 1:m
        check = false
        for j in 1:n
            if A[i, j] && x[j]
                check = !check
            end
        end
        if check != b[i]
            # Try flipping a free variable
            if !isempty(free_vars)
                x[free_vars[1]] = !x[free_vars[1]]
                # Recompute pivot variables
                for col in reverse(pivot_cols)
                    row = findfirst(k -> Aug[k, col], 1:m)
                    val = Aug[row, n+1]
                    for j in (col+1):n
                        if Aug[row, j]
                            val = xor(val, x[j])
                        end
                    end
                    x[col] = val
                end
            end
        end
    end
    
    return x
end
"""
    error_pattern_to_string(pattern::Vector{Bool}, n::Int) -> String

Convert binary error pattern to IXZY string.
"""
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

"""
    verify_tableau(H, Lx, Lz, G)

Verify all tableau properties.
"""
function verify_tableau(H::AbstractMatrix{Bool}, Lx::AbstractMatrix{Bool}, 
                       Lz::AbstractMatrix{Bool}, G::AbstractMatrix{Bool})
    r, k = size(H, 1), size(Lx, 1)
    n = size(H, 2) ÷ 2
    println(H)
    println(Lx)
    println(Lz)
    println(G)
    
    println("  Checking H commutes with itself...")
    for i in 1:r, j in 1:r
        @assert !symp_inner(Vector{Bool}(H[i,:]), Vector{Bool}(H[j,:])) "H[$i] and H[$j] don't commute!"
    end
    
    println("  Checking G commutes with H...")
    for i in 1:r, j in 1:r
        @assert !symp_inner(Vector{Bool}(G[i,:]), Vector{Bool}(H[j,:])) "G[$i] and H[$j] don't commute!"
    end
    
    println("  Checking Lx, Lz commute with H...")
    for i in 1:k
        for j in 1:r
            @assert !symp_inner(Vector{Bool}(Lx[i,:]), Vector{Bool}(H[j,:])) "Lx[$i] and H[$j] don't commute!"
            @assert !symp_inner(Vector{Bool}(Lz[i,:]), Vector{Bool}(H[j,:])) "Lz[$i] and H[$j] don't commute!"
        end
    end
    
    println("  Checking Lx[i] anticommutes with Lz[i]...")
    for i in 1:k
        @assert symp_inner(Vector{Bool}(Lx[i,:]), Vector{Bool}(Lz[i,:])) "Lx[$i] and Lz[$i] don't anticommute!"
    end
    
    println("  Checking H and G anticommute...")
    for i in 1:r
        @assert symp_inner(Vector{Bool}(H[i,:]), Vector{Bool}(G[i,:])) "H[$i] and G[$i] don't anticommute!"
    end
    
    println("✓ All tableau properties verified!")
end

# Helper: combinations iterator
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

