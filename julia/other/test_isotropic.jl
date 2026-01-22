using Random

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

"""
Reduce v in-place using an echelon basis over GF(2).
basis maps pivot_index -> pivot_row (BitVector).
"""
function gf2_reduce!(v::BitVector, basis::Dict{Int, BitVector})
    while true
        p = first_one(v)
        p == 0 && return v
        if haskey(basis, p)
            v .⊻= basis[p]   # XOR eliminate pivot
        else
            return v
        end
    end
end

function gf2_try_add!(basis::Dict{Int, BitVector}, v::BitVector)
    w = copy(v)
    gf2_reduce!(w, basis)
    p = first_one(w)
    p == 0 && return false
    basis[p] = w
    return true
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

# -----------------------------
# Batch runner: random + some edge cases
# -----------------------------
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


# Utilities
identity_matrix(m::Int) = begin
    M = zeros(Int, m, m)
    for i in 1:m
        M[i,i] = 1
    end
    M
end

zero_matrix(m::Int, n::Int) = zeros(Int, m, n)


#rand_binary_matrix(m::Int, n::Int, rng) =  Int.(rand(rng, Bool, m, n))  # 0/1 as Int

rand_binary_matrix(m::Int, n::Int, rng; p::Float64 = 0.2) = Int.(rand(rng, m, n) .< p)

# Iterate all binary matrices of size m×n (2^(m*n) total)
function each_binary_matrix(m::Int, n::Int)
    Channel{Matrix{Int}}() do ch
        # Edge cases
        if m == 0 || n == 0
            put!(ch, zeros(Int, m, n))
            return
        end
        buf = Array{Int}(undef, m, n)
        function fillpos(p::Int)
            if p > m*n
                put!(ch, copy(buf))
                return
            end
            j = ((p-1) % n) + 1
            i = ((p-1) ÷ n) + 1
            buf[i, j] = 0; fillpos(p+1)
            buf[i, j] = 1; fillpos(p+1)
        end
        fillpos(1)
    end
end



@inline function symplectic_dot(u::BitVector, v::BitVector, n::Int)
    parity = false
    @inbounds for i in 1:n
        # add mod 2: (x_i & z'_i) + (z_i & x'_i)
        parity ⊻= (u[i] & v[n+i]) ⊻ (u[n+i] & v[i])
    end
    return parity
end

# ---- GF(2) row operations on BitMatrix (safe, no slicing pitfalls) ----

@inline function xor_row!(A::BitMatrix, dst::Int, src::Int)
    n = size(A, 2)
    @inbounds for j in 1:n
        A[dst, j] ⊻= A[src, j]
    end
end

@inline function swap_rows!(A::BitMatrix, r1::Int, r2::Int)
    r1 == r2 && return
    n = size(A, 2)
    @inbounds for j in 1:n
        A[r1, j], A[r2, j] = A[r2, j], A[r1, j]
    end
end

"""
    gf2_nullspace(H::BitMatrix) -> BitMatrix

Returns a matrix N whose rows form a basis for the nullspace of H over GF(2),
i.e. H * N' = 0 (mod 2).
"""
function gf2_nullspace(H::BitMatrix)
    m, n = size(H)
    A = copy(H)

    pivot_cols = Int[]
    pivot_rows = Int[]
    row = 1

    # Reduced row echelon form over GF(2)
    for col in 1:n
        row > m && break

        # find pivot row at/below `row` with a 1 in this column
        pivot = 0
        @inbounds for r in row:m
            if A[r, col]
                pivot = r
                break
            end
        end
        pivot == 0 && continue

        swap_rows!(A, row, pivot)

        push!(pivot_cols, col)
        push!(pivot_rows, row)

        # eliminate from all other rows
        @inbounds for r in 1:m
            if r != row && A[r, col]
                xor_row!(A, r, row)
            end
        end

        row += 1
    end



    free_cols = setdiff(1:n, pivot_cols)
    N = BitMatrix(undef, length(free_cols), n)

    # Build nullspace basis vectors
    for (i, fc) in enumerate(free_cols)
        v = falses(n)
        v[fc] = true
        # pivot var = sum(A[pivot_row, free_col]*free_var)
        for (pr, pc) in zip(pivot_rows, pivot_cols)
            if A[pr, fc]
                v[pc] = true
            end
        end
        N[i, :] = v
    end

    return N
end


function concat_stabilizers_bool(Sout::AbstractMatrix{Bool}, Sin::AbstractMatrix{Bool}; 
                                 ordering::Symbol=:interleaved)
    r1, n1_times_2 = size(Sout)
    r2, n2_times_2 = size(Sin)
    
    @assert iseven(n1_times_2) && iseven(n2_times_2) "Stabilizer matrices must have even number of columns (2n)"
    n1 = n1_times_2 ÷ 2
    n2 = n2_times_2 ÷ 2
    
    N = n1 * n2  # Total number of qubits in concatenated code
    
    # Result will have r2 rows (from inner) + r1*n2 rows (from outer)
    num_rows = r2 + r1 * n2
    result = falses(num_rows, 2*N)
    
    # Index mapping: (block_index, inner_position) -> concatenated_position
    idx = if ordering == :interleaved
        (b, j) -> (j - 1) * n1 + b
    elseif ordering == :block
        (b, j) -> (b - 1) * n2 + j
    else
        error("Unsupported ordering: $ordering (use :interleaved or :block)")
    end
    
    current_row = 1
    
    # --- Part 1: Inner stabilizers (applied to ALL blocks simultaneously) ---
    for row_in in 1:r2
        # For each qubit j in the inner code, apply its Pauli to all blocks at position j
        for j in 1:n2
            # Get X and Z for qubit j in this inner stabilizer
            x_j = Sin[row_in, j]
            z_j = Sin[row_in, n2 + j]
            
            # If not identity, apply to all blocks
            if x_j || z_j
                for b in 1:n1
                    pos = idx(b, j)
                    result[current_row, pos] = x_j        # X part
                    result[current_row, N + pos] = z_j    # Z part
                end
            end
        end
        current_row += 1
    end
    
    # --- Part 2: Outer stabilizers (one copy per inner position) ---
    # For each outer stabilizer, create n2 copies (one for each inner position j)
    for row_out in 1:r1
        for j in 1:n2
            # This outer stabilizer acts on the n1 blocks at inner position j
            for b in 1:n1
                # Get X and Z for block b in this outer stabilizer
                x_b = Sout[row_out, b]
                z_b = Sout[row_out, n1 + b]
                
                # Map to concatenated position
                pos = idx(b, j)
                result[current_row, pos] = x_b        # X part
                result[current_row, N + pos] = z_b    # Z part
            end
            current_row += 1
        end
    end
    
    return result
end

function random_isotropic_basis_with_structure(n::Int, s::Int, r::Int; rng = Random.default_rng())
    @assert s ≤ n "Maximum isotropic dimension is n"
    @assert r ≤ s "Must have r ≤ s"
    @assert r ≤ n "Must have r ≤ n (need r qubits for identity)"

    rows  = BitVector[]
    basis = Dict{Int, BitVector}()  # GF(2) echelon basis for independence checks

    # -------------------------------
    # First r rows: enforce I in X-part
    # -------------------------------
    for i in 1:r
        success = false
        attempts = 0

        while !success
            attempts += 1
            attempts > 2000 && error("Could not generate independent structured row $i after many attempts")

            v = falses(2n)

            # X-part: identity bit + random A bits
            v[i] = true
            for j in (r+1):n
                v[j] = rand(rng, Bool)
            end

            # Choose Z-part to satisfy isotropy with existing rows
            if !isempty(rows)
                H_z = BitMatrix(undef, length(rows), n)
                b_z = falses(length(rows))

                for (eq, u) in enumerate(rows)
                    # coefficients are u_x
                    H_z[eq, :] .= view(u, 1:n)

                    # RHS = v_x · u_z
                    rhs = false
                    for k in 1:n
                        if v[k] && u[n + k]
                            rhs = !rhs
                        end
                    end
                    b_z[eq] = rhs
                end

                v_z = solve_linear_system_f2_with_random_gf2(H_z, b_z, rng)
                v_z === nothing && continue  # resample A and try again

                v[n+1:2n] .= v_z
            else
                # first row: free Z-part
                for j in (n+1):2n
                    v[j] = rand(rng, Bool)
                end
            end

            # Independence check (full 2n vector)
            if gf2_try_add!(basis, v)
                push!(rows, v)
                success = true
            end
        end
    end

    # -----------------------------------------
    # Remaining s-r rows: pure-Z only (X=0)
    # -----------------------------------------
    for i in (r+1):s
        success = false
        attempts = 0

        while !success
            attempts += 1
            attempts > 2000 && error("Could not generate independent pure-Z row $i after many attempts")

            # Build constraints u_x · z = 0 for all existing rows u
            H_z = BitMatrix(undef, length(rows), n)
            for (eq, u) in enumerate(rows)
                H_z[eq, :] .= view(u, 1:n)
            end

            N_z = gf2_nullspace(H_z)
            size(N_z, 1) == 0 && error("No remaining pure-Z isotropic directions")

            # Sample random z from nullspace
            z_part = falses(n)
            for j in 1:size(N_z, 1)
                if rand(rng, Bool)
                    z_part .⊻= N_z[j, :]
                end
            end

            # reject zero z (it would add nothing)
            iszero(count(z_part)) && continue

            # candidate full vector (0 | z_part)
            v = vcat(falses(n), z_part)

            # Independence check (full 2n vector!)
            if gf2_try_add!(basis, v)
                push!(rows, v)
                success = true
            end
        end
    end

    # Convert to matrix
    S = BitMatrix(undef, s, 2n)
    for i in 1:s
        S[i, :] = rows[i]
    end

    return S
end

"""
    solve_linear_system_f2_with_random_gf2(A::BitMatrix, b::BitVector, rng) -> BitVector

Solve Ax = b (mod 2) with randomization of free variables.
Returns a solution vector, or nothing if no solution exists.
"""
function solve_linear_system_f2_with_random_gf2(A::BitMatrix, b::BitVector, rng)
    m, n = size(A)
    
    # Create augmented matrix [A | b]
    Aug = hcat(copy(A), reshape(b, m, 1))
    
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
    x = falses(n)
    free_vars = setdiff(1:n, pivot_cols)
    
    # Randomize free variables
    for var in free_vars
        x[var] = rand(rng, Bool)
    end
    
    # Solve for pivot variables in reverse order
    for col in reverse(pivot_cols)
        row = findfirst(i -> Aug[i, col], 1:m)
        
        # x[col] should satisfy: x[col] + sum(Aug[row, j] * x[j] for j > col) = Aug[row, n+1]
        val = Aug[row, n+1]
        for j in (col+1):n
            if Aug[row, j] && x[j]
                val = !val
            end
        end
        x[col] = val
    end
    
    return x
end
"""
    sample_from_nullspace(N::BitMatrix, existing_rows::Vector{BitVector}, n::Int, rng) -> BitVector

Helper to sample a random nonzero vector from nullspace that's linearly independent
from existing rows.
"""
function sample_from_nullspace(N::BitMatrix, existing_rows::Vector{BitVector}, n::Int, rng)
    v = falses(size(N, 2))
    
    attempts = 0
    max_attempts = 1000
    
    while attempts < max_attempts
        v .= false
        
        # Random linear combination of nullspace basis vectors
        for j in 1:size(N, 1)
            if rand(rng, Bool)
                v .⊻= N[j, :]
            end
        end
        
        # Check if nonzero and linearly independent
        if !iszero(count(v)) && !any(u -> u == v, existing_rows)
            return v
        end
        
        attempts += 1
    end
    
    error("Could not find linearly independent vector in nullspace after $max_attempts attempts")
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



# Run if executed as a script
run_tests(trials=5000, seed=1234)
