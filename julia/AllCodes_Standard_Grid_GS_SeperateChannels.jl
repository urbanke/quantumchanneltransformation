# ============================================
# Enumerate all binary (n-k) × (2n) matrices
# in the standard sub-block form, for all r
# ============================================
include("src/Symplectic.jl")
include("src/SGS.jl")

using .Symplectic, .SGS
using QECInduced, .Symplectic, .SGS
using Base.Threads
using Plots
using LinearAlgebra
using Random











"""
    concat_stabilizers_bool(Sout::BitMatrix, Sin::BitMatrix; ordering::Symbol=:interleaved) -> BitMatrix

Concatenate two stabilizer codes in Bool form (binary symplectic representation).

Parameters:
- Sout: r1 × 2n1 outer code stabilizers
- Sin: r2 × 2n2 inner code stabilizers  
- ordering: :interleaved (default) or :block

Returns: (r1*n2 + r2) × 2(n1*n2) concatenated stabilizer matrix

The concatenation works as follows:
- Inner stabilizers: Each inner stabilizer is applied to ALL n1 blocks simultaneously
- Outer stabilizers: Each outer stabilizer is "spread" across n2 inner positions (one copy per position)
"""
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

"""
    pauli_string_to_bool(s::String) -> BitVector

Convert a Pauli string like "IXYZ" to binary symplectic form [x1,x2,...,xn,z1,z2,...,zn].
I=(0,0), X=(1,0), Z=(0,1), Y=(1,1)
"""
function pauli_string_to_bool(s::String)
    n = length(s)
    result = falses(2*n)
    
    for (i, c) in enumerate(s)
        if c == 'X' || c == 'x'
            result[i] = true          # X part
        elseif c == 'Z' || c == 'z'
            result[n + i] = true      # Z part
        elseif c == 'Y' || c == 'y'
            result[i] = true          # X part
            result[n + i] = true      # Z part
        elseif c == 'I' || c == 'i'
            # Already false
        else
            error("Invalid Pauli character: $c")
        end
    end
    
    return result
end

"""
    bool_to_pauli_string(v::AbstractVector{Bool}) -> String

Convert binary symplectic form back to Pauli string.
"""
function bool_to_pauli_string(v::AbstractVector{Bool})
    n = length(v) ÷ 2
    @assert length(v) == 2*n "Vector must have even length"
    
    chars = Char[]
    for i in 1:n
        x = v[i]
        z = v[n + i]
        
        if !x && !z
            push!(chars, 'I')
        elseif x && !z
            push!(chars, 'X')
        elseif !x && z
            push!(chars, 'Z')
        else  # x && z
            push!(chars, 'Y')
        end
    end
    
    return String(chars)
end

"""
    stabilizers_strings_to_bool(stabs::Vector{String}) -> BitMatrix

Convert a vector of Pauli strings to a BitMatrix.
"""
function stabilizers_strings_to_bool(stabs::Vector{String})
    r = length(stabs)
    @assert r > 0 "Must have at least one stabilizer"
    
    n = length(stabs[1])
    @assert all(length(s) == n for s in stabs) "All stabilizers must have same length"
    
    result = BitMatrix(undef, r, 2*n)
    for i in 1:r
        result[i, :] = pauli_string_to_bool(stabs[i])
    end
    
    return result
end

"""
    stabilizers_bool_to_strings(S::AbstractMatrix{Bool}) -> Vector{String}

Convert a BitMatrix back to vector of Pauli strings.
"""
function stabilizers_bool_to_strings(S::AbstractMatrix{Bool})
    r = size(S, 1)
    return [bool_to_pauli_string(Vector{Bool}(S[i, :])) for i in 1:r]
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



function assert_isotropic(S::BitMatrix, n::Int)
    m = size(S, 1)
    for i in 1:m, j in i+1:m
        u = BitVector(S[i, :])
        v = BitVector(S[j, :])
        @assert !symplectic_dot(u, v, n) "Rows $i and $j anticommute!"
    end
    return true
end


function random_isotropic_basis(n::Int, s::Int; rng = Random.default_rng())
    @assert s ≤ n "Maximum isotropic dimension is n"

    rows = BitVector[]
    H = BitMatrix(undef, 0, 2n)  # symplectic constraints

    for i in 1:s
        # Build constraint matrix:
        # each previous row contributes one symplectic constraint
        if !isempty(rows)
            H = BitMatrix(undef, length(rows), 2n)
            for (j,v) in enumerate(rows)
                # constraint: <v, x> = 0
                # encoded as linear equation
                H[j, 1:n]      .= view(v, n+1:2n)
                H[j, n+1:2n]   .= view(v, 1:n)
            end
        end

        N = gf2_nullspace(H)
        @assert size(N,1) > 0 "No remaining isotropic directions"

        # sample random vector from nullspace
        v = falses(2n)
        for j in 1:size(N,1)
            if rand(rng, Bool)
                v .⊻= N[j,:]
            end
        end

        # ensure linear independence
        while any(u -> u == v, rows) || iszero(count(v))
            v .= false
            for j in 1:size(N,1)
                if rand(rng, Bool)
                    v .⊻= N[j,:]
                end
            end
        end

# SAFETY CHECK: ensure v commutes with all existing rows
        for u in rows
            @assert !symplectic_dot(u, v, n) "Generated non-isotropic vector!"
        end

        push!(rows, v)
    end

    S = BitMatrix(undef, s, 2n)
    for i in 1:s
        S[i,:] = rows[i]
    end
    assert_isotropic(S, n)

    return S
end

# -------------------------------
# Left half: s × n
# [ I_r  A1  A2
#   0     0   0 ]
# -------------------------------
function build_left_half(n::Int, k::Int, r::Int, A1::AbstractMatrix{<:Integer}, A2::AbstractMatrix{<:Integer})
    @assert 0 ≤ k ≤ n "Require 0 ≤ k ≤ n"
    s  = n - k
    @assert 0 ≤ r ≤ s "Require 0 ≤ r ≤ n-k"

    # sizes
    r1 = r
    r2 = s - r

    I_r = identity_matrix(r1)                 # r×r
    top = hcat(I_r, A1, A2)                   # r×(r + r2 + k) = r×n
    mid = hcat(zero_matrix(r2, r1),
               zero_matrix(r2, r2),
               zero_matrix(r2, k))            # (s-r)×n

    vcat(top, mid)                             # s × n
end

# -------------------------------
# Right half: s × n
# [ B   C1   C2
#   D    I   E ]
# -------------------------------
function build_right_half(n::Int, k::Int, r::Int,
                          B::AbstractMatrix{<:Integer},
                          C1::AbstractMatrix{<:Integer},
                          C2::AbstractMatrix{<:Integer},
                          D::AbstractMatrix{<:Integer},
                          E::AbstractMatrix{<:Integer})
    @assert 0 ≤ k ≤ n "Require 0 ≤ k ≤ n"
    s  = n - k
    @assert 0 ≤ r ≤ s "Require 0 ≤ r ≤ n-k"

    r1 = r
    r2 = s - r

    top = hcat(B,  C1, C2)                     # r×n
    I_r2 = identity_matrix(r2)                 # (s-r)×(s-r)
    mid = hcat(D, I_r2, E)                     # (s-r)×n

    vcat(top, mid)                              # s × n
end

# -------------------------------
# Full matrix: s × (2n)
# concat horizontally: [Left | Right]
# -------------------------------
function build_full_matrix(n::Int, k::Int, r::Int,
                           A1::AbstractMatrix{<:Integer},
                           A2::AbstractMatrix{<:Integer},
                           B::AbstractMatrix{<:Integer},
                           C1::AbstractMatrix{<:Integer},
                           C2::AbstractMatrix{<:Integer},
                           D::AbstractMatrix{<:Integer},
                           E::AbstractMatrix{<:Integer})
    left  = build_left_half(n, k, r, A1, A2)               # (n-k)×n
    right = build_right_half(n, k, r, B, C1, C2, D, E)     # (n-k)×n
    hcat(left, right)                                       # (n-k)×(2n)
end

# -------------------------------
# Main enumerator (lazy)
# Emits NamedTuple: (M, r, blocks=...)
# -------------------------------
"""
    iterate_standard_block_matrices(n, k; r=nothing, visit=nothing)

Enumerate all binary matrices of size (n-k) × (2n) in the standard sub-block form.

If `r === nothing`, iterates over all r = 0..(n-k).
If `r` is specified, only generates matrices for that specific value of r.

If `visit === nothing`, returns a `Channel{NamedTuple}` yielding:
  (M = Matrix{Int}, r = Int,
   blocks = (A1, A2, B, C1, C2, D, E))

If `visit` is a function `(info)->nothing`, it is called for each matrix
and the function returns `nothing`.
"""
# -------------------------------
# Optimized orthogonality checking
# -------------------------------
"""
    check_symplectic_orthogonality(left_row, right_row)

Check if two rows satisfy the symplectic orthogonality condition:
left1 * right2 + left2 * right1 (mod 2) = 0

Returns true if orthogonal, false otherwise.
"""
function check_symplectic_orthogonality(left_row::AbstractVector, right_row::AbstractVector)
    n = length(left_row)
    @assert length(right_row) == n "Rows must have same length"
    
    # Split each row into two halves
    mid = div(n, 2)
    left1 = left_row[1:mid]
    left2 = left_row[mid+1:end]
    right1 = right_row[1:mid]
    right2 = right_row[mid+1:end]
    
    # Compute inner product: left1 * right2 + left2 * right1 (mod 2)
    sum_val = 0
    for i in 1:mid
        sum_val += left1[i] * right2[i] + left2[i] * right1[i]
    end
    
    return (sum_val % 2) == 0
end

"""
    check_row_orthogonality_incremental(full_row, existing_rows)

Check if a new row is orthogonal to all existing rows.
Returns true if orthogonal to all, false otherwise.
"""
function check_row_orthogonality_incremental(full_row::AbstractVector, existing_rows::Vector)
    for existing_row in existing_rows
        if !check_symplectic_orthogonality(full_row, existing_row)
            return false
        end
    end
    return true
end

# -------------------------------
# Main enumerator with incremental checking
# -------------------------------
"""
    iterate_standard_block_matrices_optimized(n, k; r=nothing, visit=nothing)

Optimized version with incremental orthogonality checking.
Checks constraints as early as possible to prune the search tree aggressively.
"""
function iterate_standard_block_matrices_optimized(n::Int, k::Int; r::Union{Int,Nothing}=nothing, visit=nothing)
    @assert 0 ≤ k ≤ n "Require 0 ≤ k ≤ n"
    s = n - k
    
    # Determine range of r values
    r_range = if r === nothing
        0:s
    else
        @assert 0 ≤ r ≤ s "Require 0 ≤ r ≤ n-k"
        r:r
    end

    function drive(ch::Union{Channel,Nothing})
        for r_val in r_range
            r1 = r_val
            r2 = s - r_val

            I_r = identity_matrix(r1)
            I_r2 = identity_matrix(r2)
            zero_r2_r1 = zero_matrix(r2, r1)
            zero_r2_r2 = zero_matrix(r2, r2)
            zero_r2_k = zero_matrix(r2, k)
            
            for A1 in each_binary_matrix(r1, r2)
                for A2 in each_binary_matrix(r1, k)
                    left_top = hcat(I_r, A1, A2)
                    
                    for B in each_binary_matrix(r1, r1)
                        # Early check: after choosing B, check orthogonality constraints
                        # For each row i in top, check if [left_top[i,:], B[i,:], ?, ?] can be orthogonal to itself
                        # This is: left_top[i,:] * [B[i,:], ?, ?]^T where ? will be C1, C2
                        
                        for C1 in each_binary_matrix(r1, r2)
                            for C2 in each_binary_matrix(r1, k)
                                right_top = hcat(B, C1, C2)
                                
                                # Build completed top rows
                                top_rows = Vector{Vector{Int}}()
                                valid_top = true
                                
                                for i in 1:r1
                                    full_row = vcat(left_top[i, :], right_top[i, :])
                                    
                                    # Check against all previously validated rows
                                    if !check_row_orthogonality_incremental(full_row, top_rows)
                                        valid_top = false
                                        break
                                    end
                                    
                                    push!(top_rows, full_row)
                                end
                                
                                if !valid_top
                                    continue
                                end
                                
                                # Now iterate bottom rows
                                for D in each_binary_matrix(r2, r1)
                                    for E in each_binary_matrix(r2, k)
                                        left_bottom = hcat(zero_r2_r1, zero_r2_r2, zero_r2_k)
                                        right_bottom = hcat(D, I_r2, E)
                                        
                                        # Check bottom rows against each other and against top rows
                                        all_rows = copy(top_rows)
                                        valid_full = true
                                        
                                        for i in 1:r2
                                            full_row = vcat(left_bottom[i, :], right_bottom[i, :])
                                            
                                            # Check against all previously validated rows (top + previous bottom)
                                            if !check_row_orthogonality_incremental(full_row, all_rows)
                                                valid_full = false
                                                break
                                            end
                                            
                                            push!(all_rows, full_row)
                                        end
                                        
                                        if !valid_full
                                            continue
                                        end
                                        
                                        # All checks passed - build and emit matrix
                                        M = build_full_matrix(n, k, r_val, A1, A2, B, C1, C2, D, E)
                                        info = (M=M, r=r_val, blocks=(A1=A1, A2=A2, B=B, C1=C1, C2=C2, D=D, E=E))
                                        if visit === nothing
                                            put!(ch, info)
                                        else
                                            visit(info)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
        nothing
    end

    if visit === nothing
        return Channel{NamedTuple}() do ch
            drive(ch)
        end
    else
        drive(nothing)
        return nothing
    end
end




function split_trials(n,k,r_range,trials)

    r_range_holder = zeros(length(r_range))
    i = 1 
    for R in r_range
        r_range_holder[i] = count_standard_block_matrices(n, k; r = R)
        if r_range_holder[i] == 0 
            r_range_holder[i] = trials # overflow protection 
        end 
        i+=1
    end 
    # normalize 
    r_range_holder = r_range_holder./(sum(r_range_holder))
    # proportionally allocate trials 
    r_range_holder = r_range_holder .* trials 
    return round.(r_range_holder .+ max(trials/1e5,100)) # this way each matrix has some trials 
end 



function random_standard_block_matrices_optimized(n::Int, k::Int; r::Union{Int,Nothing}=nothing, visit=nothing, trials = 1e7, rng = MersenneTwister(2025))
    @assert 0 ≤ k ≤ n "Require 0 ≤ k ≤ n"
    s = n - k
    
    # Determine range of r values
    r_range = if r === nothing
        0:s
    else
        @assert 0 ≤ r ≤ s "Require 0 ≤ r ≤ n-k"
        r:r
    end

    splitTrials = split_trials(n,k,r_range,trials)
    println(splitTrials)

    function drive(ch::Union{Channel,Nothing})
        r_val_counter = 0 
        for r_val in r_range
            r_val_counter += 1 
            r1 = r_val
            r2 = s - r_val

            I_r = identity_matrix(r1)
            I_r2 = identity_matrix(r2)
            zero_r2_r1 = zero_matrix(r2, r1)
            zero_r2_r2 = zero_matrix(r2, r2)
            zero_r2_k = zero_matrix(r2, k)
            
            for j in 1:Int(splitTrials[r_val_counter])
                A1 = rand_binary_matrix(r1,r2, rng) 
                A2 = rand_binary_matrix(r1, k, rng)
                left_top = hcat(I_r, A1, A2)
                B = rand_binary_matrix(r1, r1, rng)        
                C1 = rand_binary_matrix(r1, r2, rng)
                C2 = rand_binary_matrix(r1, k, rng)
                right_top = hcat(B, C1, C2)
                # Build completed top rows
                top_rows = Vector{Vector{Int}}()
                valid_top = true
                                
                for i in 1:r1
                    full_row = vcat(left_top[i, :], right_top[i, :])
                                    
                    # Check against all previously validated rows
                    if !check_row_orthogonality_incremental(full_row, top_rows)
                        valid_top = false
                        break
                    end
                                    
                    push!(top_rows, full_row)
                end
                                
                if !valid_top
                    continue
                end
                                
                # Now iterate bottom rows
                                # Now iterate bottom rows
                D = rand_binary_matrix(r2, r1, rng)
                E = rand_binary_matrix(r2, k, rng)
                left_bottom = hcat(zero_r2_r1, zero_r2_r2, zero_r2_k)
                right_bottom = hcat(D, I_r2, E)
                                                                
                # Check bottom rows against each other and against top rows
                all_rows = copy(top_rows)
                valid_full = true
                                        
                for i in 1:r2
                    full_row = vcat(left_bottom[i, :], right_bottom[i, :])
                                            
                    # Check against all previously validated rows (top + previous bottom)
                    if !check_row_orthogonality_incremental(full_row, all_rows)
                        valid_full = false
                        break
                    end
                                            
                    push!(all_rows, full_row)
                end
                                        
                if !valid_full
                    continue
                end
                                        
                # All checks passed - build and emit matrix
                M = build_full_matrix(n, k, r_val, A1, A2, B, C1, C2, D, E)
                info = (M=M, r=r_val, blocks=(A1=A1, A2=A2, B=B, C1=C1, C2=C2, D=D, E=E))
                if visit === nothing
                    put!(ch, info)
                else
                    visit(info)
                end
            end
        end
        nothing
    end

    if visit === nothing
        return Channel{NamedTuple}() do ch
            drive(ch)
        end
    else
        drive(nothing)
        return nothing
    end
end


# -------------------------------
# Counting utility (how many total)
# -------------------------------
"""
    count_standard_block_matrices(n, k; r=nothing)

Return total count of emitted matrices.

If r === nothing: sums over all r = 0..(n-k)
If r is specified: returns count for that specific r value

For each r, number of variable bits is:
  Left:  r*(s-r) + r*k
  Right: r*r + r*(s-r) + r*k + (s-r)*r + (s-r)*k
Total per r: T(r) = 3*r*s - 2*r^2 + k*s, where s = n-k.
"""
function count_standard_block_matrices(n::Int, k::Int; r::Union{Int,Nothing}=nothing)
    @assert 0 ≤ k ≤ n "Require 0 ≤ k ≤ n"
    s = n - k
    
    r_range = if r === nothing
        0:s
    else
        @assert 0 ≤ r ≤ s "Require 0 ≤ r ≤ n-k"
        r:r
    end
    
    total = 0
    for r_val in r_range
        # bits = number of variable entries for this r
        s1 = n - k - r_val
        # bits = number of variable entries for this r
        bits = s1*r_val + k*r_val + r_val*r_val + s1*r_val + k*r_val + s1*r_val + s1*k
        total += 2^bits
    end
    return total
end

# ============================================
# Example Usage
# ============================================

function test_each_binary_matrix()
    println("Testing each_binary_matrix(1, 1):")
    count = 0
    for M in each_binary_matrix(1, 1)
        count += 1
        println("  Matrix $count: $M")
    end
    println("  Total: $count (expected 2)")
    
    println("\nTesting each_binary_matrix(2, 2):")
    count = 0
    for M in each_binary_matrix(2, 2)
        count += 1
        println("  Matrix $count:")
        println(M)
    end
    println("  Total: $count (expected 16)")
end

function smith_rep_maker(n)
    S = falses(n-1,2n)
    for i in 1:(n-1)
        S[i,[i+n,2n]] .= true 
    end 
    return S
end 

function repitition_code_check(ChannelType, n; pz=nothing, p_1 = 1, q_2 = 0 , points=15, customP=nothing, customP1 = nothing, customP2 = nothing, δ = .3, newBest = nothing, threads = Threads.nthreads(), pz_range_override = nothing)
    s = n - 1  # Number of rows in the (n-k) × (2n) matrix
    
    # Initialize best trackers for each grid point

    S_best = [falses(s, 2n) for _ in 1:points]  # Best matrix at each grid point
    r_best = fill(-1, points)  # Best r value at each grid point
    
    # Compute pz if not provided
    if pz === nothing 
        pz = findZeroRate(f, 0, 0.5; maxiter=1000, ChannelType=ChannelType, customP=customP)
    end 

    if pz_range_override === nothing 
        pz_range = range(.236,.272, length=points)
        pz_range = range(pz - pz*δ/2, pz + pz*δ/4, length=points)   
    else 
        pz_range = pz_range_override 
    end  

    #pz_range = range(.236,.272, length=points)
    #pz_range = range(0.2334285714285714 - 0.0025714285714285856, 0.2334285714285714 + 0.0025714285714285856, length = points)

    if newBest === nothing 
        hb_best = QECInduced.sweep_hashing_grid(pz_range, ChannelType; customP = customP)
    else 
        hb_best = newBest
    end 
    
    # just checking the smith reptition matrix 
    mat = smith_rep_maker(n) 
    
    # Convert to Bool matrix
    S = Matrix{Bool}(mat)
        

 q_1 = 1-(p_1 + q_2)
    # Check the induced channel at all grid points
    hb_grid = QECInduced.check_induced_channel(S, pz; ChannelType=ChannelType, sweep=true, ps=pz_range, customP=customP)*p_1 .+ check_induced_channel(S, pz; ChannelType=ChannelType, sweep=true, ps=pz_range, customP=customP1)*q_1 .+ check_induced_channel(S, pz; ChannelType=ChannelType, sweep=true, ps=pz_range, customP=customP2)*q_2
    # Find which grid points improved
    improved_indices = findall(hb_grid .> (hb_best .+ eps()))

    
    # Update best for each improved point
    if !isempty(improved_indices)
        for idx in improved_indices
            hb_best[idx] = hb_grid[idx]
            S_best[idx] = copy(S)
        end
        
        println("\n" * "=" ^ 70)
        println("Smith Code Improved!")
        println("Improved at $(length(improved_indices)) grid point(s): $improved_indices")
        println("\nGrid point details:")
        for idx in improved_indices
            println("  Point $idx: pz=$(round(pz_range[idx], digits=4)), hb=$(round(hb_best[idx], digits=6))")
        end
        println("\nS_best (showing first improved point) =")
        println(Symplectic.build_from_bits(S_best[improved_indices[1]]))
        println("=" ^ 70 * "\n")
    end

    return hb_best, S_best
end 



function All_Codes_Random_SGS(ChannelType, n, k; p_1 = 1, q_2 = 0 , pz=nothing, points=15, customP=nothing, customP1 = nothing, customP2 = nothing, δ = .3, newBest = nothing, trials = 1e7, rng = MersenneTwister(2025), pz_range_override = nothing)
    s = n - k  # Number of rows in the (n-k) × (2n) matrix
    
    # Initialize best trackers for each grid point

    S_best = [falses(s, 2n) for _ in 1:points]  # Best matrix at each grid point
    
    # Compute pz if not provided
    if pz === nothing 
        pz = findZeroRate(f, 0, 0.5; maxiter=1000, ChannelType=ChannelType, customP=customP)
    end 


    if pz_range_override === nothing 
        pz_range = range(.236,.272, length=points)
        pz_range = range(pz - pz*δ/2, pz + pz*δ/4, length=points)   
    else 
        pz_range = pz_range_override 
    end  

    if newBest === nothing 
        hb_best = QECInduced.sweep_hashing_grid(pz_range, ChannelType; customP = customP)
    else 
        hb_best = newBest
    end 
        

 q_1 = 1-(p_1 + q_2)
    println("=" ^ 70)
    println("Generating binary matrices ($s × $(2*n)) in standard block form")
    println("Parameters: n=$n, k=$k, s=$s, grid_points=$points")
    println("pz range: [$(pz_range[1]), $(pz_range[end])]")
    println("=" ^ 70)
    
    # Calculate total possible without constraints
    
    count = 0
    last_print_count = 0
    print_interval = max(1, div(trials, 20))  # Print ~20 times
    
    println("\nStarting enumeration with orthogonality constraints...\n")
    
    for i in 1:trials
        M = random_isotropic_basis(n, n-k)
        count += 1
        
        # Convert to Bool matrix
        S = Matrix{Bool}(M)
        # Check the induced channel at all grid points
        hb_grid = QECInduced.check_induced_channel(S, pz; ChannelType=ChannelType, sweep=true, ps=pz_range, customP=customP)*p_1 .+ check_induced_channel(S, pz; ChannelType=ChannelType, sweep=true, ps=pz_range, customP=customP1)*q_1 .+ check_induced_channel(S, pz; ChannelType=ChannelType, sweep=true, ps=pz_range, customP=customP2)*q_2
        
        # Find which grid points improved
        improved_indices = findall(hb_grid .> (hb_best .+ eps()))
        
        # Update best for each improved point
        if !isempty(improved_indices)
            for idx in improved_indices
                hb_best[idx] = hb_grid[idx]
                S_best[idx] = copy(S)
            end
            
            println("\n" * "=" ^ 70)
            println("NEW BEST FOUND! (Matrix #$count)")
            println("Improved at $(length(improved_indices)) grid point(s): $improved_indices")
            println("\nGrid point details:")
            for idx in improved_indices
                println("  Point $idx: pz=$(round(pz_range[idx], digits=4)), hb=$(round(hb_best[idx], digits=6))")
            end
            println("\nS_best (showing first improved point) =")
            println(Symplectic.build_from_bits(S_best[improved_indices[1]]))
            println("=" ^ 70 * "\n")
        end
        
    end
    
    println("\n" * "=" ^ 70)
    println("SEARCH COMPLETE")
    println("=" ^ 70)
    println("Valid matrices found (satisfying orthogonality): $count")
        
    return hb_best, S_best
end




"""
    All_Codes_DFS(ChannelType, n, k; pz=nothing, r_specific=nothing)

Adapted version to work with the standard block matrix enumeration.

Parameters:
- ChannelType: The type of quantum channel
- n: Code parameter n
- k: Code parameter k (s = n-k is the number of rows)
- pz: Optional depolarization parameter (computed if not provided)
- r_specific: Optional specific value of r to test (tests all r if nothing)

Returns:
- hb_best: Best value found
- S_best: Best stabilizer matrix found
- r_best: The r value that gave the best result
"""
function All_Codes_DFS(ChannelType, n, k; pz=nothing, p_1 = 1, q_2 = 0, r_specific=nothing, points=15, customP=nothing, customP1=nothing, customP2 = nothing, δ = .3, newBest = nothing, threads = Threads.nthreads(), trials = 1e6, useTrials = false, pz_range_override = nothing)
    s = n - k  # Number of rows in the (n-k) × (2n) matrix
    
    # Initialize best trackers for each grid point

    S_best = [falses(s, 2n) for _ in 1:points]  # Best matrix at each grid point
    r_best = fill(-1, points)  # Best r value at each grid point
    
    # Compute pz if not provided
    if pz === nothing 
        pz = findZeroRate(f, 0, 0.5; maxiter=1000, ChannelType=ChannelType, customP=customP)
    end 

    if pz_range_override === nothing 
        pz_range = range(.236,.272, length=points)
        pz_range = range(pz - pz*δ/2, pz + pz*δ/4, length=points)   
    else 
        pz_range = pz_range_override 
    end  

    #pz_range = range(.236,.272, length=points)
    #pz_range = range(0.2334285714285714 - 0.0025714285714285856, 0.2334285714285714 + 0.0025714285714285856, length = points)

    if newBest === nothing 
        hb_best = QECInduced.sweep_hashing_grid(pz_range, ChannelType; customP = customP)
    else 
        hb_best = newBest
    end 
    println(hb_best)
    println("=" ^ 70)
    println("Generating binary matrices ($s × $(2*n)) in standard block form")
    println("Parameters: n=$n, k=$k, s=$s, grid_points=$points")
    if r_specific !== nothing
        println("Testing only r=$r_specific")
    else
        println("Testing all r values from 0 to $s")
    end
    println("pz range: [$(pz_range[1]), $(pz_range[end])]")
    println("=" ^ 70)
    
    # Calculate total possible without constraints
    total_possible_no_constraints = count_standard_block_matrices(n, k; r=r_specific)
    println("\nTotal matrices without orthogonality constraints: $total_possible_no_constraints")
    
    count = 0
    count_by_r = Dict{Int,Int}()
    last_print_count = 0
    print_interval = max(1, div(total_possible_no_constraints, 20))  # Print ~20 times
    
    println("\nStarting enumeration with orthogonality constraints...\n")
        

 q_1 = 1-(p_1 + q_2)
    # Use the optimized iterator with orthogonality checking
    for info in iterate_standard_block_matrices_optimized(n, k; r=r_specific)
        count += 1
        r_val = info.r
        count_by_r[r_val] = get(count_by_r, r_val, 0) + 1
        
        # Convert to Bool matrix
        S = Matrix{Bool}(info.M)
        
        # Check the induced channel at all grid points
        hb_grid = QECInduced.check_induced_channel(S, pz; ChannelType=ChannelType, sweep=true, ps=pz_range, customP=customP)*p_1 .+ check_induced_channel(S, pz; ChannelType=ChannelType, sweep=true, ps=pz_range, customP=customP1)*q_1 .+ check_induced_channel(S, pz; ChannelType=ChannelType, sweep=true, ps=pz_range, customP=customP2)*q_2
        # Find which grid points improved
        improved_indices = findall(hb_grid .> (hb_best .+ eps()))

        
        # Update best for each improved point
        if !isempty(improved_indices)
            for idx in improved_indices
                hb_best[idx] = hb_grid[idx]
                S_best[idx] = copy(S)
                r_best[idx] = r_val
            end
            
            println("\n" * "=" ^ 70)
            println("NEW BEST FOUND! (Matrix #$count, r=$r_val)")
            println("Improved at $(length(improved_indices)) grid point(s): $improved_indices")
            println("\nGrid point details:")
            for idx in improved_indices
                println("  Point $idx: pz=$(round(pz_range[idx], digits=4)), hb=$(round(hb_best[idx], digits=6))")
            end
            println("\nS_best (showing first improved point) =")
            println(Symplectic.build_from_bits(S_best[improved_indices[1]]))
            println("=" ^ 70 * "\n")
        end

        if (count > trials) && useTrials
            break
        end 
    end
    
    println("\n" * "=" ^ 70)
    println("SEARCH COMPLETE")
    println("=" ^ 70)
    println("Valid matrices found (satisfying orthogonality): $count")
    println("Total possible (without constraints): $total_possible_no_constraints")
    println("Efficiency gain: $(round((1 - count/total_possible_no_constraints)*100, digits=1))% pruned")
    
    println("\nBreakdown by r:")
    for r_val in sort(collect(keys(count_by_r)))
        println("  r=$r_val: $(count_by_r[r_val]) matrices checked")
    end
    return hb_best, S_best, r_best
end




"""
    All_Codes_Random(ChannelType, n, k; pz=nothing, r_specific=nothing)

Randomly generate stabilizer matrices

Parameters:
- ChannelType: The type of quantum channel
- n: Code parameter n
- k: Code parameter k (s = n-k is the number of rows)
- pz: Optional depolarization parameter (computed if not provided)
- r_specific: Optional specific value of r to test (tests all r if nothing)

Returns:
- hb_best: Best value found
- S_best: Best stabilizer matrix found
- r_best: The r value that gave the best result
"""
function All_Codes_Random(ChannelType, n, k; pz=nothing, p_1 = 1, q_2 = 0 , r_specific=nothing, points=15, customP=nothing, customP1 = nothing, customP2 = nothing, δ = .3, newBest = nothing, trials = 1e7, rng = MersenneTwister(2025), pz_range_override = nothing)
    s = n - k  # Number of rows in the (n-k) × (2n) matrix
    
    # Initialize best trackers for each grid point

    S_best = [falses(s, 2n) for _ in 1:points]  # Best matrix at each grid point
    r_best = fill(-1, points)  # Best r value at each grid point
    
    # Compute pz if not provided
    if pz === nothing 
        pz = findZeroRate(f, 0, 0.5; maxiter=1000, ChannelType=ChannelType, customP=customP)
    end 
    

 q_1 = 1-(p_1 + q_2)
    if pz_range_override === nothing 
        pz_range = range(.236,.272, length=points)
        pz_range = range(pz - pz*δ/2, pz + pz*δ/4, length=points)   
    else 
        pz_range = pz_range_override 
    end  

    if newBest === nothing 
        hb_best = QECInduced.sweep_hashing_grid(pz_range, ChannelType; customP = customP)
    else 
        hb_best = newBest
    end 

    println("=" ^ 70)
    println("Generating binary matrices ($s × $(2*n)) in standard block form")
    println("Parameters: n=$n, k=$k, s=$s, grid_points=$points")
    if r_specific !== nothing
        println("Testing only r=$r_specific")
    else
        println("Testing all r values from 0 to $s")
    end
    println("pz range: [$(pz_range[1]), $(pz_range[end])]")
    println("=" ^ 70)
    
    # Calculate total possible without constraints
    total_possible_no_constraints = count_standard_block_matrices(n, k; r=r_specific)
    if total_possible_no_constraints == 0 
        total_possible_no_constraints = trials 
    end 
    println("\nTotal matrices without orthogonality constraints: $total_possible_no_constraints")
    
    count = 0
    count_by_r = Dict{Int,Int}()
    last_print_count = 0
    print_interval = max(1, div(total_possible_no_constraints, 20))  # Print ~20 times
    
    println("\nStarting enumeration with orthogonality constraints...\n")
    
    # Use the optimized iterator with orthogonality checking
    for info in random_standard_block_matrices_optimized(n, k; r=r_specific, trials = trials, rng = rng)
        count += 1
        r_val = info.r
        count_by_r[r_val] = get(count_by_r, r_val, 0) + 1
        
        # Convert to Bool matrix
        S = Matrix{Bool}(info.M)
        # Check the induced channel at all grid points
        hb_grid = QECInduced.check_induced_channel(S, pz; ChannelType=ChannelType, sweep=true, ps=pz_range, customP=customP)*p_1 .+ check_induced_channel(S, pz; ChannelType=ChannelType, sweep=true, ps=pz_range, customP=customP1)*q_1 .+ check_induced_channel(S, pz; ChannelType=ChannelType, sweep=true, ps=pz_range, customP=customP2)*q_2
        
        # Find which grid points improved
        improved_indices = findall(hb_grid .> (hb_best .+ eps()))
        
        # Update best for each improved point
        if !isempty(improved_indices)
            for idx in improved_indices
                hb_best[idx] = hb_grid[idx]
                S_best[idx] = copy(S)
                r_best[idx] = r_val
            end
            
            println("\n" * "=" ^ 70)
            println("NEW BEST FOUND! (Matrix #$count, r=$r_val)")
            println("Improved at $(length(improved_indices)) grid point(s): $improved_indices")
            println("\nGrid point details:")
            for idx in improved_indices
                println("  Point $idx: pz=$(round(pz_range[idx], digits=4)), hb=$(round(hb_best[idx], digits=6))")
            end
            println("\nS_best (showing first improved point) =")
            println(Symplectic.build_from_bits(S_best[improved_indices[1]]))
            println("=" ^ 70 * "\n")
        end
        
    end
    
    println("\n" * "=" ^ 70)
    println("SEARCH COMPLETE")
    println("=" ^ 70)
    println("Valid matrices found (satisfying orthogonality): $count")
    println("Total possible (without constraints): $total_possible_no_constraints")
    println("Efficiency gain: $(round((1 - count/total_possible_no_constraints)*100, digits=1))% pruned")
    
    println("\nBreakdown by r:")
    for r_val in sort(collect(keys(count_by_r)))
        println("  r=$r_val: $(count_by_r[r_val]) matrices checked")
    end
    
    return hb_best, S_best
end




"""
    All_Codes_DFS_parallel(ChannelType, n, k; pz=nothing, use_threads=true)

Parallel version that searches all r values independently using Julia threads.
Can be more efficient since different r values can be checked in parallel.

To use threading, start Julia with: `julia -t auto` or `julia -t 8` (for 8 threads)
Check available threads with: `Threads.nthreads()`
"""

function All_Codes_DFS_parallel(ChannelType, n, k; pz=nothing, p_1 = 1, q_2 = 0 , use_threads=true, points = 15, customP = nothing, customP1 = nothing, customP2 = nothing, δ = .3, newBest = nothing, trials = 1e6, useTrials = false, pz_range_override = nothing) 
    s = n - k

    if pz === nothing 
        pz = findZeroRate(f, 0, 0.5; maxiter=1000, ChannelType=ChannelType, customP = customP)
    end


    if pz_range_override === nothing 
        pz_range = range(.236,.272, length=points)
        pz_range = range(pz - pz*δ/2, pz + pz*δ/4, length=points)   
    else 
        pz_range = pz_range_override 
    end  

    n_threads = Threads.nthreads()
    println("=" ^ 70)
    println("PARALLEL SEARCH: Testing each r value independently")
    println("Parameters: n=$n, k=$k, s=$s")
    println("Available threads: $n_threads")
    println("Using threads: $use_threads")
    println("=" ^ 70)
    
    if use_threads && n_threads == 1
        @warn "Only 1 thread available. Start Julia with `julia -t auto` for multi-threading."
    end
    
    # Pre-allocate results array
    r_values = collect(0:s)
    n_r = length(r_values)
    results = Vector{Any}(undef, n_r)
    #s_best = Vector{Any}(undef, points)
    s_best = [falses(s, 2n) for _ in 1:points]  # Best matrix at each grid point

    
    if use_threads && n_threads > 1
        # Parallel execution using threads
        println("\nStarting parallel search across $n_r r values using $n_threads threads...")
        
        Threads.@threads for i in 1:n_r
            r_val = r_values[i]
            println("Thread $(Threads.threadid()): Starting r=$r_val")
            
            hb, S, r = All_Codes_DFS(ChannelType, n, k; pz=pz, p_1 = p_1, q_2 = q_2, r_specific=r_val, customP = customP, customP1 = customP1, customP2 = customP2, δ = δ, newBest = newBest, points = points, threads = 0, trials = trials, useTrials = useTrials, pz_range_override = pz_range)
            results[i] = (r=r_val, hb=hb, S=S)
            
            println("Thread $(Threads.threadid()): Completed r=$r_val, hb=$hb")
        end
    else
        # Sequential execution (fallback)
        println("\nRunning sequential search (no threading)...")
        for i in 1:n_r
            r_val = r_values[i]
            println("\n--- Starting search for r=$r_val ---")
            
            hb, S, r = All_Codes_DFS(ChannelType, n, k; pz=pz, p_1 = p_1, q_2 = q_2, r_specific=r_val, points = points, customP = customP, customP1 = customP1, customP2 = customP2, δ = δ, newBest = newBest, trials = trials, useTrials = useTrials, pz_range_override = pz_range)
            results[i] = (r=r_val, hb=hb, S=S)
        end
    end
    if newBest === nothing
        total_best = QECInduced.sweep_hashing_grid(pz_range, ChannelType; customP = customP)
    else 
        total_best = newBest
    end

    for i in 1:n_r 
        replaceIndices = findall(results[i].hb .> total_best)
        if !isempty(replaceIndices)
            s_best[replaceIndices] = (results[i].S)[replaceIndices]
            total_best = max.(total_best, results[i].hb)
        end
    end
    # Find overall best
    #best_idx = argmax([res.hb for res in results])
    #best_result = results[best_idx]
    
    println("\n" * "=" ^ 70)
    println("PARALLEL SEARCH COMPLETE")
    println("-" ^ 70)
    println("Results by r value:")
    for res in results
        println("  r=$(res.r): hb=$(res.hb)")
    end
    println("-" ^ 70)
    println("OVERALL BEST:")
    #println("  r_best = $(best_result.r)")
    println("  overall_best = $(total_best)")
    println("=" ^ 70)
    return total_best, s_best
end





"""
    check_threading_setup()

Utility function to check if threading is properly configured.
"""
function check_threading_setup()
    n_threads = Threads.nthreads()
    
    println("=" ^ 70)
    println("JULIA THREADING CONFIGURATION")
    println("=" ^ 70)
    println("Number of threads: $n_threads")
    
    if n_threads == 1
        println("\n⚠️  WARNING: Only 1 thread available!")
        println("\nTo enable multi-threading, restart Julia with:")
        println("  julia -t auto          # Use all available cores")
        println("  julia -t 4             # Use 4 threads")
        println("  julia -t 8             # Use 8 threads")
        println("\nOr set the environment variable:")
        println("  export JULIA_NUM_THREADS=auto")
    else
        println("\n✓ Multi-threading is enabled!")
        println("\nTesting thread distribution:")
        
        counts = zeros(Int, n_threads)
        Threads.@threads for i in 1:1000
            tid = Threads.threadid()
            counts[tid] += 1
        end
        
        println("\nWork distribution across threads:")
        for (tid, count) in enumerate(counts)
            if count > 0
                bar = "█" ^ div(count, 10)
                println("  Thread $tid: $count iterations $bar")
            end
        end
    end
    println("=" ^ 70)
end


"""
    compare_enumeration_methods(n, k)

Compare the old and new enumeration methods to verify correctness.
Only for small n, k where enumeration is feasible.
"""
function compare_enumeration_methods(n, k)
    s = n - k
    
    println("=" ^ 70)
    println("COMPARING ENUMERATION METHODS")
    println("Parameters: n=$n, k=$k, s=$s")
    println("=" ^ 70)
    
    # Count without orthogonality
    println("\n1. Counting all matrices (no orthogonality)...")
    count_all = 0
    time_all = @elapsed begin
        for info in iterate_standard_block_matrices_optimized(n, k)
            # This uses the function that has orthogonality checking
            # To count ALL, we'd need a version without checking
            count_all += 1
        end
    end
    
    expected_all = count_standard_block_matrices(n, k)
    
    println("   Matrices with orthogonality: $count_all")
    println("   Total possible (formula): $expected_all")
    println("   Time: $(round(time_all * 1000, digits=2)) ms")
    println("   Reduction: $(expected_all - count_all) matrices pruned by orthogonality")
    
    println("\n" * "=" ^ 70)
end


function printCodes(base_grid, points, pz_range, s_best, hashing, ChannelType)
    open("hashing_bound_envelope_"*ChannelType*".txt", "w") do file
        for i in 1:points
            if base_grid[i] > hashing[i]
                s_best_point = Symplectic.build_from_bits(s_best[i])
                write(file, "Point: $(pz_range[i])\n")
                write(file, "Induced Hashing Bound: $(base_grid[i])\n")
                write(file, "S Matrix:\n")
                write(file, join(string.(s_best_point), " ") * "\n\n")
            end
        end
        write(file, string(base_grid) * "\n")
        write(file, "[")
        for i in 1:(length(pz_range) - 1) 
            write(file, string(pz_range[i]) * ",")
        end 
        write(file, string(pz_range[end]) * "]")
    end
end 

function printCodesSlurm(base_grid, points, pz_range, s_best, hashing, ChannelType)
        for i in 1:points
            if base_grid[i] > hashing[i]
                s_best_point = Symplectic.build_from_bits(s_best[i])
                println("Point: $(pz_range[i])\n")
                println("Induced Hashing Bound: $(base_grid[i])\n")
                println("S Matrix:\n")
                println(join(string.(s_best_point), " ") * "\n\n")
            end
        end
        println(string(base_grid) * "\n")
        println("[")
        for i in 1:(length(pz_range) - 1)
            println(string(pz_range[i]) * ",")
        end
        println(string(pz_range[end]) * "]")
end


function envelope_finder(n_range, ChannelType; pz = nothing, p_1 = 1, q_2 = 0 , customP = nothing, customP1 = nothing, customP2 = nothing, points = 15, δ = .3, randomSearch = false, useTrials = false, trials = 1e7, rng = MersenneTwister(2025), pz_range_override = nothing)


    if pz === nothing 
        pz = findZeroRate(f, 0, 0.5; maxiter=1000, ChannelType=ChannelType, customP = customP)
    end
    println(pz)

    if pz_range_override === nothing 
        pz_range = range(.236,.272, length=points)
        pz_range = range(pz - pz*δ/2, pz + pz*δ/4, length=points)   
    else 
        pz_range = pz_range_override 
    end 
    for i in 1:length(pz_range)
        print(pz_range[i])
        print(", ")
    end
    println("]")
 q_1 = 1-(p_1 + q_2)
    hashing = QECInduced.sweep_hashing_grid(pz_range, ChannelType; customP = customP)*p_1 .+ QECInduced.sweep_hashing_grid(pz_range, ChannelType; customP = customP1)*q_1 .+ QECInduced.sweep_hashing_grid(pz_range, ChannelType; customP = customP2)*q_2
    best_grid = copy(hashing)
    println(hashing)
    println(length(hashing))
    println("HASHING")
    base_grid = copy(hashing)
    s_best = Vector{Any}(undef, points)
    trials_unaltered = trials 
    elapsed_time = @elapsed begin
        for n in n_range
            # do this first - check if smiths codes are better 
            best_grid, S_grid = repitition_code_check(ChannelType, n; pz = pz, p_1 = p_1, q_2 = q_2, points= points, customP= customP, customP1 = customP1, customP2 = customP2, δ = δ, newBest = best_grid, pz_range_override = pz_range)
            improve_indices = findall(best_grid .> base_grid)
            s_best[improve_indices] = S_grid[improve_indices]
            base_grid = max.(base_grid,best_grid)
            #for k in 1:(n-1)#Int(floor(.6*n))
            for k in 1:n-1
                elapsed_time_internal = @elapsed begin
                    base_trials = count_standard_block_matrices(n, k) 
                    if base_trials == 0 
                        base_trials = trials 
                    end 
                    if randomSearch #&& (base_trials ≥ trials_unaltered)
                        println("Using Random Search")
                        #best_grid, S_grid = All_Codes_Random(ChannelType, n, k; customP = customP, points = points, δ = δ, newBest = best_grid, trials = trials, rng = rng, pz_range_override = pz_range)
                        best_grid, S_grid = All_Codes_Random_SGS(ChannelType, n, k; pz = pz, p_1 = p_1, q_2 = q_2, customP = customP, customP1 = customP1, customP2 = customP2, points = points, δ = δ, newBest = best_grid, trials = trials, pz_range_override = pz_range)
                        improve_indices = findall(best_grid .> base_grid)
                        s_best[improve_indices] = S_grid[improve_indices]
                        base_grid = max.(base_grid,best_grid)
                    else 
                        println("Using Iterative Search")
                        best_grid, S_grid = All_Codes_DFS_parallel(ChannelType, n, k; pz = pz, p_1 = p_1, q_2 = q_2, customP = customP, customP1 = customP1, customP2 = customP2, points = points, δ = δ, newBest = best_grid, trials = trials, useTrials = useTrials, pz_range_override = pz_range)
                        improve_indices = findall(best_grid .> base_grid)
                        s_best[improve_indices] = S_grid[improve_indices]
                        base_grid = max.(base_grid,best_grid)
                    end
                end  
                println("Elapsed time: $elapsed_time_internal seconds") 
                printCodes(base_grid, points, pz_range, s_best, hashing, ChannelType)
            end
         end
    end
    println("Total time: $elapsed_time seconds")
    p_plot = copy(pz_range)
    if customP === nothing 
        #p_plot = pz_range
    else 
        p_plot = zeros(points)
        for i in 1:points
            p_plot[i] = customP(pz_range[i]; plot = true)
        end 
    end
    println("p: ", pz_range)
    plt = plot(
        p_plot, hashing;
        label = "Original channel (per-qubit 1 - H(p))",
        xlabel = ChannelType * " probability p",
        ylabel = "Hashing bound",
        title = "Hashing bounds vs p",
        marker = :circle,
        linewidth = 2,
    )

    plot!(plt, p_plot, base_grid; label = "Induced (per-syndrome conditional entropy)", marker = :square, linewidth = 2)

    # Save figure (and print the path)
    outfile = "hashing_bound_envelope.png"
    savefig(plt, outfile)
    println("Saved plot to $(outfile)")
    return hashing, base_grid, s_best
end


function ninexz(x; tuple = false, plot = false) # this is an example of customP, which gives the same one smith did 
    z = x/9
    pI = (1-z)*(1-x) 
    pX = x*(1-z) 
    pZ = z*(1-x)
    pY = z*x
    if tuple # this should always be here, do not touch 
        return (pI, pX, pZ, pY)
    end
    if plot # this is to plot different things (for example, smith plots 1-pI instead of pX despite working with pX)
        return 1-pI 
    end 
    return [pI, pX, pZ, pY]
end 


function depolar(p; tuple = false, plot = false) # this is an example of customP, which gives the same one smith did 
    pI = (1-p)
    pX = p/3
    pZ = p/3
    pY = p/3
    if tuple # this should always be here, do not touch 
        return (pI, pX, pZ, pY)
    end
    if plot # this is to plot different things (for example, smith plots 1-pI instead of pX despite working with pX)
        return p 
    end 
    return [pI, pX, pZ, pY]
end 

function twoPauli(p; tuple = false, plot = false) # this is an example of customP, which gives the same one smith did 
    pI = (1-p)
    pX = p/2
    pZ = p/2
    pY = 0.0 
    if tuple # this should always be here, do not touch 
        return (pI, pX, pZ, pY)
    end
    if plot # this is to plot different things (for example, smith plots 1-pI instead of pX despite working with pX)
        return p
    end 
    return [pI, pX, pZ, pY]
end 

function indy(x; tuple = false, plot = false)
    z = x 
    pI = (1-z)*(1-x) 
    pX = x*(1-z) 
    pZ = z*(1-x)
    pY = z*x
    if tuple # this should always be here, do not touch 
        return (pI, pX, pZ, pY)
    end
    if plot # this is to plot different things (for example, smith plots 1-pI instead of pX despite working with pX)
        return 1-pI 
    end 
    return [pI, pX, pZ, pY]
end 



function special(x; tuple = false, plot = false)
        pI = 0.727799413564745
pZ = 0.2721354797909989
pX = 3.255332212835746e-5
pY = 3.255332212835746e-5

    if tuple # this should always be here, do not touch 
        return (pI, pX, pZ, pY)
    end
    if plot # this is to plot different things (for example, smith plots 1-pI instead of pX despite working with pX)
        return 1-pI 
    end 
    return [pI, pX, pZ, pY]
end 
# Options: 
# ChannelType = "Independent" or "Depolarizing" w/ customP = nothing 
# ChannelType = Anything w/ a custom customP function (example being ninexz)


function special1(x; tuple = false, plot = false)
pI = 0.4365075000000005
pZ =  0.4365075000000004
pX = 0.06349250000000005
pY = 0.06349250000000005

    if tuple # this should always be here, do not touch 
        return (pI, pX, pZ, pY)
    end
    if plot # this is to plot different things (for example, smith plots 1-pI instead of pX despite working with pX)
        return 1-pI 
    end 
    return [pI, pX, pZ, pY]
end 


function special2(x; tuple = false, plot = false)
pI = 0.4984659917818757
pZ =  0.4984659917818757
pX = 0.0015340082181245283
pY = 0.0015340082181245283

    if tuple # this should always be here, do not touch 
        return (pI, pX, pZ, pY)
    end
    if plot # this is to plot different things (for example, smith plots 1-pI instead of pX despite working with pX)
        return 1-pI 
    end 
    return [pI, pX, pZ, pY]
end 


function main()
    ChannelType = "special" 
    n_range = 2:1:5
    points = 1
    p_1 = 0.5071504382417463
    q_2 = 0.2959604700198096
    #p_1 = .5
    #pz_range_override = range(.188, 0.1906, length = points)
    #pz_range_override = range(.2447, 0.2447, length = points)
    #pz_range_override = range(0.233, .272, length = points)
    pz_range_override = range(1,1,length = 1)
    #pz_range_override = range(0.230, .233, length = points)
    #pz_range_override = range(0.24414285714285713, 0.24692857142857144, length = points)
    #pz_range_override = range(.1835, 0.188, length = points)
    hashing, base_grid, s_best = envelope_finder(n_range, ChannelType; customP = special, customP1 = special1, customP2 = special2, pz = 1, p_1 = p_1, q_2 = q_2, randomSearch = false, useTrials = false, trials = 7e4, points = points, pz_range_override = pz_range_override)
end

# Run the main function

main()
