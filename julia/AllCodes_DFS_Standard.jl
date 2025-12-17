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

# Utilities
identity_matrix(m::Int) = begin
    M = zeros(Int, m, m)
    for i in 1:m
        M[i,i] = 1
    end
    M
end

zero_matrix(m::Int, n::Int) = zeros(Int, m, n)

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

# -------------------------------
# Main enumerator (lazy) with pruning
# -------------------------------
"""
    iterate_standard_block_matrices(n, k; r=nothing, visit=nothing, check_orthogonality=true)

Enumerate all binary matrices of size (n-k) × (2n) in the standard sub-block form.

If `r === nothing`, iterates over all r = 0..(n-k).
If `r` is specified, only generates matrices for that specific value of r.

If `check_orthogonality=true`, prunes the search tree by checking symplectic orthogonality
constraints as blocks are built up.

If `visit === nothing`, returns a `Channel{NamedTuple}` yielding:
  (M = Matrix{Int}, r = Int,
   blocks = (A1, A2, B, C1, C2, D, E))

If `visit` is a function `(info)->nothing`, it is called for each matrix
and the function returns `nothing`.
"""
function iterate_standard_block_matrices(n::Int, k::Int; r::Union{Int,Nothing}=nothing, visit=nothing, check_orthogonality::Bool=true)
    @assert 0 ≤ k ≤ n "Require 0 ≤ k ≤ n"
    s = n - k
    
    # Determine range of r values
    r_range = if r === nothing
        0:s
    else
        @assert 0 ≤ r ≤ s "Require 0 ≤ r ≤ n-k"
        r:r  # Single value range
    end

    function drive(ch::Union{Channel,Nothing})
        for r_val in r_range
            r1 = r_val
            r2 = s - r_val

            # Build blocks incrementally with orthogonality checks
            
            for A1 in each_binary_matrix(r1, r2)      # r × (s-r)
                for A2 in each_binary_matrix(r1, k)       # r × k
                    # After A1, A2: we have the top-left block
                    # Left top = [I_r A1 A2]
                    I_r = identity_matrix(r1)
                    left_top = hcat(I_r, A1, A2)
                    
                    for B in each_binary_matrix(r1, r1)      # r × r
                        for C1 in each_binary_matrix(r1, r2)      # r × (s-r)
                            for C2 in each_binary_matrix(r1, k)       # r × k
                                # Right top = [B C1 C2]
                                right_top = hcat(B, C1, C2)
                                
                                # Check orthogonality of top rows (only among themselves so far)
                                if check_orthogonality && !check_partial_matrix_orthogonality(left_top, right_top)
                                    # Top rows fail orthogonality - skip all D, E combinations
                                    continue
                                end
                                
                                for D in each_binary_matrix(r2, r1)      # (s-r) × r
                                    for E in each_binary_matrix(r2, k)       # (s-r) × k
                                        # Now check full matrix orthogonality
                                        if check_orthogonality
                                            # Bottom left = [0 0 0]
                                            # Bottom right = [D I_(s-r) E]
                                            left_bottom = hcat(zero_matrix(r2, r1), zero_matrix(r2, r2), zero_matrix(r2, k))
                                            I_r2 = identity_matrix(r2)
                                            right_bottom = hcat(D, I_r2, E)
                                            
                                            # Build full left and right
                                            left_full = vcat(left_top, left_bottom)
                                            right_full = vcat(right_top, right_bottom)
                                            
                                            if !check_partial_matrix_orthogonality(left_full, right_full)
                                                # Full matrix fails - skip this combination
                                                continue
                                            end
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
function All_Codes_DFS(ChannelType, n, k; pz=nothing, r_specific=nothing)
    s = n - k  # Number of rows in the (n-k) × (2n) matrix
    
    hb_best = -1.0e9 # close to -inf so that anything beats it 
    S_best = falses(s, 2n)
    r_best = -1
    
    # Compute pz if not provided
    if pz === nothing 
        pz = findZeroRate(f, 0, 0.5; maxiter=1000, ChannelType=ChannelType)
    end 
    
    println("=" ^ 70)
    println("Generating binary matrices ($s × $(2*n)) in standard block form")
    println("Parameters: n=$n, k=$k, s=$s")
    if r_specific !== nothing
        println("Testing only r=$r_specific")
    else
        println("Testing all r values from 0 to $s")
    end
    println("=" ^ 70)
    
    # Calculate total possible without constraints
    total_possible_no_constraints = count_standard_block_matrices(n, k; r=r_specific)
    println("\nTotal matrices without orthogonality constraints: $total_possible_no_constraints")
    
    count = 0
    count_by_r = Dict{Int,Int}()
    last_print_count = 0
    print_interval = max(1, div(total_possible_no_constraints, 20))  # Print ~20 times
    
    println("\nStarting enumeration with orthogonality constraints...\n")
    
    # Use the optimized iterator with orthogonality checking
    for info in iterate_standard_block_matrices_optimized(n, k; r=r_specific)
        count += 1
        r_val = info.r
        count_by_r[r_val] = get(count_by_r, r_val, 0) + 1
        
        # Convert to Bool matrix (your code expects Bool)
        S = Matrix{Bool}(info.M)
        
        # Check the induced channel
        hb_temp = QECInduced.check_induced_channel(S, pz; ChannelType=ChannelType)
        
        # Update best if improved
        if hb_temp >= hb_best
            hb_best = hb_temp
            S_best = copy(S)
            r_best = r_val
            
            println("\n" * "=" ^ 70)
            println("NEW BEST FOUND! (Matrix #$count, r=$r_val)")
            println("pz = $pz")
            println("hb_best = $hb_best")
            println("S_best =")
            println(Symplectic.build_from_bits(S_best))
            println("=" ^ 70 * "\n")
        end
        
        # Periodic progress updates
        if count - last_print_count >= print_interval
            println("Progress: $count matrices checked (r=$r_val, best_so_far=$hb_best)")
            last_print_count = count
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
    
    println("\nBest result:")
    println("  r_best = $r_best")
    println("  hb_best = $hb_best")
    println("  pz = $pz")
    println("=" ^ 70)
    
    return hb_best, S_best, r_best
end


"""
    All_Codes_DFS_parallel(ChannelType, n, k; pz=nothing)

Parallel version that searches all r values independently.
Can be more efficient since different r values can be checked in parallel.
"""
function All_Codes_DFS_parallel(ChannelType, n, k; pz=nothing)
    s = n - k
    
    if pz === nothing 
        pz = findZeroRate(f, 0, 0.5; maxiter=1000, ChannelType=ChannelType)
    end
    
    println("=" ^ 70)
    println("PARALLEL SEARCH: Testing each r value independently")
    println("Parameters: n=$n, k=$k, s=$s")
    println("=" ^ 70)
    
    # Search each r value
    results = []
    for r_val in 0:s
        println("\n--- Starting search for r=$r_val ---")
        hb, S, r = All_Codes_DFS(ChannelType, n, k; pz=pz, r_specific=r_val)
        push!(results, (r=r_val, hb=hb, S=S))
    end
    
    # Find overall best
    best_idx = argmax([res.hb for res in results])
    best_result = results[best_idx]
    
    println("\n" * "=" ^ 70)
    println("OVERALL BEST ACROSS ALL r VALUES:")
    println("  r_best = $(best_result.r)")
    println("  hb_best = $(best_result.hb)")
    println("=" ^ 70)
    
    return best_result.hb, best_result.S, best_result.r
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


function main()
    n, k = 5, 1  # => s = n-k = 2; matrices are 2 × (2n) = 2 × 6
    ChannelType = "Independent"
    All_Codes_DFS(ChannelType, n, k)
end

# Run the main function
main()