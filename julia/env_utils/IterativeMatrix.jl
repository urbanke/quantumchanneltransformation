module IterativeMatrix
export iterate_standard_block_matrices_optimized, All_Codes_DFS_parallel, All_Codes_DFS, count_standard_block_matrices


include("../src/Symplectic.jl")
include("../src/SGS.jl")
include("../env_utils/EnvelopeUtil.jl")

using .Symplectic, .SGS, .EnvelopeUtil
using QECInduced
using Base.Threads
using Plots
using LinearAlgebra
using Random




# Utilities
identity_matrix(m::Int) = begin
    M = zeros(Int, m, m)
    for i in 1:m
        M[i,i] = 1
    end
    M
end

zero_matrix(m::Int, n::Int) = zeros(Int, m, n)


"""
    each_binary_matrix_gray_uint(m,n)

Gray-code order over m×n matrices (requires m*n <= (8*sizeof(UInt))).
Yields (E::Matrix{Int}, i::Int, j::Int) where (i,j) is the flipped bit;
first yield is all-zeros with i=j=0.
"""
function each_binary_matrix_gray_uint(m::Int, n::Int)
    Channel{Tuple{Matrix{Int},Int,Int}}() do ch
        if m == 0 || n == 0
            put!(ch, (zeros(Int, m, n), 0, 0))
            return
        end
        L = m*n
        @assert L <= 8*sizeof(UInt) "m*n too large for UInt Gray iterator"

        buf = zeros(Int, m, n)
        put!(ch, (copy(buf), 0, 0))

        prev_g = UInt(0)
        total = UInt(1) << L

        for t in UInt(1):(total-1)
            g = t ⊻ (t >> 1)
            diff = g ⊻ prev_g
            bit = trailing_zeros(diff)      # 0-based

            p = Int(bit) + 1
            j = ((p - 1) % n) + 1
            i = ((p - 1) ÷ n) + 1

            buf[i, j] ⊻= 1
            put!(ch, (copy(buf), i, j))

            prev_g = g
        end
    end
end


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


# Matrix Mult Helpers 


mul2(X,Y) = (X*Y) .% 2


@inline function xor_row_with_row!(P::Matrix{Int}, i::Int, A2T::Matrix{Int}, t::Int)
    @inbounds for j in 1:size(P,2)
        P[i,j] ⊻= A2T[t,j]
    end
end

@inline function xor_into!(Dst::Matrix{Int}, X::Matrix{Int}, Y::Matrix{Int})
    @assert size(Dst) == size(X) == size(Y)
    @inbounds for i in 1:size(Dst,1), j in 1:size(Dst,2)
        Dst[i,j] = X[i,j] ⊻ Y[i,j]
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


"""
    count_standard_block_matrices_constrained(n, k; r=nothing)

Return total count of emitted matrices.

If r === nothing: sums over all r = 0..(n-k)
If r is specified: returns count for that specific r value

Total per r: T(r) = 2*(rx*rz + rx*k) + rz*k + div((rx*(rx+1)),2) 
"""
function count_standard_block_matrices_constrained(n::Int, k::Int; r::Union{Int,Nothing}=nothing)
    @assert 0 ≤ k ≤ n "Require 0 ≤ k ≤ n"
    s = n - k
    
    r_range = if r === nothing
        0:s
    else
        @assert 0 ≤ r ≤ s "Require 0 ≤ r ≤ n-k"
        r:r
    end
    
    total = 0
    for rx in r_range
        # bits = number of variable entries for this r
        rz = n - k - rx
        # bits = number of variable entries for this r
        bits = 2*(rx*rz + rx*k) + rz*k + div((rx*(rx+1)),2) 
        total += 2^bits
    end
    return total
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
# orthogonality checking
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

"""
    each_B_sym(S::Matrix{Int})

Iterate all B (same size as BBT) such that B ⊻ B' == S over GF(2).
Free bits: r(r+1)/2 (upper triangle including diagonal).
Only have to do upper triangular (and then the lower triangular is forced since BBT = B + B^T)
"""
function each_B_sym(BBT::Matrix{Int})
    r, c = size(BBT)
    Channel{Matrix{Int}}() do ch
        # choose all upper-tri (including diag) bits via a buffer U
        U = zeros(Int, r, r)

        # linear index over upper triangle positions (i<=j)
        upp = [(i,j) for i in 1:r for j in i:r]
        L = length(upp)

        function fillpos(p::Int)
            if p > L
                B = zeros(Int, r, r)
                # set upper+diag
                for (i,j) in upp
                    B[i,j] = U[i,j]
                end
                # force lower from constraint: BBT[i,j] = B[i,j] ⊻ B[j,i]
                for i in 2:r, j in 1:i-1
                    B[i,j] = BBT[i,j] ⊻ B[j,i]
                end
                put!(ch, B)
                return
            end
            (i,j) = upp[p]
            U[i,j] = 0; fillpos(p+1)
            U[i,j] = 1; fillpos(p+1)
        end

        fillpos(1)
    end
end
# -------------------------------
# Main enumerator with incremental checking
# -------------------------------

"""
    iterate_standard_block_matrices_optimized_constraints(n, k; r=nothing, visit=nothing)

Optimized version with orthogonality constraints. 
D = A1 + A2E^T 
B+B^T = A1C1^T + A2C2^T + C1A1^T + C2A2^T
Guarantees commutativity
"""
function iterate_standard_block_matrices_optimized_constraints(n::Int, k::Int; r::Union{Int,Nothing}=nothing, visit=nothing)
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
                A1T = copy(A1')                   # r2×r1 (materialize)

                for A2 in each_binary_matrix(r1, k)
                    A2T = copy(A2')                   # k×r1 (materialize)

                    for C1 in each_binary_matrix(r1, r2)
                        C1T = copy(C1')                   # r2×r1 (materialize)
                        m1 = mul2(A1,C1T) .⊻ mul2(C1,A1T)

                        for C2 in each_binary_matrix(r1, k)
                            C2T = copy(C2')                   # k×r1 (materialize)
                            m2 = mul2(A2,C2T) .⊻ mul2(C2,A2T) 
                            BBT = m2 .⊻ m1  # Construct (B+B^T) explicitly 

                            P   = zeros(Int, r2, r1)          # running value of E*A2' mod 2
                            D   = zeros(Int, r2, r1)          # reuse buffer

                            for (E, fi, fj) in each_binary_matrix_gray_uint(r2, k)
                                # Update P incrementally (skip first all-zero emit)
                                if fi != 0
                                    # E[fi,fj] flipped, so row fi of P toggles by row fj of A2T
                                    xor_row_with_row!(P, fi, A2T, fj)
                                end

                                # Now D = A1' ⊻ P
                                xor_into!(D, A1T, P) 

                                for B in each_B_sym(BBT)
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


"""
    All_Codes_DFS(channelParamFunc, n, k; pz=nothing, r_specific=nothing)

Adapted version to work with the standard block matrix enumeration.

Parameters:
- channelParamFunc: The type of quantum channel
- n: Code parameter n
- k: Code parameter k (s = n-k is the number of rows)
- pz: Optional depolarization parameter (computed if not provided)
- r_specific: Optional specific value of r to test (tests all r if nothing)

Returns:
- hb_best: Best value found
- S_best: Best stabilizer matrix found
- r_best: The r value that gave the best result
"""
function All_Codes_DFS(channelParamFunc, n, k, p_range; r_specific=nothing,  newBest = nothing, threads = Threads.nthreads(), trials = 1e6, useTrials = false, concated = nothing, placement = "inner") 
    s = n - k  # Number of rows in the (n-k) × (2n) matrix
    
    # Initialize best trackers for each grid point
    points = length(p_range)
    S_best = [falses(s, 2n) for _ in 1:points]  # Best matrix at each grid point
    r_best = fill(-1, points)  # Best r value at each grid point
    
    if newBest === nothing 
        hb_best = QECInduced.sweep_hashing_grid(p_range, channelParamFunc)
    else 
        hb_best = newBest
    end 
    println(hb_best)
    println("=" ^ 70)
    println("Generating binary matrices ($s × $(2*n)) in standard block form")

    if r_specific !== nothing
        println("Parameters: n=$n, k=$k, s=$s, r=$r_specific, grid_points=$points")
    else
        println("Parameters: n=$n, k=$k, s=$s, grid_points=$points")
        println("Testing all r values from 0 to $s")
    end
    println("pz range: [$(p_range[1]), $(p_range[end])]")
    println("=" ^ 70)
    
    # Calculate total possible without constraints
    total_possible_no_constraints = count_standard_block_matrices_constrained(n, k; r=r_specific)
    println("\nTotal matrices without orthogonality constraints: $total_possible_no_constraints")
    
    count = 0
    count_by_r = Dict{Int,Int}()
    last_print_count = 0
    print_interval = max(1, div(total_possible_no_constraints, 20))  # Print ~20 times
    
    println("\nStarting enumeration with orthogonality constraints...\n")
    
    # Use the optimized iterator with orthogonality checking
    for info in iterate_standard_block_matrices_optimized_constraints(n, k; r=r_specific)
        count += 1
        r_val = info.r
        count_by_r[r_val] = get(count_by_r, r_val, 0) + 1
        
        # Convert to Bool matrix
        S = Matrix{Bool}(info.M)
        
        if !isnothing(concated) 
            if placement == "inner"
                S = concat_stabilizers_bool(S, concated)
            else 
                S = concat_stabilizers_bool(concated, S)
            end
        end
        # Check the induced channel at all grid points
        hb_grid = QECInduced.check_induced_channel(S, 0, channelParamFunc; sweep=true, ps=p_range, threads = threads)
        # Find which grid points improved
        improved_indices = findall(hb_grid .> (hb_best .+ 1e-14)) # something slightly bigger than eps to account for numerical error

        
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
                println("  Point $idx: pz=$(round(p_range[idx], digits=4)), hb=$(round(hb_best[idx], digits=6))")
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
    
    println("\nBreakdown by r:")
    for r_val in sort(collect(keys(count_by_r)))
        println("  r=$r_val: $(count_by_r[r_val]) matrices checked")
    end
    return hb_best, S_best#, r_best
end




"""
    All_Codes_DFS_parallel(channelParamFunc, n, k; pz=nothing, use_threads=true)

Parallel version that searches all r values independently using Julia threads.
Can be more efficient since different r values can be checked in parallel.

To use threading, start Julia with: `julia -t auto` or `julia -t 8` (for 8 threads)
Check available threads with: `Threads.nthreads()`
"""

function All_Codes_DFS_parallel(channelParamFunc, n, k, p_range; use_threads=true, newBest = nothing, trials = 1e6, useTrials = false, concated = nothing, placement = "inner")  
    s = n - k
    points = length(p_range)
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
    s_best = [falses(s, 2n) for _ in 1:points]  # Best matrix at each grid point
    
    if use_threads && n_threads > 1
        # Parallel execution using threads
        println("\nStarting parallel search across $n_r r values using $n_threads threads...")
        
        Threads.@threads for i in 1:n_r
            r_val = r_values[i]
            println("Thread $(Threads.threadid()): Starting r=$r_val")
            
            hb, S = All_Codes_DFS(channelParamFunc, n, k, p_range; r_specific=r_val, newBest = newBest, threads = 0, trials = trials, useTrials = useTrials, concated = concated, placement = placement)
            results[i] = (r=r_val, hb=hb, S=S)
            
            println("Thread $(Threads.threadid()): Completed r=$r_val, hb=$hb")
        end
    else
        # Sequential execution (fallback)
        println("\nRunning sequential search (no threading)...")
        for i in 1:n_r
            r_val = r_values[i]
            println("\n--- Starting search for r=$r_val ---")
            
            hb, S = All_Codes_DFS(channelParamFunc, n, k, p_range; r_specific=r_val, newBest = newBest, trials = trials, useTrials = useTrials, concated = concated, placement = placement)
            results[i] = (r=r_val, hb=hb, S=S)
        end
    end
    total_best = QECInduced.sweep_hashing_grid(p_range, channelParamFunc)
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



# ============================================
# Test Functs
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






end # module