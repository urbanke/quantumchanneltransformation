module IterativeMatrixKernel
export iterate_standard_block_matrices_optimized, All_Codes_DFS_parallel, All_Codes_Iterate, count_standard_block_matrices_constrained


include("../src/Symplectic.jl")
include("../src/SGS.jl")
include("KernelUtil.jl")

using .Symplectic, .SGS, .KernelUtil
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

# Iterate all binary matrices of size m×n (2^(m*n) total)
function rand_binary_matrix(m::Int, n::Int, rng; p = .5)
    return Int.(rand(rng, m, n) .< p)
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
    for rz in r_range
        # bits = number of variable entries for this r
        rx = n - k - rz
        # bits = number of variable entries for this r
        bits = rz*k + rz*k + div(rz*(rz+1),2) + rx*k #+ rx*rz  
        total += 2^bits
    end
    return total
end



function build_S(n::Int, k::Int, s::Int, rz::Int, rx::Int, Cz::AbstractMatrix{<:Integer}, Cx::AbstractMatrix{<:Integer}, Bz::AbstractMatrix{<:Integer}, Ax::AbstractMatrix{<:Integer})
    I_rz = identity_matrix(rz) 
    Z_rz_rx = zero_matrix(rz,rx) 
    #return hcat(I_r1, Z_r1_r2, Cz, Cx, Bz, Ax)
    return hcat(Ax, Z_rz_rx, Cx, I_rz, Bz, Cz) 
end 

function build_S̃(n::Int, k::Int, s::Int, rz::Int, rx::Int, C̃x::AbstractMatrix{<:Integer}, Ãx::AbstractMatrix{<:Integer})
    I_rx = identity_matrix(rx) 
    Z_rx_rz = zero_matrix(rx,rz)
    Z_rx_k = zero_matrix(rx,k)
    Z_rx_rx = zero_matrix(rx,rx)
    #return hcat(Z_r2_r1, I_r2, Z_r2_k, C̃x, Z_r2_r2, Ãx)
    return hcat(Ãx, I_rx, C̃x, Z_rx_rz, Z_rx_rx, Z_rx_k)  
end 

function build_Lz(n::Int, k::Int, s::Int, rz::Int, rx::Int, C̃x::AbstractMatrix{<:Integer}, Cx::AbstractMatrix{<:Integer})
    I_k = identity_matrix(k) 
    Z_k_rz = zero_matrix(k,rz)
    Z_k_rx = zero_matrix(k,rx)
    Z_k_k = zero_matrix(k,k)
    #return hcat(Z_k_r1, Z_k_r2, I_k, Z_k_k, C̃x', Cx')
    return hcat(Cx', Z_k_rx, Z_k_k, Z_k_rz, C̃x', I_k) 
end 


function build_Lx(n::Int, k::Int, s::Int, rz::Int, rx::Int, Cz::AbstractMatrix{<:Integer})
    I_k = identity_matrix(k) 
    Z_k_rz = zero_matrix(k,rz)
    Z_k_rx = zero_matrix(k,rx)
    Z_k_k = zero_matrix(k,k)
    Z_k_rx = zero_matrix(k, rx) 
    #return hcat(Z_k_r1, Z_k_r2, Z_k_k, I_k, Z_k_r2, Cz')
    return hcat(Cz', Z_k_rx, I_k, Z_k_rz, Z_k_rx, Z_k_k)
end 


function build_G(n::Int, k::Int, s::Int, rz::Int, rx::Int)
    Z_rx_rz = zero_matrix(rx,rz)
    Z_rx_rx = zero_matrix(rx,rx)
    Z_rx_k  = zero_matrix(rx,k )
    I_rx   = identity_matrix(rx) 
    return hcat(Z_rx_rz, Z_rx_rx, Z_rx_k, Z_rx_rz, I_rx, Z_rx_k)
end 

function build_G̃(n::Int, k::Int, s::Int, rz::Int, rx::Int)
    Z_rz_rz = zero_matrix(rz,rz)
    Z_rz_rx = zero_matrix(rz,rx)
    Z_rz_k  = zero_matrix(rz,k )
    I_rz   = identity_matrix(rz) 
    return hcat(I_rz, Z_rz_rx, Z_rz_k, Z_rz_rz, Z_rz_rx, Z_rz_k)
end 



function build_full_tableau(n::Int, k::Int, s::Int, rz::Int, rx::Int,
                           Cz::AbstractMatrix{<:Integer},
                           Cx::AbstractMatrix{<:Integer},
                           C̃x::AbstractMatrix{<:Integer},
                           Bz::AbstractMatrix{<:Integer},
                           Ax::AbstractMatrix{<:Integer},
                           Ãx::AbstractMatrix{<:Integer})
    S  =  build_S(n, k, s, rz, rx, Cz, Cx, Bz, Ax) # r1 rows 
    S̃  =  build_S̃(n, k, s, rz, rx, C̃x, Ãx)         # r2 rows
    Lz = build_Lz(n, k, s, rz, rx, C̃x, Cx)         # k rows 
    Lx = build_Lx(n, k, s, rz, rx, Cz)             # k rows 
    G  =  build_G(n, k, s, rz, rx)                 # r2 rows 
    G̃  =  build_G̃(n, k, s, rz, rx)                 # r1 rows
    return Matrix{Bool}(vcat(S, S̃)), Matrix{Bool}(Lz), Matrix{Bool}(Lx), Matrix{Bool}(vcat(G, G̃))          # (n)×(2n)
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


function rand_B_sym(BBT::Matrix{Int}, rng)
    r, c = size(BBT)
    B = rand_binary_matrix(r, r, rng)
    for i in 2:r, j in 1:i-1
        B[i,j] = BBT[i,j] ⊻ B[j,i]
    end
    return B 
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
function iterate_standard_block_matrices_kernel(n::Int, k::Int, trials::Int, matNum::Int; r::Union{Int,Nothing}=nothing, visit=nothing)
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
            rz = r_val
            rx = s - r_val

            for Cz in each_binary_matrix(rz, k) # r1 x k 
                CzT = copy(Cz')                   # k×r1 (materialize)
                for Cx in each_binary_matrix(rz, k) # r x k 
                    CxT = copy(Cx')                   # k×r1 (materialize)
                    Abase = mul2(Cz,CxT) .⊻ mul2(Cx,CzT) # r x r ? 
                    #for Ãx in each_binary_matrix(rx, rz) # r2xr1
                    Ãx = zero_matrix(rx, rz)
                        for C̃x in each_binary_matrix(rx, k) # r2 x k 
                            C̃xT = copy(C̃x')   
                            Bz = Ãx' .⊻ mul2(Cz,C̃xT) # r1 x r2 
                            for Ax in each_B_sym(Abase) # r x r 
                                T = build_full_tableau(n, k, s, rz, rx, Cz, Cx, C̃x, Bz, Ax, Ãx)
                                info = (S=T[1], T=T, r=r_val,blocks=(Cz=Cz, Cx=Cx, C̃x=C̃x, Bz=Bz, Ax=Ax, Ãx=Ãx))# , blocks=(A1=A1, A2=A2, B=B, C1=C1, C2=C2, D=D, E=E)) idk if i need the blocks right now 
                                if visit === nothing
                                    put!(ch, info)
                                else
                                    visit(info)
                                end
                            end
                        end
                    #end
                end
            end
        end
        nothing
    end

    function drive_rd(ch::Union{Channel,Nothing}, trials::Int, rng)
        for r_val in r_range
            for count in 1:trials
                rz = r_val
                rx = s - r_val
                Cz = rand_binary_matrix(rz, k, rng) # r1 x k 
                CzT = copy(Cz')                   # k×r1 (materialize)
                Cx = rand_binary_matrix(rz, k, rng) # r x k 
                CxT = copy(Cx')                   # k×r1 (materialize)
                Abase = mul2(Cz,CxT) .⊻ mul2(Cx,CzT) # r x r ? 
                #Ãx = rand_binary_matrix(rx, rz) # r2xr1
                Ãx = zero_matrix(rx, rz)
                C̃x = rand_binary_matrix(rx, k, rng) # r2 x k 
                C̃xT = copy(C̃x')   
                Bz = Ãx' .⊻ mul2(Cz,C̃xT) # r1 x r2 
                Ax = rand_B_sym(Abase, rng) # r x r \
                T = build_full_tableau(n, k, s, rz, rx, Cz, Cx, C̃x, Bz, Ax, Ãx)
                info = (S=T[1], T=T, r=r_val, blocks=(Cz=Cz, Cx=Cx, C̃x=C̃x, Bz=Bz, Ax=Ax, Ãx=Ãx))
                if visit === nothing
                    put!(ch, info)
                else
                    visit(info)
                end
            end
        end
        nothing
    end
    if (matNum <= trials) && (matNum > 0) # with a large enough power it sign flips
        if visit === nothing
            return Channel{NamedTuple}() do ch
                drive(ch)
            end
        else
            drive(nothing)
            return nothing
        end
    else
        if visit === nothing
            return Channel{NamedTuple}() do ch
                drive_rd(ch, trials,  MersenneTwister(1337))
            end
        else
            drive_rd(nothing, trials, MersenneTwister(1337))
            return nothing
        end
    end 
end







function All_Codes_Iterate(channelParamFunc, n, k, p_range; r_specific=nothing,  newBest = nothing, threads = Threads.nthreads(), trials = 1e6, useTrials = false, concated = nothing, placement = "inner") 
    s = n - k  # Number of rows in the (n-k) × (2n) matrix
    # Initialize best trackers for each grid point
    points = length(p_range)
    S_best = [falses(2n, 2n) for _ in 1:points]  # Best matrix at each grid point
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
    println("p range: [$(p_range[1]), $(p_range[end])]")
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
    for info in iterate_standard_block_matrices_kernel(n, k, trials, total_possible_no_constraints; r=r_specific)
        count += 1
        r_val = info.r
        count_by_r[r_val] = get(count_by_r, r_val, 0) + 1
        # Convert to Bool matrix
        T = info.T
        # Check the induced channel at all grid points
        hb_grid = QECInduced.check_induced_channel(T, 0, channelParamFunc; sweep=true, ps=p_range, threads = threads, FullTableau = true)
        # Find which grid points improved
        improved_indices = findall(hb_grid .> (hb_best .+ 1e-14)) # something slightly bigger than eps to account for numerical error

        
        # Update best for each improved point
        if !isempty(improved_indices)
            for idx in improved_indices
                hb_best[idx] = hb_grid[idx]
                S_best[idx] = copy(info.S)
                r_best[idx] = r_val
            end
            
            println("\n" * "=" ^ 70)
            println("NEW BEST FOUND! (Matrix #$count, r=$r_val)")
            println("Ãx = $((Matrix{Bool}(info.blocks.Ãx)))")
            println("Ax = $((Matrix{Bool}(info.blocks.Ax)))")
            println("Bz = $((Matrix{Bool}(info.blocks.Bz)))")
            println("Cx = $(Matrix{Bool}(info.blocks.Cx)))")
            println("C̃x = $((Matrix{Bool}(info.blocks.C̃x)))")
            println("Cz = $((Matrix{Bool}(info.blocks.Cz)))")

            println("Improved at $(length(improved_indices)) grid point(s): $improved_indices")
            println("\nGrid point details:")
            for idx in improved_indices
                println("  Point $idx: pz=$(round(p_range[idx], digits=4)), hb=$(round(hb_best[idx], digits=6))")
            end
            println("\nS_best (showing first improved point) =")
            for i in 1:4
                println(Symplectic.build_from_bits(T[i]))
            end 
            println("=" ^ 70 * "\n")
        end

        if (count > trials) && useTrials
            break
        end 
    end
    
    println("\n" * "=" ^ 70)
    println("SEARCH COMPLETE")
    println("=" ^ 70)
    
    println("\nBreakdown by r:")
    for r_val in sort(collect(keys(count_by_r)))
        println("  r=$r_val: $(count_by_r[r_val]) matrices checked")
    end
    return hb_best, S_best#, r_best
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