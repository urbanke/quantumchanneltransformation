include("src/Symplectic.jl")
include("src/SGS.jl")

using .Symplectic, .SGS
using QECInduced, .Symplectic, .SGS
using Base.Threads
using Plots
"""
Iterate through all possible binary matrices of size (n, r) one row at a time,
with early termination based on row compatibility checks.

Parameters:
- n: number of rows
- r: number of columns
- verify_fn: function(matrix, current_row) that returns true if rows 1:current_row are compatible

The verify function receives the partial matrix and the index of the row just added.
Return false to skip all matrices that extend this partial configuration.
"""

function iterate_upper_triangular_matrices(n::Int, r::Int, verify_fn::Function)
    nb = div(n,2)
    Channel() do ch
        matrix = zeros(Int, r, n)

        function build_rows(row_idx::Int)
            if row_idx > r
                put!(ch, copy(matrix))
                return
            end


            # Number of columns allowed for this row
            allowed_cols = n - row_idx + 1

            for i in 0:(2^allowed_cols - 1)
                temp = i
                # Fill allowed columns only
                for col in n:-1:row_idx
                    matrix[row_idx, col] = temp & 1
                    temp >>= 1
                end
                # Ensure columns before start_col are zero
                for col in 1:(row_idx - 1)
                    matrix[row_idx, col] = 0
                end

                if verify_fn(matrix, row_idx, nb)
                    build_rows(row_idx + 1)
                end
            end
        end

        build_rows(1)
    end
end


function iterate_binary_matrices_with_check(n::Int, r::Int, verify_fn::Function)
    Channel() do ch
        # Build matrix recursively, row by row
        nb = div(n,2) # binary n (2n is for quaternay)
        matrix = zeros(Int, r, n)
        
        function build_rows(row_idx::Int)
            if row_idx > r
                # All rows filled and verified - yield complete matrix
                put!(ch, copy(matrix))
                return
            end
            
            # Try all 2^n possible values for this row
            for i in 0:(2^n - 1)
                # Fill current row
                temp = i
                for col in n:-1:1
                    matrix[row_idx, col] = temp & 1
                    temp >>= 1
                end
                
                # Verify compatibility with previous rows
                if verify_fn(matrix, row_idx, nb)
                    # If valid, continue to next row
                    build_rows(row_idx + 1)
                end
                # If invalid, skip entire subtree (all extensions of this partial matrix)
            end
        end
        
        build_rows(1)
    end
end

function good_code(matrix, current_row,n)
    S = Matrix{Bool}(matrix[1:current_row,:])
    if current_row == 1
        return true  # First row always valid
    end
    # this checks depth first if it is stabilizing & full rank & canonical 
    # depth first meaning that it only compares some of the matrix (up to current row) 
    return (Symplectic.valid_code(S)) & (SGS.rank_f2(S) == current_row) & (is_canonical(matrix, current_row,n))
end

function is_canonical(matrix, current_row,n) # since both our channels are symmetric [I,X,Z,Y] is the exact same code as [I,Z,X,Y]. 
    # X↔Z flip: swap columns [1:n] with columns [n+1:2n]
    # Compare lexicographically row by row, column by column

    for row in 1:current_row
        for col in 1:(2*n)
            current_bit = matrix[row, col]
            
            # In flipped version, first half and second half are swapped
            if col <= n
                # This is an X bit (first half), in flipped it's a Z bit (from second half)
                flipped_bit = matrix[row, col + n]
            else
                # This is a Z bit (second half), in flipped it's an X bit (from first half)
                flipped_bit = matrix[row, col - n]
            end
            
            # Lexicographic comparison
            if current_bit < flipped_bit
                return true   # Current is smaller, canonical
            elseif current_bit > flipped_bit
                return false  # Flipped is smaller, not canonical
            end
            
            # Bits equal, continue to next column
        end
    end
    
    # Matrices are identical under X↔Z flip, canonical
    return true
end


function All_Codes_DFS(ChannelType, n, k; pz = nothing)
    r = n-k 
    hb_best = -1.0e9 # close to -inf so that anything beats it 
    S_best  = falses(r, 2n)  

    # whatever this is in your codebase
    if pz === nothing 
        pz = findZeroRate(f, 0, 0.5; maxiter=1000, ChannelType=ChannelType)
    end 
    println("Generating binary matrices ($r × $(2*n)) in a depth-first stabilizing approach:\n")
        count = 0
    total_possible = 2^(2*n*r)
    println("Total possible: $total_possible")
    for matrix in iterate_upper_triangular_matrices(2*n, r, good_code)
        count += 1
        S = Matrix{Bool}(matrix)
        hb_temp = QECInduced.check_induced_channel(S, pz; ChannelType = ChannelType)
        if hb_temp >= hb_best
            hb_best = hb_temp
            S_best  = copy(S)
        end

        if count % 2 == 0
            println("Matrix Num ", count)
            println("pz = ", pz)
            println("hb_best = ", hb_best)
            println("S_best = ")
            println(Symplectic.build_from_bits(S_best))
        end
    end
    
    println("Valid matrices found: $count")
    println("Efficiency gain: $(round((1 - count/total_possible)*100, digits=1))% pruned")
    return hb_best, S_best
end

ChannelType = "Independent"


function calc_ent(p, ChannelType)
    if ChannelType == "Independent"
        pc = [(1-p)^2, p-p^2, p-p^2, p^2]
    else # Depolar
        pc = [1-p, p/3, p/3, p/3]
    end 
    h = 0 
    for i in 1:4 
        h -= pc[i]*log2(pc[i])
    end
    return h 
end 




function All_Codes_DFS_envelope(ps, n_range, ChannelType)
    h_best = zeros(length(ps)) 
    for i in 1:length(ps)    
        h_best[i] = calc_ent(ps[i], ChannelType)
    end 
    h_best = 1 .- h_best
    h_old = copy(h_best)
    
    for n in n_range
        for k in 1:n-2
            r = n-k 

            println("Generating binary matrices ($r × $(2*n)) in a depth-first stabilizing approach:\n")
            count = 0
            total_possible = 2^(2*n*r)
            #println("Total possible: $total_possible")
            for matrix in iterate_upper_triangular_matrices(2*n, r, good_code)
                count += 1
                S = Matrix{Bool}(matrix)
                hb_temp = QECInduced.check_induced_channel(S, -1; ChannelType = ChannelType, sweep = true, ps = ps)

                h_best = max.(hb_temp, h_best)
            end
        end
    end
    return h_best, h_old
#    println("Valid matrices found: $count")
#    println("Efficiency gain: $(round((1 - count/total_possible)*100, digits=1))% pruned")

end 



ps = .1:.0001:.15
elapsed_time = @elapsed begin
    h_best, h_old = All_Codes_DFS_envelope(ps, [3,4,5], ChannelType)
end
println("Elapsed time: $elapsed_time seconds") 
println(h_best .- h_old)
plt = plot(
    ps, h_old;
    label = "Original channel (per-qubit 1 - H(p))",
    xlabel = ChannelType * " probability p",
    ylabel = "Hashing bound",
    title = "Hashing bounds vs p",
    linewidth = 2,
)

plot!(plt, ps, h_best; label = "Envelope", linewidth = 2)
outfile = "envelope.png"
savefig(plt, outfile)
println("Saved plot to $(outfile)")


#= 
n = 3
k = 1
elapsed_time = @elapsed begin
    bestH, bestS = All_Codes_DFS(ChannelType, n, k; pz = .3)
end
println("Elapsed time: $elapsed_time seconds") 
println(bestH)
println(bestS)
=#

#=function getEnvelope(ps, n_min, n_max, ChannelType; threads::Int=nthreads())
    best_code_rate_min3_max5 = zeros(length(ps))
    best_k = zeros(Int, length(ps))
    best_n = zeros(Int, length(ps))
    iterators = length(ps)

    Threads.@threads for i in 1:iterators
        p = ps[i]
        h_best = calc_ent(p, ChannelType)
        k_best = -1
        n_best = -1
        for n in [3,5]
            for k in 1:n-2
                println("="^70)
                println(n," ",k)
                h, _ = All_Codes_DFS(ChannelType, n, k; pz = p)
                if h > h_best
                    h_best = h
                    k_best = k
                    n_best = n
                end
            end
        end
        best_code_rate_min3_max5[i] = h_best
        best_k[i] = k_best
        best_n[i] = n_best
    end

    return best_code_rate_min3_max5, best_k, best_n
end

best_code_rate_min3_max5, best_k, best_n = getEnvelope(ps, 3,5, ChannelType;threads = nthreads())

println(best_code_rate_min3_max5)
=# 

