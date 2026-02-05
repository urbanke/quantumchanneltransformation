module KernelUtil
export repitition_code_check, check_threading_setup, concat_stabilizers_bool, printCodes, printCodesSlurm, h

include("../src/Symplectic.jl")
include("../src/SGS.jl")


using .Symplectic, .SGS
using QECInduced
using Base.Threads
using Plots
using LinearAlgebra
using Random




function smith_rep_maker(n)
    S = falses(n-1,2n)
    for i in 1:(n-1)
        S[i,[i+n,2n]] .= true 
    end 
    return S
end 


function repitition_code_check(channelParamFunc, n, p_range; newBest = nothing, threads = Threads.nthreads(),  concated = nothing, placement = "inner") 
    s = n - 1  # Number of rows in the (n-k) × (2n) matrix
    
    # Initialize best trackers for each grid point
    points = length(p_range)
    S_best = [falses(s, 2n) for _ in 1:points]  # Best matrix at each grid point
    r_best = fill(-1, points)  # Best r value at each grid point

    if newBest === nothing 
        hb_best = QECInduced.sweep_hashing_grid(p_range, channelParamFunc)
    else 
        hb_best = newBest
    end 
    
    # just checking the smith reptition matrix 
    mat = smith_rep_maker(n) 
    
    # Convert to Bool matrix
    S = Matrix{Bool}(mat)
    
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
            println("  Point $idx: pz=$(round(p_range[idx], digits=4)), hb=$(round(hb_best[idx], digits=6))")
        end
        println("\nS_best (showing first improved point) =")
        println(Symplectic.build_from_bits(S_best[improved_indices[1]]))
        println("=" ^ 70 * "\n")
    end

    return hb_best, S_best
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



function printCodes(base_grid, points, p_range, s_best, hashing, FileName, slurm)
    if slurm 
        printCodesSlurm(base_grid, points, p_range, s_best, hashing)
        return 
    else 
        open(FileName*".txt", "w") do file
            for i in 1:points
                if base_grid[i] > hashing[i]
                    s_best_point = Symplectic.build_from_bits(s_best[i])
                    write(file, "Point: $(p_range[i])\n")
                    write(file, "Induced Hashing Bound: $(base_grid[i])\n")
                    write(file, "S Matrix:\n")
                    write(file, join(string.(s_best_point), " ") * "\n\n")
                end
            end
            write(file, string(base_grid) * "\n")
            write(file, "[")
            for i in 1:(length(p_range) - 1) 
                write(file, string(p_range[i]) * ",")
            end 
            write(file, string(p_range[end]) * "]")
        end
    end 
end 

function printCodesSlurm(base_grid, points, p_range, s_best, hashing)
        for i in 1:points
            if base_grid[i] > hashing[i]
                s_best_point = Symplectic.build_from_bits(s_best[i])
                println("Point: $(p_range[i])\n")
                println("Induced Hashing Bound: $(base_grid[i])\n")
                println("S Matrix:\n")
                println(join(string.(s_best_point), " ") * "\n\n")
            end
        end
        println(string(base_grid) * "\n")
        println("[")
        for i in 1:(length(p_range) - 1)
            println(string(p_range[i]) * ",")
        end
        println(string(p_range[end]) * "]")
end

function printCodesAllCodes(H_all, S_all, p, c, n,k,r,FileName)
    sortedH = sortperm(H_all, rev = true)
    orderedCodes = S_all[sortedH]
    orderedH = H_all[sortedH]
    points = length(H_all)
    open(FileName*".txt", "w") do file
        write(file, "Code Search: ($n,$k,$r) \n")
        write(file, "Hashing Bound: $c \n")
        write(file, "Channel P:"*p*"\n")
        for i in 1:points
                s = Symplectic.build_from_bits(orderedCodes[i])
                write(file, "Induced Hashing Bound: $(orderedH[i])\n")
                write(file, "S Matrix:\n")
                write(file, join(string.(s), " ") * "\n\n")
        end
    end 
end 

function h(p)
    return -p*log2(p) 
end 



end 
