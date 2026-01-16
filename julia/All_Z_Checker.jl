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
using Printf 


function all_z_code_check(ChannelType, n; pz=nothing, points=15, customP=nothing, δ = .3, newBest = nothing, threads = Threads.nthreads(), pz_range_override = nothing, concated = nothing, placement = "inner") 
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
    S = falses(1,2n)
    S[1,(n+1):end] .= true 

    # Convert to Bool matrix
    S = Matrix{Bool}(S)
    
    # Check the induced channel at all grid points
    hb_grid = QECInduced.check_induced_channel(S, pz; ChannelType=ChannelType, sweep=true, ps=pz_range, customP=customP, threads = threads)
    # Find which grid points improved
    improved_indices = findall(hb_grid .> (hb_best .+ eps()))

    dif = copy(hb_grid)
    # Update best for each improved point
    if !isempty(improved_indices)
        for idx in improved_indices
            dif .-= hb_best
            hb_best[idx] = hb_grid[idx]
            S_best[idx] = copy(S)
        end
        
        println("\n" * "=" ^ 70)
        println("$n Length All Z Code Improved!")
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


function envelope_finder(n_range, ChannelType; pz = nothing, customP = nothing, points = 15, δ = .3, randomSearch = false, useTrials = false, trials = 1e7, rng = MersenneTwister(2025), pz_range_override = nothing, concated = nothing, placement = "inner", lowerrate = 1, upperate = 1) 
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


    hashing = QECInduced.sweep_hashing_grid(pz_range, ChannelType; customP = customP)
    best_grid = copy(hashing)
    println(hashing)
    println(length(hashing))
    println("HASHING")
    base_grid = copy(hashing)
    s_best = Vector{Any}(undef, points)
    elapsed_time = @elapsed begin
        for n in n_range
            # do this first - check if smiths codes are better 
            elapsed_time_internal = @elapsed begin
                best_grid, S_grid = all_z_code_check(ChannelType, n; pz = pz, points= points, customP= customP, δ = δ, newBest = best_grid, pz_range_override = pz_range, concated = nothing, placement = placement) 
                improve_indices = findall(best_grid .> base_grid)
                s_best[improve_indices] = S_grid[improve_indices]
                base_grid = max.(base_grid,best_grid)
            end
            println("Elapsed time: $elapsed_time_internal seconds") 
            printCodes(base_grid, points, pz_range, s_best, hashing, ChannelType)
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
 pI = 0.7277731462389959
 pX = 0.27216170306033616
 pZ = 3.2575350335105956e-5
 pY = 3.2575350335105956e-5

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

function main()
    ChannelType = "SMALL_P_SKEW" 
    n_range = 1:1:14
    points = 15
    concated = nothing# Bool[0 0 0 0 0 0 1 1 1 1 1 1] #nothing # Bool[0 0 0 0 0 1 0 0 0 1; 0 0 0 0 0 0 1 0 0 1; 0 0 0 0 0 0 0 1 0 1; 0 0 0 0 0 0 0 0 1 1]
    placement = "outer"
    #pz_range_override = range(.188, 0.1906, length = points)
    #pz_range_override = range(.2447, 0.2447, length = points)
    pz_range_override = range(0.225, .23, length = points)
    #pz_range_override = range(0.1904775,0.1904775,length = 1)
    #pz_range_override = range(0.230, .233, length = points)
    #pz_range_override = range(0.24414285714285713, 0.24692857142857144, length = points)
    #pz_range_override = range(.1835, 0.188, length = points)
    #pz_range_override = range(.1893, 0.1904775, length = points)
    #pz_range_override = range(0.18889285714285714, 0.19022857142857144, length = points)
    lowerrate = 1
    upperate = 1
    hashing, base_grid, s_best = envelope_finder(n_range, ChannelType; customP = ninexz, pz = nothing, randomSearch = true, useTrials = true, trials = 2^15 + 1, points = points, pz_range_override = pz_range_override, concated = concated, placement = placement, lowerrate = lowerrate, upperate = upperate)
end

# Run the main function

main()
