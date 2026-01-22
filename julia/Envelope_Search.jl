# ============================================
# Enumerate all binary (n-k) × (2n) matrices
# in the standard sub-block form, for all r
# ============================================
include("src/Symplectic.jl")
include("src/SGS.jl")
include("env_utils/EnvelopeUtil.jl")

include("env_utils/Isotropic.jl")
include("env_utils/IterativeMatrix.jl")
include("env_utils/Channels.jl")

using .Symplectic, .SGS,  .EnvelopeUtil, .Isotropic, .IterativeMatrix, .Channels
using QECInduced
using Base.Threads
using Plots
using LinearAlgebra
using Random





function envelope_finder(n_range, channelParamFunc; pz = nothing, points = 15, δ = .3, randomSearch = false, useTrials = false, trials = 1e7, rng = MersenneTwister(2025), pz_range_override = nothing, concated = nothing, placement = "inner", lowerrate = 1, upperate = 1, FileName = "hashing_bound_envelope") 
    if pz === nothing 
        pz = findZeroRate(f, 0, 0.5, channelParamFunc; maxiter=1000)
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


    hashing = QECInduced.sweep_hashing_grid(pz_range, channelParamFunc)
    best_grid = copy(hashing)
    println(hashing)
    println(length(hashing))
    println("HASHING")
    base_grid = copy(hashing)
    s_best = Vector{Any}(undef, points)
    elapsed_time = @elapsed begin
        for n in n_range
            # we know the smiths codes are good, so we should guarantee that we check them.  
            best_grid, S_grid = repitition_code_check(channelParamFunc, n; pz = pz, points= points, δ = δ, newBest = best_grid, pz_range_override = pz_range, concated = nothing, placement = placement) 
            improve_indices = findall(best_grid .> base_grid)
            s_best[improve_indices] = S_grid[improve_indices]
            base_grid = max.(base_grid,best_grid)
            if !isnothing(concated) #should also check just concatenatng it with the smith code (same idea as above)
                best_grid, S_grid = repitition_code_check(channelParamFunc, n; pz = pz, points= points, δ = δ, newBest = best_grid, pz_range_override = pz_range, concated = concated, placement = placement)
                improve_indices = findall(best_grid .> base_grid)
                s_best[improve_indices] = S_grid[improve_indices]
                base_grid = max.(base_grid,best_grid)
            end
            lowerk = Int(min(max(ceil(n*lowerrate), 1) , n-1))
            higherk = Int(min(floor(n*upperate - eps()), n-1)) 
            for k in lowerk:higherk
                elapsed_time_internal = @elapsed begin
                    for r in 0:0
                        base_trials = count_standard_block_matrices(n, k; r) 
                        if base_trials <= 0 
                            base_trials = trials 
                        end 
                        if randomSearch && (base_trials ≥ trials)
                            println("Using Random Search")
                            best_grid, S_grid = All_Codes_Random_SGS(channelParamFunc, n, k, r; pz = pz, points = points, δ = δ, newBest = best_grid, trials = trials, pz_range_override = pz_range, concated = concated, placement = placement, rng = rng)
                            improve_indices = findall(best_grid .> base_grid)
                            s_best[improve_indices] = S_grid[improve_indices]
                            base_grid = max.(base_grid,best_grid)
                        else 
                            println("Using Iterative Search")
                            best_grid, S_grid = All_Codes_DFS(channelParamFunc, n, k; threads = 0, r_specific = r, pz = pz, points = points, δ = δ, newBest = best_grid, trials = trials, pz_range_override = pz_range, concated = concated, placement = placement)
                            improve_indices = findall(best_grid .> base_grid)
                            s_best[improve_indices] = S_grid[improve_indices]
                            base_grid = max.(base_grid,best_grid)
                        end
                    end
                end  
                println("Elapsed time: $elapsed_time_internal seconds") 
                printCodes(base_grid, points, pz_range, s_best, hashing, FileName)
            end
         end
    end
    println("Total time: $elapsed_time seconds")
    p_plot = zeros(points)
    for i in 1:points
        p_plot[i] = channelParamFunc(pz_range[i]; plot = true)
    end 
    println("p: ", pz_range)
    plt = plot(
        p_plot, hashing;
        label = "Original channel (per-qubit 1 - H(p))",
        xlabel = "probability p",
        ylabel = "Hashing bound",
        title = "Hashing bounds vs p",
        marker = :circle,
        linewidth = 2,
    )

    plot!(plt, p_plot, base_grid; label = "Induced (per-syndrome conditional entropy)", marker = :square, linewidth = 2)

    # Save figure (and print the path)
    outfile = FileName*".png"
    savefig(plt, outfile)
    println("Saved plot to $(outfile)")
    return hashing, base_grid, s_best
end



function main()
    n_range = 2:1:14 # The range of n searched over 
    trials = 2^15 + 1 # This is the number of trials. If there are more possiblem matrices than trials, do random search; else, do DFS. 
    # Warning: This trials is searched for each (n,k,r) triple. So you may want a fewer trials than you think, because it will do it multiple times for each (n,k) tuple 
    channelParamFunc = Channels.Independent_Skewed_X_Nine # The type of channel searched over - see env_utils/Channels.jl for the options 
    p_range = range(0.24, .25, length = 15) # The range of $p$ searched over 
    points = length(p_range) # the number of points searched over 
    concated = nothing # what the code is concatenated with. Options are nothing or a code constructed with Bool[] (see below for examples)
    # Bool[0 0 0 0 0 0 1 1 1 1 1 1] #nothing # Bool[0 0 0 0 0 1 0 0 0 1; 0 0 0 0 0 0 1 0 0 1; 0 0 0 0 0 0 0 1 0 1; 0 0 0 0 0 0 0 0 1 1]
    placement = "outer" # Where the concatenated code is. Options are inner or outer (does nothing if concated = nothing)
    lowerrate = 0 # This is the lower range of k that will be searched, i.e. floor(n*lowerrate) (will ensure that k > 0) 
    upperate = 1 # This is the upper range of k that will be searched, i.e. ceil(n*upperate) (will ensure that k < n) 
    hashing, base_grid, s_best = envelope_finder(n_range, channelParamFunc; pz = nothing, randomSearch = true, trials = 2^15 + 1, 
                                                points = points, pz_range_override = p_range, concated = concated,
                                                 placement = placement, lowerrate = lowerrate, upperate = upperate)
end

# Run the main function

main()
