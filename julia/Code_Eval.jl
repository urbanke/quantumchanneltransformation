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


function code_eval(n, k, R, channelParamFunc, p; randomSearch = true, useTrials = false, trials = 1e7, rng = MersenneTwister(2025), concated = nothing, placement = "inner", FileName = "hashing_bound_envelope") 
    c = 1 - sum(h.(channelParamFunc(p)))
    p_print = join(string.(channelParamFunc(p)), " ")
    elapsed_time = @elapsed begin
        H_all = [] 
        S_all = []
        for r = R
            base_trials = count_standard_block_matrices(n, k; r) 
            if base_trials <= 0 
                base_trials = trials + 1
            end 
            if randomSearch && (base_trials > trials)
                println("Using Random Search")
                H_all1, S_all1 = Isotropic.All_Codes_Random_AllCodes(channelParamFunc, n, k, r, p;  trials = trials, concated = concated, placement = placement, rng = rng)
                append!(H_all, H_all1)
                append!(S_all, S_all1)
            else 
                println("Using Iterative Search")
                H_all1, S_all1 = IterativeMatrix.All_Codes_Iterate_AllCodes(channelParamFunc, n, k, p; threads = 0, r_specific = r, trials = trials, concated = concated, placement = placement)
                append!(H_all, H_all1)
                append!(S_all, S_all1)
            end
        end
    end  
    EnvelopeUtil.printCodesAllCodes(H_all, S_all, p_print, c, n,k,R, FileName)
    println("Total time: $elapsed_time seconds")
    return 
end



function main()
    n = 9 # n val
    k = 1 # k val 
    r = [0] # r_x val   
    trials = 4^5# Num of trials  If there are more possiblem matrices than trials, do random search; else, do DFS. 
    channelParamFunc = Channels.Depolarizing # The type of channel searched over - see env_utils/Channels.jl for the options 
    p = .19  # The range of $p$ searched over (do range(p,p, length =1) if you want to search one point)
    concated = nothing # what the code is concatenated with. Options are nothing or a code constructed with Bool[] (see below for examples)
    # Bool[0 0 0 0 0 0 1 1 1 1 1 1] #nothing # Bool[0 0 0 0 0 1 0 0 0 1; 0 0 0 0 0 0 1 0 0 1; 0 0 0 0 0 0 0 1 0 1; 0 0 0 0 0 0 0 0 1 1]
    placement = "outer" # Where the concatenated code is. Options are inner or outer (does nothing if concated = nothing)
    FileName = "All_Codes_Depolarizing_91"
    code_eval(n, k, r, channelParamFunc, p; randomSearch = true, trials = trials, concated = concated,
                                                placement = placement, FileName = FileName)
end

# Run the main function

main()
