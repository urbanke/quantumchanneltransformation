
include("../src/Symplectic.jl")
include("../src/SGS.jl")
include("../env_utils/Channels.jl")

using .Symplectic, .SGS
using QECInduced, .Channels
using Base.Threads
using Plots
using LinearAlgebra
using Random
using Printf 


function all_z_code_check(customP, n, pz) 
    s = n - 1  # Number of rows in the (n-k) × (2n) matrix    
    S = falses(1,2n)
    S[1,(n+1):end] .= true 
    S = Matrix{Bool}(S)
    hb = QECInduced.sweep_hashing_grid(pz_range, customP)
    hb_ind = QECInduced.check_induced_channel(S, pz, customP; sweep=false, threads = threads)
    return hb_ind .- hb
end 

# this is allegedly where the root is for each n (from another program). This just tests to see if the difference is small. 
myDict = Dict(5 => 0.23660000000000037, 16 => 0.22560000000000016, 12 => 0.22980000000000067, 8 => 0.23150000000000048, 6 => 0.23260000000000036, 11 => 0.23160000000000006, 9 => 0.2327000000000008, 14 => 0.22910000000000075, 3 => 0.24040000000000106, 7 => 0.23430000000000062, 4 => 0.23380000000000023, 13 => 0.23100000000000054, 15 => 0.22560000000000113, 2 => 0.2340000000000002, 10 => 0.2305000000000006)
hbDict = Dict() 
points = 1 
for n in 2:16
    p = myDict[n] 
    println(p) 
    pz_range_override = range(p,p,length = points)
    hb = all_z_code_check(Channels.ninexz, n, pz) 
    hbDict[n] = hb 
    println(hb)
end 

println(hbDict)

print("[")
for i in 2:15
    print(myDict[i], ", ")
end 
println(myDict[16], "]")






