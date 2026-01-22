
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
    return hb_grid .- hb_best
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


myDict = Dict(5 => 0.23660000000000037, 16 => 0.22560000000000016, 12 => 0.22980000000000067, 8 => 0.23150000000000048, 6 => 0.23260000000000036, 11 => 0.23160000000000006, 9 => 0.2327000000000008, 14 => 0.22910000000000075, 3 => 0.24040000000000106, 7 => 0.23430000000000062, 4 => 0.23380000000000023, 13 => 0.23100000000000054, 15 => 0.22560000000000113, 2 => 0.2340000000000002, 10 => 0.2305000000000006)
hbDict = Dict() 
points = 1 
for n in 2:16
    p = myDict[n] 
    println(p) 
    pz_range_override = range(p,p,length = points)
    hb = all_z_code_check("Ignore", n; pz = 0, points= points, customP= ninexz, pz_range_override = pz_range_override) 
    hbDict[n] = hb
    println(hb)
end 

println(hbDict)

print("[")
for i in 2:15
    print(myDict[i], ", ")
end 
println(myDict[16], "]")











