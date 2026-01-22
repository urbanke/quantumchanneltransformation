# 5-qubit code, r = 4; k = 1
# Pauli vector format is (u | v) with length 2n
#  - X on qubit i -> u_i = 1, v_i = 0
#  - Z on qubit i -> u_i = 0, v_i = 1

include("../src/Symplectic.jl")

using QECInduced, .Symplectic


# Create the edge-pair pattern for a given i (Z at i and at n - i + 1)
function edge_pair(n::Int, i::Int)
    @assert 1 ≤ i ≤ n "i must be between 1 and n"
    String([ (p == i || p == n - i + 1) ? 'Z' : 'I' for p in 1:n ])
end

# Build the full pattern vector
function build_pattern(n::Int, k::Int)
    @assert 1 ≤ k ≤ n "k must be between 1 and n"
    rows = String[]
    # First k-1 rows: symmetric Zs moving inward
    for i in 1:k-1
        push!(rows, edge_pair(n, i))
    end
    # Last row: k Zs followed by n-k Is
    push!(rows, repeat("Z", k) * repeat("I", n - k))
    rows
end


function ninexz_maker(rate, n; searchDown = false)
    # find closest rate    
    k = 0 
    if searchDown
        for i in (n-1):-1:(1) 
            if i/n < rate 
                k = i 
                break 
            end 
        end 
    else
        for i in 1:(n-1) 
            if i/n > rate 
                k = i 
                break 
            end 
        end 
    end
    println(k/n)
    s = n-k 
    # make code of the form i found 
    return build_pattern(n,s)
end 

# Choose a code (default: 5-qubit perfect code)
# Stabilizers = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
# Other options you sometimes toggle:
# Stabilizers = ["ZZIIIIIII", "IZZIIIIII", "IIIZZIIII", "IIIIZZIII","IIIIIIZZI","IIIIIIIZZ","IIIXXXXXX","XXXXXXIII"] # 9-qubit Shor
# Stabilizers = ["ZZIZZIZZI", "IZZIZZIZZ", "IIIXXXXXX", "XXXXXXIII"]  # 9-qubit, rate 5/9, Bacon-Shor
# Stabilizers = ["XXI", "IXX"]  # 3-qubit repetition
# Stabilizers = ["ZZI", "IZZ"]  # 3-qubit repetition
# Stabilizers = ["XXIII", "IXXII", "IIXXI", "IIIXX"] # alt 5-qubit repetition flavor
# Stabilizers = ["IYIIX", "IIIIX"]
#Stabilizers = ["ZZIII", "ZIZII","ZIIZI", "ZIIIZ"]
#Stabilizers = ["IIIIIIZ", "IIIIIZI", "IIIIZII", "IIZZZII", "IXZIZZZ"
#= 
Grid point details:
  Point 3: pz=0.236, hb=0.037423

S_best (showing first improved point) =
Any["ZIIIIIIIZ", "IZIIIIIZI", "IIZIIIZII", "IIIZIZIII", "IIIIZZZZZ"]
=#
#7,1 repitition code (Smith Paper)
#Stabilizers = ["ZZIIIII", "ZIZIIII","ZIIZIII", "ZIIIZII", "ZIIIIZI", "ZIIIIIZ"]
#7,1 code found for independent channel (through DFS) 
#Stabilizers = ["IIIIIIZ", "IIIIIZI", "IIIIZII", "IIIZIII", "XZIZZZZ", "XIXZZZZ"]
#7,5 code found for independent channel (through DFS) 
#Stabilizers = ["IIIIIZZ", "IIIIXZI"]
#3,1 code found for independent channel (through DFS) 
CHANNEL = "Independent"
Stabilizers = ["IXX", "ZXI"]
Stabilizers = ["IIIII", "IIIII", "IIIII", "IIIII"]
Stabilizers = ["XIIIX", "IXIIX", "IIXIX", "IIIXX"]
Stabilizers = ["ZIIIZ", "IZIIZ", "IIZZI"]
Stabilizers = ["ZIIIIIZ", "IZIIIIZ", "IIZIIIZ", "IIIZIIZ", "IIIIZIZ", "IIIIIZZ"]
Stabilizers = ["ZIIIIIIIIIZ", "IZIIIIIIIZI", "IIZIIIIIZII", "IIIZIIIZIII", "IIIIZIZIIII", "IIIIIZIIIIZ"]
Stabilizers = ["ZIIIIIIIIIZ", "IZIIIIIIZII", "IIZIIIIIIZI", "IIIZIIIIZII", "IIIIZIIIIIZ", "IIIIIZIIZII", "IIIIIIZIIZI", "IIIIIIIZIIZ"]
Stabilizers = ["ZIIIIIIIIIZ", "IZIIIIIIIIZ", "IIZIIIIIIIZ", "IIIZIIIIIIZ", "IIIIZIIIIIZ", "IIIIIZIIIIZ", "IIIIIIZIIIZ", "IIIIIIIZIIZ", "IIIIIIIIZIZ", "IIIIIIIIIZZ"]
Stabilizers = ["ZIIIZ", "IZIIZ", "IIZZI"]
Stabilizers = ["XIIIIIIIIIZ", "IXIIIIIIIIZ", "IIXIIIIIIIZ", "IIIXIIIIIIZ", "IIIIXIIIIIZ", "IIIIIXIIIIZ", "IIIIIIXIIIZ", "IIIIIIIXIIZ", "IIIIIIIIXIZ", "IIIIIIIIIXZ"]
Stabilizers = ["ZZZZIII", "ZIIIIIZ", "IZIIIZI", "IIIZZZZ"]
Stabilizers = ["ZIIIIIIIIIZ", "IZIIIIIIIZI", "IIZIIIIIZII", "IIIZIIIZIII", "IIIIZIZIIII", "IIIIIZZZZZZ"]
Stabilizers = ["XIIIIIZ", "IXIIIIZ", "IIXIIZI", "IIIXIIZ", "IIIIZIZ", "IIIIIZZ"]
Stabilizers = ["ZIIZII", "IZIIZI", "IIZIIZ"]
Stabilizers = ["ZZ"]
Stabilizers = ["ZIIIIIZ", "IZIIIIZ", "IIZIIIZ", "IIIZIIZ", "IIIIZIZ", "IIIIIZI"]
Stabilizers = ["ZIIIIIZ", "IZIIIIZ", "IIZIIIZ", "IIIZIIZ", "IIIIZIZ", "IIIIIZZ"]
Stabilizers = ["XIZII", "IXIIZ", "IIZIZ", "IIIZZ"]
Stabilizers = ["ZIIIZ", "IZIIZ", "IIZIZ", "IIIZZ"]
Stabilizers = ["ZIIIZ", "IZIIZ", "IIZZI"]
Stabilizers = ["ZIIIIIIIZ", "IZIIIIIIZ", "IIZIIIIZI", "IIIZIIIZI", "IIIIZIZII", "IIIIIZZII"]
#Stabilizers = ["ZIIIIIIIZ", "IZIIIIIIZ", "IIZIIIIIZ", "IIIZIIIIZ", "IIIIZIIIZ", "IIIIIZIIZ", "IIIIIIZIZ", "IIIIIIIZZ"]

println("="^70)

# Channel parameter
#px = 0.11002786443835955
#px = 0.234
px = .2357857142857143 
px = 0.2664285714285714
pz = px/9
#@show px,pz
px1 = 0.24414285714285713 
px1 = 0.233
px2 = 0.24692857142857144
P = range(px1, px2, length = 5000)
for px in P
    p = 0.18892857142857142

#    px = px1 + .00001*j
    #println(px)
    #CHANNEL = "Depolarizing"
    # Single-qubit Pauli channel tuple (pI, pX, pZ, pY)
    if CHANNEL == "Depolarizing"
        # Depolarizing: [1-p, p/3, p/3, p/3]
        p_channel = [1 - p, p/3, p/3, p/3]
    else
        # Independent X/Z flips: [(1-p)^2, p(1-p), p(1-p), p^2]
        p_channel = [(1 - px) * (1 - pz), px * (1 - pz), pz * (1 - px), px * pz] 
    end
    Stabilizers = ["ZZ"]
    S = Symplectic.build_from_stabs(Stabilizers)
    S = Matrix{Bool}(S)
    H, Lx, Lz, G = QECInduced.tableau_from_stabilizers(S)
    #println("\nComputing induced-channel distribution and per-syndrome hashing bound (new definition):")
    hashing_induced_1 = QECInduced.induced_channel_and_hashing_bound(H, Lx, Lz, G, p_channel)
    #println("Induced (per-syndrome) hashing bound returned by kernel: (k - Σ_s p(s) H(p(a',b'|s)))/n")
    #@show hashing_induced_1
    Stabilizers = ["ZZ"]
    S = Symplectic.build_from_stabs(Stabilizers)
    S = Matrix{Bool}(S)
    H, Lx, Lz, G = QECInduced.tableau_from_stabilizers(S)
    #println("\nComputing induced-channel distribution and per-syndrome hashing bound (new definition):")
    hashing_induced_2 = QECInduced.induced_channel_and_hashing_bound(H, Lx, Lz, G, p_channel)
    #println("Induced (per-syndrome) hashing bound returned by kernel: (k - Σ_s p(s) H(p(a',b'|s)))/n")
    #@show hashing_induced_2
    Stabilizers = ["ZIIIIIZ", "IZIIIZI", "IIZIZII", "IIIZZZZ"]
    S = Symplectic.build_from_stabs(Stabilizers)
    S = Matrix{Bool}(S)
    H, Lx, Lz, G = QECInduced.tableau_from_stabilizers(S)
    #println("\nComputing induced-channel distribution and per-syndrome hashing bound (new definition):")
    hashing_induced_3 = QECInduced.induced_channel_and_hashing_bound(H, Lx, Lz, G, p_channel)
    #println("Induced (per-syndrome) hashing bound returned by kernel: (k - Σ_s p(s) H(p(a',b'|s)))/n")
    #@show hashing_induced_3
    if hashing_induced_3 > max(hashing_induced_2, hashing_induced_1) 
        println("yay")
        println(px)
    end
end





