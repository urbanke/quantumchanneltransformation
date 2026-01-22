# 5-qubit code, r = 4; k = 1
# Pauli vector format is (u | v) with length 2n
#  - X on qubit i -> u_i = 1, v_i = 0
#  - Z on qubit i -> u_i = 0, v_i = 1

include("src/Symplectic.jl")

using QECInduced, .Symplectic


"""
Concatenate two stabilizer codes given only their stabilizer generators (as Pauli strings).

Inputs:
- Sout :: Vector{String}  -- outer-code stabilizers, each of length n1 (e.g., ["ZZ"])
- Sin  :: Vector{String}  -- inner-code stabilizers, each of length n2 (e.g., ["XIX","IXX"])
- ordering :: Symbol      -- :interleaved (default) or :block
    * :interleaved → index(b,j) = (j-1)*n1 + b  (matches your example)
    * :block       → index(b,j) = (b-1)*n2 + j  (blocks contiguous)

Output:
- Vector{String} of concatenated stabilizers on n1*n2 qubits.

Rule:
- For each inner stabilizer s_in ∈ Sin, produce one row that applies s_in to *every block*.
- For each outer stabilizer s_out ∈ Sout, and each inner position j=1..n2, produce one row that
  places s_out’s letters at the j-th qubits of every block (I elsewhere).
Thus total rows = length(Sin) + length(Sout)*n2.
"""
function concat_stabilizers_strings(Sout::Vector{String}, Sin::Vector{String}; ordering::Symbol=:interleaved)
    @assert !isempty(Sout) && !isempty(Sin) "Both outer and inner stabilizer sets must be nonempty"
    n1 = length(Sout[1])
    n2 = length(Sin[1])
    @assert all(length(s) == n1 for s in Sout) "All outer stabilizers must have same length n1"
    @assert all(length(s) == n2 for s in Sin)  "All inner stabilizers must have same length n2"

    # validate alphabet
    valid = Set(['I','X','Y','Z'])
    @assert all(all(c in valid for c in s) for s in Sout) "Outer stabilizers must be in {I,X,Y,Z}"
    @assert all(all(c in valid for c in s) for s in Sin)  "Inner stabilizers must be in {I,X,Y,Z}"

    N = n1 * n2
    # index mapping
    idx = ordering == :interleaved ?
        ((b,j)->(j-1)*n1 + b) :
        ordering == :block ?
        ((b,j)->(b-1)*n2 + j) :
        error("Unsupported ordering: $ordering (use :interleaved or :block)")

    out = String[]

    # --- Inner stabilizers: apply each inner row across ALL blocks simultaneously ---
    for s_in in Sin
        chars = fill('I', N)
        for j in 1:n2
            c = s_in[j]
            if c != 'I'
                for b in 1:n1
                    chars[idx(b,j)] = c
                end
            end
        end
        push!(out, String(chars))
    end

    # --- Outer stabilizers: for EACH inner position j, place s_out letters at that j across blocks ---
    for s_out in Sout
        for j in 1:n2
            chars = fill('I', N)
            for b in 1:n1
                c = s_out[b]
                if c != 'I'
                    chars[idx(b,j)] = c
                end
            end
            push!(out, String(chars))
        end
    end

    return out
end




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



Stabilizers = ["ZZIIIIIII", "ZIZIIIIII", "IIIZZIIII", "IIIZIZIII", "IIIIIIZZI", "IIIIIIZIZ", "XXXXXXIII", "XXXIIIXXX"]
Stabilizers = ["ZIZIIIIII", "IZZIIIIII", "IIIZIZIII", "IIIIZZIII", "IIIIIIZIZ", "IIIIIIIZZ", "XXXIIIXXX", "XXXXXXIII"]

#Stabilizers = ["ZZ"]

#Stabilizers = ["ZIIIZ", "IZIIZ", "IIZIZ", "IIIZZ"]
Stabilizers = ["ZIIIIIIIZ", "IZIIIIIIZ", "IIZIIIIIZ", "IIIZIIIIZ", "IIIIZIIIZ", "IIIIIZIIZ", "IIIIIIZIZ", "IIIIIIIZZ"]
Stabilizers = ["XXXIIIZZZ", "IIIXXXZZZ", "ZIZIIIIII", "IIIZIZIII", "IIIIIIZIZ", "IZZIIIIII", "IIIIZZIII", "IIIIIIIZZ"]
Stabilizers = ["ZIIIZIIIIIIIIII","IZIIZIIIIIIIIII","IIZIZIIIIIIIIII","IIIZZIIIIIIIIII","IIIIIZIIIZIIIII","IIIIIIZIIZIIIII","IIIIIIIZIZIIIII","IIIIIIIIZZIIIII","IIIIIIIIIIZIIIZ","IIIIIIIIIIIZIIZ","IIIIIIIIIIIIZIZ","IIIIIIIIIIIIIZZ","XXXXXIIIIIXXXXX","XXXXXXXXXXIIIII"]

Stabilizers = ["ZZZ"]
Stabilizers = ["XIIZII", "IZZIII", "IZIZZZ", "IZIIIZ"]
Stabilizers = ["ZIIIIIIZ", "IZIIIIIZ", "IIZIIIIZ", "IIIZIIZI", "IIIIZIZI", "IIIIIZZI"]
Stabilizers = ["ZZZZZIIZ", "IIZZZIIZ", "IIZIZZZI", "ZZIIIZZI"]

Stabilizers = ["ZZ"]
#Stabilizers = ninexz_maker(.5, 17; searchDown = false)
#Stabilizers = ["IZXXX", "ZIXIX", "IIXXI"]
#Stabilizers = ["ZZXII", "ZIIXX", "IIXXI"]
#Stabilizers = ["ZIIXX", "IZIIX", "IIZXI"]
#Stabilizers = ["XIIXX", "IXIXX", "IIXXI"]
#CHANNEL = "Depolarizing"
#Stabilizers = ["ZZZZZZ"]
#Stabilizers = ["ZIIIIIZ", "IZIIIZI", "IIZIZII", "IIIZZZZ"]
#Stabilizers1 = ["XX"]
#Stabilizers = ["ZIIIZ", "IZIIZ", "IIZZI"]
#Stabilizers1 = ["XIX","IXX"]
#Stabilizers = ["ZIIIZ", "IZIIZ", "IIZIZ", "IIIZZ"]

#Stabilizers = ["ZZZZZZ"]
#Stabilizers = ["XIX", "IZI"]
#Stabilizers = ["XIIX", "IXIX", "IIXX"]
#Stabilizers1 = ["ZZZZZZ"]
#Stabilizers1 = ["XXYZ", "XZYX", "ZIZI"]
#Stabilizers1 = ["IIIXX", "ZXIIX", "ZXXII", "IIXIX"]
#Stabilizers1 = ["XYXZ", "YXZX", "YYXX"]
#Stabilizers1 = ["ZIIIZ", "IZIIZ", "IIZIZ", "IIIZZ"]
#Stabilizers = ["ZIIIIIIIIIIIIIZ","IZIIIIIIIIIIIIZ","IIZIIIIIIIIIIIZ","IIIZIIIIIIIIIIZ","IIIIZIIIIIIIIIZ","IIIIIZIIIIIIIIZ","IIIIIIZIIIIIIIZ","IIIIIIIZIIIIIIZ","IIIIIIIIZIIIIIZ","IIIIIIIIIZIIIIZ","IIIIIIIIIIZIIIZ","IIIIIIIIIIIZIIZ","IIIIIIIIIIIIZIZ","IIIIIIIIIIIIIZZ"]
#Stabilizers = ["ZIZ", "IZZ"]
Stabilizers = ["ZZZZZZZZZZZZZZ"]
#Stabilizers = ["XX"]
Stabilizers = ["ZIIIZ", "IZIIZ", "IIZIZ", "IIIZZ"]
Stabilizers = ["ZIIZ", "IZIZ", "IIZZ"]
#Stabilizers = ["ZZ"]

#Stabilizers = concat_stabilizers_strings(Stabilizers1, Stabilizers) # I think the first input should be bigger (i think it is the outer code)


S = Symplectic.build_from_stabs(Stabilizers)
#@show S
# Ensure it's a plain Bool matrix
S = Matrix{Bool}(S)

# Build tableau/logicals
H, Lx, Lz, G = QECInduced.tableau_from_stabilizers(S)


#@show size(H)  # (r, 2n)
#@show size(Lx) # (k, 2n)
#@show size(Lz) # (k, 2n)
#@show size(G)  # (r, 2n)

@show H
@show Lx
@show Lz
@show G

# check that each of H, Lx, Lz, G commute within themselves
@show Symplectic.sanity_check(H, Lx, Lz, G)

points = 50
pz_range = range(0.18885714285714286,  0.18892857142857142, length = points)
pz_range_override = range(0.18889285714285714, 0.19022857142857144, length = points)
pz_range_override = range(0.233, .272, length = points)
pz_range = range(.231, 0.232, length = points)
pz_range = range(0.23, .25, length = points)

#pz_range = range(.233, 0.234, length = points)
pz = 0 

customP = ninexz # change this 
ChannelType = "Custom" # Dont change this 


hashing = QECInduced.sweep_hashing_grid(pz_range, ChannelType; customP = customP)
hb_grid = QECInduced.check_induced_channel(S, pz; ChannelType=ChannelType, sweep=true, ps=pz_range, customP=customP, threads = 0)
#@show p_channel
print("[")
for i in pz_range[1:end-1]
    print(ninexz(i;plot=true), ", ")
end 
println(ninexz(pz_range[end];plot=true), "]")
println(hashing)
#println(hb_grid .- hashing)
println(hb_grid) 







