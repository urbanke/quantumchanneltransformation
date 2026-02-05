# 5-qubit code, r = 4; k = 1
# Pauli vector format is (u | v) with length 2n
#  - X on qubit i -> u_i = 1, v_i = 0
#  - Z on qubit i -> u_i = 0, v_i = 1

include("../src/Symplectic.jl")
include("../env_utils/Channels.jl")

using QECInduced, .Symplectic, .Channels


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

function printNice(S)
    print("[")
    for i in 1:size(S)[1] - 1
        print("\"")
        print(build_from_bits(S[i,:]))
        print("\", ")
    end 
    i = size(S)[1] 
    println("\"",build_from_bits(S[i,:]),"\"]")
end

function build_from_bits(S)
    n = div(length(S),2) 
    Xs = S[1:n]
    Zs = S[n+1:(2*n)]
        tempString = ""
        for j in 1:n 
            if Xs[j] & Zs[j]
                tempString = tempString*"Y" 
            elseif Xs[j] 
                tempString = tempString*"X" 
            elseif Zs[j] 
                tempString = tempString*"Z" 
            else 
                tempString = tempString*"I" 
            end
        end
    return tempString
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

Stabilizers1 = ["ZIIIZ", "IZIIZ", "IIZIZ", "IIIZZ"]
Stabilizers = ["ZIIZ", "IZIZ", "IIZZ"]
Stabilizers = ["ZIZ", "IZZ"]
Stabilizers = ["ZIZ"]
Stabilizers = ["ZIIIIIII","ZZIIIIII","ZIZIIIII","IIIIIIIX","IIIIIIXX","IIIIIXIX"]
Stabilizers = ["ZIIIZ", "IZIIZ", "IIZIZ", "IIIZZ"]
Stabilizers = ["ZIIIIIIIIIIIIIII", "ZZIIIIIIIIIIIIII", "ZIZIIIIIIIIIIIII", "ZZZZIIIIIIIIIIII","ZIIIZIIIIIIIIIII","ZZIIZZIIIIIIIIII","ZIIIIIIIZIIIIIII","IIIIIIIIXIXIXIXI",
               "IIIIIIIIIIIIIIIX", "IIIIIIIIIIIIIIXX", "IIIIIIIIIIIIIXIX", "IIIIIIIIIIIIXXXX","IIIIIIIIIIIXIIIX","IIIIIIIIIIXXIIXX","XIIIIIIIXIIIIIII"]

Stabilizers = ["ZIIIIIIIIIIIIIII", "ZZIIIIIIIIIIIIII", "ZIZIIIIIIIIIIIII", "ZIIIZIIIIIIIIIII","ZIIIIIIIZIIIIIII","ZIZIIIIIZIZIIIII","ZIIIZIIIZIIIZIII",
               "IIIIIIIIIIIIIIIX", "IIIIIIIIIIIIIIXX", "IIIIIIIIIIIIIXIX", "IIIIIIIIIIIXIIIX","IIIIIIIXIIIIIIIX","IIIIIIIXIXIIIXIX","IIIXIIIXIIIXIIIX","IIIIIIIIXIXIXIXI"]

Stabilizers = ["ZZZZZZZZ", "IZIZIZIZ", "IIZZIIZZ", 
               "XXXXXXXX", "XIXIXIXI", "XXIIXXII",]


Stabilizers = ["ZXIIIIIIIIIIIIII", "ZIXIIIIIIIIIIIII", "YZZXIIIIIIIIIIII", "YZIIIIIIZXIIIIII", "YZIIZXIIIIIIIIII", "YIZIZIXIIIIIIIII", "XYYYYZZXIIIIIIII", "YZIIIIIIZXIIIIII", "YIZIIIIIZIXIIIII", "XYYZIIIIYZZXIIII", "YIIIZIIIZIIIXIII", "XYIIYZIIYZIIZXII","XIYIYIZIYIZIZIXI","ZXXXXYYZXYYYYZZX"]


Stabilizers = ["XZZYZYYXZYYXYXXZ", "ZYYXYXXZYXXZXZZY", "IIIIIIIIXZZYZYYX", "IIIIIIIIZYYXYXXZ", "IIIIXZZYIIIIZYYX", "IIIIZYYXIIIIYXXZ", "IIXZIIZYIIZYIIYX", "IIZYIIYXIIYXIIXZ", "IXIZIZIYIZIYIYIX", "IZIYIYIXIYIXIXIZ", "IIIIIIIIIIIIXZZY", "IIIIIIXZIIIIIIZY","IIIIIIZYIIIIIIYX", "IIIIIIIIIIXZIIZY", "IIIIIIIIIIZYIIYX"]

Stabilizers = ["XIIIIIIIII", "IXIIIIIIIX", "IIXIIIIIIX", "IIIXIIIIIX", "IIIIXIIIIX", "IIIIIXIIIX", "IIIIIIXIIX", "IIIIIIIXIX", "IIIIIIIIXX"]
Stabilizers = ["XIIIIIIIII", "IXIIIIIIII", "IIXIIIIIII", "IIIXIIIIII", "IIIIXIIIII", "IIIIIXIIIX", "IIIIIIXIIX", "IIIIIIIXIX", "IIIIIIIIXX"]
Stabilizers = ["ZIIIZ", "IZIIZ", "IIZIZ", "IIIZZ"]
Stabilizers1 = ["XX"]



#println(Stabilizers)

# optional - concatenate stabilizers into n1 x n2 length one 
#Stabilizers = concat_stabilizers_strings(Stabilizers, Stabilizers1) # I think the first input should be bigger (i think it is the outer code)channelParamFunc = Channels.Independent_Skewed_X_Nine # change this 
Stabilizers = ["XXXXZZZZYZ", "YIIIIIIIXI", "IZIIIIIIXI", "IIZIIIIIXI", "IIIYIIIIXI", "IIIIXIIIIX", "IIIIIXIIIX", "IIIIIIXIIX", "IIIIIIIXIX"]
Stabilizers = ["ZZZZXXXXYX", "YIIIIIIIZI", "IXIIIIIIZI", "IIXIIIIIZI", "IIIYIIIIZI", "IIIIZIIIIZ", "IIIIIZIIIZ", "IIIIIIZIIZ", "IIIIIIIZIZ"]


channelParamFunc = Channels.Depolarizing # change this 
points = 15
pz_range = range(0.19035625, .1906, length = 15)
hashing = QECInduced.sweep_hashing_grid(pz_range, channelParamFunc)



S = Symplectic.build_from_stabs(Stabilizers)
#@show S
# Ensure it's a plain Bool matrix
S = Matrix{Bool}(S)

# Build tableau/logicals
H, Lx, Lz, G = QECInduced.tableau_from_stabilizers(S)

#H= Symplectic.build_from_stabs( ["ZZZZZZZZ", "IZIZIZIZ", "IIIIZZZZ", "XXXXXXXX", "XIXIXIXI", "XXXXIIII"])
#Lx=    Symplectic.build_from_stabs( ["IIZZIIZZ", "IIIIIZIZ"])
#Lz=    Symplectic.build_from_stabs(["XIXIIIII", "XXIIXXII"])
#G=           Symplectic.build_from_stabs(["XIIIIIII", "XXIIIIII", "XIIIXIII", "IIIIIIIZ", "IIIIIIZZ", "IIIZIIIZ"])



#@show size(H)  # (r, 2n)
#@show size(Lx) # (k, 2n)
#@show size(Lz) # (k, 2n)
#@show size(G)  # (r, 2n)


print("Stabilizers= ")
printNice(H)
print("logicalX=    ")
printNice(Lx)
print("logicalZ=    ")
printNice(Lz)
print("G=           ")
printNice(G)

# check that each of H, Lx, Lz, G commute within themselves
@show Symplectic.sanity_check(H, Lx, Lz, G)



pz = 0 

hb_grid = QECInduced.check_induced_channel(S, pz, channelParamFunc; sweep=true, ps=pz_range, threads = 0)
#@show p_channel

print("[")
for i in pz_range[1:end-1]
    print(channelParamFunc(i;plot=true), ", ")
end 
println(channelParamFunc(pz_range[end];plot=true), "]")

println(hashing)

println(hb_grid) 







