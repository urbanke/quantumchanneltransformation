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
#Stabilizers = ["ZIIIIIIIZ", "IZIIIIIIZ", "IIZIIIIIZ", "IIIZIIIIZ", "IIIIZIIIZ", "IIIIIZIIZ", "IIIIIIZIZ", "IIIIIIIZZ"]
#Stabilizers = ["ZIIIZIIIIIIIIII","IZIIZIIIIIIIIII","IIZIZIIIIIIIIII","IIIZZIIIIIIIIII","IIIIIZIIIZIIIII","IIIIIIZIIZIIIII","IIIIIIIZIZIIIII","IIIIIIIIZZIIIII","IIIIIIIIIIZIIIZ","IIIIIIIIIIIZIIZ","IIIIIIIIIIIIZIZ","IIIIIIIIIIIIIZZ","XXXXXIIIIIXXXXX","XXXXXXXXXXIIIII"]

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
Stabilizers = ["XIX", "IZI"]
#Stabilizers = ["XIIX", "IXIX", "IIXX"]
#Stabilizers1 = ["ZZZZZZ"]
#Stabilizers1 = ["XXYZ", "XZYX", "ZIZI"]
#Stabilizers1 = ["IIIXX", "ZXIIX", "ZXXII", "IIXIX"]
#Stabilizers1 = ["XYXZ", "YXZX", "YYXX"]
#Stabilizers1 = ["ZIIIZ", "IZIIZ", "IIZIZ", "IIIZZ"]
#Stabilizers = ["ZIIIIIIIIIIIIIZ","IZIIIIIIIIIIIIZ","IIZIIIIIIIIIIIZ","IIIZIIIIIIIIIIZ","IIIIZIIIIIIIIIZ","IIIIIZIIIIIIIIZ","IIIIIIZIIIIIIIZ","IIIIIIIZIIIIIIZ","IIIIIIIIZIIIIIZ","IIIIIIIIIZIIIIZ","IIIIIIIIIIZIIIZ","IIIIIIIIIIIZIIZ","IIIIIIIIIIIIZIZ","IIIIIIIIIIIIIZZ"]
#Stabilizers1 = ["ZIZ", "IZZ"]
#Stabilizers1 = ["ZIIIZ", "IZIIZ", "IIZIZ", "IIIZZ"]

#Stabilizers = concat_stabilizers_strings(Stabilizers1, Stabilizers)

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

# Channel parameter
px = 0.233
pz = px/9
p = 0.1905
px = 0.1094384294502969
p = 0.19022857142857144
#px = 0.1094384294502969
pz = px 
p = 0.1904775
CHANNEL = "Depolarizing"



# from the [3,1] code at depolarizing p = .1905 - both the good and bad channel  (0.6673869999999998)
pI = 0.43649999999999983
pX =  0.43649999999999983
pZ = 0.06349999999999997
pY = 0.06349999999999999

pI = 0.8095
pX = 0.187430741833449
pZ = 0.001534629083275521
pY = 0.001534629083275521


# pI = 0.7277731462389959
# pX = 0.27216170306033616
# pZ = 3.2575350335105956e-5
# pY = 3.2575350335105956e-5





#CHANNEL = "Special"
#p = 0.1
#CHANNEL = "Depolarizing"
# Single-qubit Pauli channel tuple (pI, pX, pZ, pY)
if CHANNEL == "Depolarizing"
    # Depolarizing: [1-p, p/3, p/3, p/3]
    p_channel = [1 - p, p/3, p/3, p/3]
elseif CHANNEL == "Independent"
    # Independent X/Z flips: [(1-p)^2, p(1-p), p(1-p), p^2]
    p_channel = [(1 - px) * (1 - pz), px * (1 - pz), pz * (1 - px), px * pz] 
else 
    p_channel = [pI, pX, pZ, pY]
end 



#@show p_channel

println("\nHashing bound of the ORIGINAL physical channel (per-qubit):")
hashing_orig = 1 - QECInduced.H(p_channel)
@show hashing_orig

println("\nComputing induced-channel distribution and per-syndrome hashing bound (new definition):")
hashing_induced = QECInduced.induced_channel_and_hashing_bound(H, Lx, Lz, G, p_channel)
#@show size(pbar)
#@show pbar
println("Induced (per-syndrome) hashing bound returned by kernel: (k - Σ_s p(s) H(p(a',b'|s)))/n")
@show hashing_induced




function inducedChannelFromRepCode(m,PI,PX,PZ,PY)
    pizp = PI + PZ
    pizm = PI - PZ 
    pxyp = PX + PY 
    pxym = PX - PY 
    pib = 0
    pzb = 0 
    for i in 0:Int(floor(m/2))
        mci = binomial(m,i) 
        pib += mci*pizp^(m-i)*pxyp^(i) 
        pib += mci*pizm^(m-i)*pxym^(i) 

        pzb += mci*pizp^(m-i)*pxyp^(i) 
        pzb -= mci*pizm^(m-i)*pxym^(i) 
    end 
    pib = pib/2 
    pzb = pzb/2 
    pxb = .5*(1 + (pizm + pxym)^m) - pib
    pyb = .5*(1 - (pizm + pxym)^m) - pzb

    return pib, pxb, pzb, pyb 
end 



# uncomment to simulate concatenated code # 
# I switch PZ and PX because in theory you should do a Z code followed by an X code 
# but if you do this we can do a Z code followed by a Z code and it is the same 
#= 

m = 3

PI, PZ, PX, PY = inducedChannelFromRepCode(m, p_channel[1], p_channel[2], p_channel[3], p_channel[4]) 
p_channel = [PI, PX, PZ, PY] 

println("\nComputing induced-channel distribution and per-syndrome hashing bound (with $m concatenation):")
hashing_induced = QECInduced.induced_channel_and_hashing_bound(H, Lx, Lz, G, p_channel)/m
#@show size(pbar)
#@show pbar
println("Induced (per-syndrome) hashing bound returned by kernel: (k - Σ_s p(s) H(p(a',b'|s)))/n")
@show hashing_induced

=# 
#=
# Grids (p vs bounds) — uses the same public sweep helpers.
if CHANNEL == "Depolarizing"
    grid = QECInduced.sweep_depolarizing_grid(H, Lx, Lz, G; p_min = 0.0, p_max = 0.5, step = 0.01, threads = 4)
else
    grid = QECInduced.sweep_independent_grid(H, Lx, Lz, G; p_min = 0.0, p_max = 0.5, step = 0.01, threads = 4)
end

println("\nGrid columns are assumed as [p, hashing_bound_original, hashing_bound_induced]:")
println("grid:\n", grid)

# -----------------------------
# Plot (p, original per-qubit bound, induced per-syndrome bound)
# -----------------------------
ps      = grid[:, 1]
origHB  = grid[:, 3]  # 1 - H(p_channel) per qubit
indHB   = grid[:, 2]  # (k - Σ_s p(s) H(· | s))/n from the updated kernel

# Bring in Plots (install if missing)
try
    using Plots
catch
    import Pkg; Pkg.add("Plots"); using Plots
end

plt = plot(
    ps, origHB;
    label = "Original channel (per-qubit 1 - H(p))",
    xlabel = CHANNEL * " probability p",
    ylabel = "Hashing bound",
    title = "Hashing bounds vs p",
    marker = :circle,
    linewidth = 2,
)

plot!(plt, ps, indHB; label = "Induced (per-syndrome conditional entropy)", marker = :square, linewidth = 2)

# Save figure (and print the path)
outfile = "hashing_bounds_vs_p.png"
savefig(plt, outfile)
println("Saved plot to $(outfile)")
=#
