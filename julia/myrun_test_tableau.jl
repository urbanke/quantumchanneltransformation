# 5-qubit code, r = 4; k = 1
# Pauli vector format is (u | v) with length 2n
#  - X on qubit i -> u_i = 1, v_i = 0
#  - Z on qubit i -> u_i = 0, v_i = 1

include("src/Symplectic.jl")


using QECInduced, .Symplectic


#Stabilizers = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"] # 5-qubit repetition code
Stabilizers = ["ZZIIIIIII", "IZZIIIIII", "IIIZZIIII", "IIIIZZIII","IIIIIIZZI","IIIIIIIZZ","IIIXXXXXX","XXXXXXIII"] # 9-qubit Shor code
#Stabilizers = ["ZZIZZIZZI", "IZZIZZIZZ", "IIIXXXXXX", "XXXXXXIII"]  # 9-qubit, rate 5/9, Bacon-Shor code
#Stabilizers = ["XXI", "IXX"]  # 3-qubit repetition code 
#Stabilizers = ["ZZI", "IZZ"]  # 3-qubit repetition code 
#Stabilizers = ["XXIII", "IXXII", "IIXXI", "IIIXX"] # 5-qubit repetition code
CHANNEL = "Independent" # Choose one 
#CHANNEL = "Depolarizing"
#S = Symplectic.build_from_stabs(Stabilizers)
S = Bool[0 0 0 1 0 0 0 0 0 0; 0 0 1 0 0 0 0 0 0 0; 0 1 0 0 0 0 0 0 0 0; 1 0 0 0 0 0 0 0 0 0]
@show S


# Ensure it's a plain Bool matrix
S = Matrix{Bool}(S)


# Build tableau/logicals
H, Lx, Lz, G = QECInduced.tableau_from_stabilizers(S)

@show size(H)  # (r, 2n)
@show size(Lx) # (k, 2n)
@show size(Lz) # (k, 2n)
@show size(G)  # (r, 2n)

@show H
@show Lx
@show Lz
@show G

# check that each of H, Lx, Lz, G commute within themselves
@show Symplectic.sanity_check(H,Lx,Lz,G)

# k should be 1 for the 5-qubit code
# @assert size(Lx, 1) == 1 && size(Lz, 1) == 1 "Expected k=1 logical qubit"

# channel paramter
#p = 0.11002786457538605
p = 0.11002786443835955
if CHANNEL == "Depolarizing"
# original depolarizing channel 
  p_channel = [1-p, p/3, p/3, p/3]
else
# original independent channel 
p_channel = [(1-p)*(1-p), p*(1-p), p*(1-p), p*p]
end 
 


@show p
@show p_channel

println("Hashing bound of original channel") 
hashing_orig = 1 - QECInduced.H(p_channel) 
@show hashing_orig
# Call the public wrapper: it expects keyword `p::Float64`
pbar, hashing_induced = QECInduced.induced_channel_and_hashing_bound(H, Lx, Lz, G, p_channel)

@show size(pbar)
@show pbar
@show hashing_induced

if CHANNEL == "Depolarizing"
grid = QECInduced.sweep_depolarizing_grid(H, Lx, Lz, G; p_min=0.0, p_max=0.5, step=0.01, threads=4)
else
grid = QECInduced.sweep_independent_grid(H, Lx, Lz, G; p_min=0.0, p_max=0.5, step=0.01, threads=4)
end
println("grid:\n", grid)



# -----------------------------
# Plot the (p, hashing_orig, hashing_induced) triplets you provided
# -----------------------------
# Data format: each row is [p, hashing_bound_original, hashing_bound_induced]

ps  = grid[:, 1]
hib = grid[:, 2]  # original hashing bound
hob =  grid[:, 3]  # induced hashing bound

# Bring in Plots (install if missing)
try
    using Plots
catch
    import Pkg; Pkg.add("Plots"); using Plots
end

plt = plot(
    ps, hob;
    label = "Original channel",
    xlabel = CHANNEL*" probability p",
    ylabel = "Hashing bound",
    title = "Hashing bounds vs p",
    marker = :circle,
    linewidth = 2,
)

plot!(plt, ps, hib; label = "Induced channel", marker = :square, linewidth = 2)

# Save figure (and print the path)
outfile = "hashing_bounds_vs_p.png"
savefig(plt, outfile)
println("Saved plot to $(outfile)")
