# 5-qubit code, r = 4; k = 1
# Pauli vector format is (u | v) with length 2n
#  - X on qubit i -> u_i = 1, v_i = 0
#  - Z on qubit i -> u_i = 0, v_i = 1

include("src/Symplectic.jl")

using QECInduced, .Symplectic

# Choose a code (default: 5-qubit perfect code)
# Stabilizers = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
# Other options you sometimes toggle:
# Stabilizers = ["ZZIIIIIII", "IZZIIIIII", "IIIZZIIII", "IIIIZZIII","IIIIIIZZI","IIIIIIIZZ","IIIXXXXXX","XXXXXXIII"] # 9-qubit Shor
# Stabilizers = ["ZZIZZIZZI", "IZZIZZIZZ", "IIIXXXXXX", "XXXXXXIII"]  # 9-qubit, rate 5/9, Bacon-Shor
# Stabilizers = ["XXI", "IXX"]  # 3-qubit repetition
# Stabilizers = ["ZZI", "IZZ"]  # 3-qubit repetition
# Stabilizers = ["XXIII", "IXXII", "IIXXI", "IIIXX"] # alt 5-qubit repetition flavor
# Stabilizers = ["IYIIX", "IIIIX"]
# Stabilizers = ["ZZIII", "ZIZII","ZIIZI", "ZIIIZ"]
Stabilizers = ["ZZIIIII", "ZIZIIII","ZIIZIII", "ZIIIZII", "ZIIIIZI", "ZIIIIIZ"]

CHANNEL = "Independent"
#CHANNEL = "Depolarizing"

S = Symplectic.build_from_stabs(Stabilizers)
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
@show Symplectic.sanity_check(H, Lx, Lz, G)

# Channel parameter
px = 0.26
pz = px/9.0
@show px,pz

# Single-qubit Pauli channel tuple (pI, pX, pZ, pY)
if CHANNEL == "Depolarizing"
    # Depolarizing: [1-p, p/3, p/3, p/3]
    p_channel = [1 - p, p/3, p/3, p/3]
else
    # Independent X/Z flips: [(1-p)^2, p(1-p), p(1-p), p^2]
    p_channel = [(1 - px) * (1 - pz), px * (1 - pz), pz * (1 - px), px * pz]
end

@show p_channel

println("\nHashing bound of the ORIGINAL physical channel (per-qubit):")
hashing_orig = 1 - QECInduced.H(p_channel)
@show hashing_orig

println("\nComputing induced-channel distribution and per-syndrome hashing bound (new definition):")
pbar, hashing_induced = QECInduced.induced_channel_and_hashing_bound(H, Lx, Lz, G, p_channel)
@show size(pbar)
@show pbar
println("Induced (per-syndrome) hashing bound returned by kernel: (k - Σ_s p(s) H(p(a',b'|s)))/n")
@show hashing_induced

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
origHB  = grid[:, 2]  # 1 - H(p_channel) per qubit
indHB   = grid[:, 3]  # (k - Σ_s p(s) H(· | s))/n from the updated kernel

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

