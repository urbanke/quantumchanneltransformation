# 5-qubit code, r = 4; k = 1
# Pauli vector format is (u | v) with length 2n
#  - X on qubit i -> u_i = 1, v_i = 0
#  - Z on qubit i -> u_i = 0, v_i = 1

include("src/Symplectic.jl")

using QECInduced, .Symplectic


Stabilizers = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
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
@show Symplectic.sanity_check(H,Lx,Lz,G)

# k should be 1 for the 5-qubit code
@assert size(Lx, 1) == 1 && size(Lz, 1) == 1 "Expected k=1 logical qubit"

# Depolarizing channel with probability p
p = 0.10

# Call the public wrapper: it expects keyword `p::Float64`
pbar, hashing = QECInduced.induced_channel_and_hashing_bound(H, Lx, Lz, G; p=p)

@show size(pbar)
@show hashing

grid = QECInduced.sweep_depolarizing_grid(H, Lx, Lz, G; p_min=0.0, p_max=0.3, step=0.05, threads=4)
println("grid:\n", grid)