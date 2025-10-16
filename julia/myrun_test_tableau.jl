# 5-qubit code, r = 4; k = 1
# Pauli vector format is (u | v) with length 2n
#  - X on qubit i -> u_i = 1, v_i = 0
#  - Z on qubit i -> u_i = 0, v_i = 1

include("/home/revelio/QEC/quantumchanneltransformation/julia/src/Symplectic.jl")

using QECInduced, .Symplectic

n = 5 
S = falses(4, 2n)  # 4 stabilizers, 2n columns

# Stabilizers: X Z Z X I, I X Z Z X, X I X Z Z, Z X I X Z
# Row 1: (u=10010 | v=01100)
S[1, 1] = true  # u1 
S[1, 4] = true  # u4
S[1, 2+n] = true  # v2 
S[1, 3+n] = true  # v3

# Row 2: (u=01001 | v=00110)
S[1, 2] = true  # u2
S[1, 5] = true  # u5
S[1, 3+n] = true  # v3 
S[1, 4+n] = true  # v4

# Row 3: (u=10100 | v=00011)
S[1, 1] = true  # u1
S[1, 3] = true  # u3
S[1, 4+n] = true  # v4 
S[1, 5+n] = true  # v5

# Row 4: (u=01010 | v=10001)
S[1, 2] = true  # u2
S[1, 4] = true  # u4
S[1, 1+n] = true  # v1
S[1, 5+n] = true  # v5

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
@show Symplectic.check_commutes(H)
@show Symplectic.check_commutes(Lx)
@show Symplectic.check_commutes(Lz)
@show Symplectic.check_commutes(G)


@show Symplectic.symp_inner(view(Lx,1,:), view(Lz,1,:))

for i in 1:size(H,1), j in 1:size(H,1)                                                                              
        @show i, j, symp_inner(view(H,i,:), view(G,j,:))                                                    
end


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

