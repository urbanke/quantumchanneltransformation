module QECInduced

export tableau_from_stabilizers,
       induced_channel_and_hashing_bound,
       sweep_depolarizing_grid,
       demo

include("Symplectic.jl")
include("SGS.jl")
include("Bitpack.jl")
include("Induced.jl")
include("ParallelSweep.jl")

using .Symplectic
using .SGS
using .Bitpack
using .Induced
using .ParallelSweep

"""
    tableau_from_stabilizers(S::AbstractMatrix{Bool})

Compute a full stabilizer tableau `(H, Lx, Lz, G)` from an independent set
of stabilizers `S ∈ F₂^{r×2n}` using Symplectic Gram–Schmidt (SGS).
- `H` generates the same stabilizer group as `S` (rowspan equal).
- `G` is the pure-error block, `HJGᵀ = I`.
- `Lx, Lz` are k logical pairs with `Lx JLzᵀ = I`.
"""
tableau_from_stabilizers(S) = SGS.tableau_from_stabilizers(S)

"""
    induced_channel_and_hashing_bound(H, Lx, Lz, G; p=0.1)

Compute the induced logical channel p̄(a′,b′) and hashing bound `(k - H(p̄))/n`
under the depolarizing channel with parameter `p`.
Uses bit-packing and calls into the Rust kernel for speed.
Returns `(pbar::Matrix{Float64}, hashing_bound::Float64)`.
"""
function induced_channel_and_hashing_bound(H, Lx, Lz, G; p::Float64=0.1)
    (pI, pX, pZ, pY) = (1-p, p/3, p/3, p/3)
    Induced.induced_channel_and_hashing_bound(H, Lx, Lz, G, (pI, pX, pZ, pY))
end

"""
    sweep_depolarizing_grid(H, Lx, Lz, G; p_min=0.0, p_max=1.0, step=0.01, threads=Threads.nthreads())

Parallel sweep of depolarizing parameter p in [p_min, p_max] with the given step.
Returns a 2-column matrix `[p  hashing_bound]`.
"""
sweep_depolarizing_grid(H, Lx, Lz, G; p_min=0.0, p_max=1.0, step=0.01, threads=Threads.nthreads()) =
    ParallelSweep.sweep_depolarizing_grid(H, Lx, Lz, G; p_min, p_max, step, threads)

"""
    demo()

Tiny demo: n=1, k=1 (no stabilizers). Shows single p and small sweep.
"""
function demo()
    n = 1
    H = falses(0, 2n)
    G = falses(0, 2n)
    Lx = falses(1, 2n); Lx[1, 1] = true  # X on qubit 1
    Lz = falses(1, 2n); Lz[1, n+1] = true  # Z on qubit 1

    pbar, hb = induced_channel_and_hashing_bound(H, Lx, Lz, G; p=0.1)
    @info "pbar shape = $(size(pbar)) sum=$(sum(pbar)) hashing_bound=$hb"

    grid = sweep_depolarizing_grid(H, Lx, Lz, G; p_min=0.0, p_max=0.2, step=0.05, threads=2)
    @info "grid =\n$(grid)"
    nothing
end

end # module

