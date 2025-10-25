module QECInduced

export tableau_from_stabilizers,
       induced_channel_and_hashing_bound,
       sweep_depolarizing_grid,
       demo, 
       check_induced_channel

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



function H(p)
    s = 0.0
    @inbounds for v in p
        if v > 0
            s -= v * log2(v)
        end
    end
    return s
end


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
function induced_channel_and_hashing_bound(H, Lx, Lz, G, p_channel::AbstractVector{<:Real})
    @assert length(p_channel) == 4 "expected p_channel = [pI, pX, pZ, pY]"
    pI, pX, pZ, pY = p_channel
    return Induced.induced_channel_and_hashing_bound(
        H, Lx, Lz, G,
        (float(pI), float(pX), float(pZ), float(pY))
    )
end

"""
    sweep_depolarizing_grid(H, Lx, Lz, G; p_min=0.0, p_max=1.0, step=0.01, threads=Threads.nthreads())

Parallel sweep of depolarizing parameter p in [p_min, p_max] with the given step.
Returns a 3-column matrix `[p  induced_hashing_bound 1-h(p)]`.
"""
sweep_depolarizing_grid(H, Lx, Lz, G; p_min=0.0, p_max=1.0, step=0.01, threads=Threads.nthreads()) =
    ParallelSweep.sweep_depolarizing_grid(H, Lx, Lz, G; p_min, p_max, step, threads)

"""
    sweep_indepdent_grid(H, Lx, Lz, G; p_min=0.0, p_max=1.0, step=0.01, threads=Threads.nthreads())

Parallel sweep of independent channel with parameter p in [p_min, p_max] with the given step.
Returns a 2-column matrix `[p  induced_hashing_bound 1-h(p)]`.
"""
sweep_independent_grid(H, Lx, Lz, G; p_min=0.0, p_max=1.0, step=0.01, threads=Threads.nthreads()) =
    ParallelSweep.sweep_independent_grid(H, Lx, Lz, G; p_min, p_max, step, threads)


"""

    InduceChannel(Stabilizers, ChannelType)

Takes in a list of stabilizers, as well as the ChannelType (currently only Depolarizing or Independent). If there is a moment where the induced channel is both better than 0 AND H(p_channel), it returns true
Stabilizer must be in boolean form not XYZ form. 
"""
function check_induced_channel(S; ChannelType = "Independent")
# Build tableau/logicals
    H, Lx, Lz, G = QECInduced.tableau_from_stabilizers(S)

# check that each of H, Lx, Lz, G commute within themselves
    @assert(Symplectic.sanity_check(H,Lx,Lz,G) == true, "Error Constructing Tableau")


    if ChannelType == "Depolarizing"
        grid = QECInduced.sweep_depolarizing_grid(H, Lx, Lz, G; p_min=0.0, p_max=0.5, step=0.01, threads=4)
    else
        grid = QECInduced.sweep_independent_grid(H, Lx, Lz, G; p_min=0.0, p_max=0.5, step=0.01, threads=4)
    end

    ps  = grid[:, 1]
    hib = grid[:, 2]  # original hashing bound
    hob =  grid[:, 3]  # induced hashing bound

    good_hib = (hib .> hob) .& (hib .> 0) # this is checking that it is both beating the original channel and also non-zero

    if sum(good_hib) .> 0 # if there is at least one of these 
        return true
    end
    return false
end  


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

