module Induced

using ..Bitpack

export induced_channel_and_hashing_bound

# Resolve platform-specific default path for the Rust library.
const DEFAULT_RUST_LIB = get(ENV, "QEC_RUST_LIB",
    Sys.islinux()  ? joinpath(@__DIR__, "..", "..", "rust_kernel", "target", "release", "librust_kernel.so") :
    Sys.isapple()  ? joinpath(@__DIR__, "..", "..", "rust_kernel", "target", "release", "librust_kernel.dylib") :
                     joinpath(@__DIR__, "..", "..", "rust_kernel", "target", "release", "rust_kernel.dll"))

"""
    induced_channel_and_hashing_bound(H, Lx, Lz, G, p_tuple) -> (pbar, hashing_bound)

Compute the ML-centered induced distribution and the **per-syndrome hashing bound** under an i.i.d.
Pauli channel with single-qubit probabilities `p_tuple = (pI,pX,pZ,pY)`.

Definitions:
  • For each syndrome s, compute Pₛ(a,b) = ∑ₜ P_chan(E(t,a,b,s)).
  • Let (a*(s), b*(s)) be the ML pair for s, and define a′=a⊻a*(s), b′=b⊻b*(s).
  • p̄(a′,b′ | s) = Pₛ(a′⊻a*(s), b′⊻b*(s)) / p̄(s), with p̄(s)=∑_{a′,b′} Pₛ(…).
  • H̄ = ∑ₛ p̄(s)·H( p̄(a′,b′ | s) ).

Return values:
  • `pbar::Matrix{Float64}`: ML-centered *marginal* over (a′,b′), normalized over all s.
  • `hashing_bound::Float64`: (k - H̄)/n  (note: **conditional-entropy** form).

The Rust kernel computes H̄ directly; the ABI and allocation shape are unchanged.
"""
function induced_channel_and_hashing_bound(H::AbstractMatrix{Bool},
                                           Lx::AbstractMatrix{Bool},
                                           Lz::AbstractMatrix{Bool},
                                           G::AbstractMatrix{Bool},
                                           p_tuple::NTuple{4,Float64})
    pI, pX, pZ, pY = p_tuple
    r, n2 = size(H)
    @assert iseven(n2); n = n2 >>> 1
    k = size(Lx,1)
    S = 1 << r
    A = 1 << k

    # bit-pack bases
    HU, HV, _ = Bitpack.to_bitpacked_uv(H)
    GU, GV, _ = Bitpack.to_bitpacked_uv(G)
    LxU, LxV, _ = Bitpack.to_bitpacked_uv(Lx)
    LzU, LzV, _ = Bitpack.to_bitpacked_uv(Lz)

    # patterns
    patt_r = falses(S, r)
    for i in 0:S-1, b in 1:r
        patt_r[i+1, r-b+1] = isodd((i >> (b-1)) & 0x1)
    end
    patt_k = falses(A, k)
    for i in 0:A-1, b in 1:k
        patt_k[i+1, k-b+1] = isodd((i >> (b-1)) & 0x1)
    end

    # precompute bitpacked spans (vectorized over t,a,b,s)
    TtabU = Bitpack.span_bitpacked(patt_r, HU)
    TtabV = Bitpack.span_bitpacked(patt_r, HV)
    ALU   = Bitpack.span_bitpacked(patt_k, LxU)
    ALV   = Bitpack.span_bitpacked(patt_k, LxV)
    BLU   = Bitpack.span_bitpacked(patt_k, LzU)
    BLV   = Bitpack.span_bitpacked(patt_k, LzV)
    SGU   = Bitpack.span_bitpacked(patt_r, GU)
    SGV   = Bitpack.span_bitpacked(patt_r, GV)

    # Prepare output pbar
    pbar = zeros(Float64, A, A)

    @assert isfile(DEFAULT_RUST_LIB) "Rust library not found at: $(DEFAULT_RUST_LIB). Set QEC_RUST_LIB if needed."

    # NOTE: return now uses H̄ (per-syndrome conditional entropy), not H(p̄(a′,b′)).
    hashing_bound = ccall((:compute_pbar_and_hashing_bound, DEFAULT_RUST_LIB), Float64,
        ( Ptr{UInt64}, Ptr{UInt64}, Csize_t, Csize_t,
          Ptr{UInt64}, Ptr{UInt64}, Csize_t,
          Ptr{UInt64}, Ptr{UInt64}, Csize_t,
          Ptr{UInt64}, Ptr{UInt64}, Csize_t,
          Cuint, Cuint, Cuint, Cdouble, Cdouble, Cdouble, Cdouble,
          Ptr{Float64}
        ),
        pointer(TtabU), pointer(TtabV), size(TtabU,1), size(TtabU,2),
        pointer(ALU),   pointer(ALV),   size(ALU,1),
        pointer(BLU),   pointer(BLV),   size(BLU,1),
        pointer(SGU),   pointer(SGV),   size(SGU,1),
        UInt32(n), UInt32(r), UInt32(k), pI, pX, pZ, pY,
        pointer(pbar)
    )

    return pbar, hashing_bound
end

end # module


