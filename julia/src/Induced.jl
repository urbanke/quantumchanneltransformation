module Induced

using ..Bitpack

export induced_channel_and_hashing_bound

# Resolve platform-specific default path for the Rust library.
const DEFAULT_RUST_LIB = get(ENV, "QEC_RUST_LIB",
    Sys.islinux()  ? joinpath(@__DIR__, "..", "..", "rust-kernel", "target", "release", "librust_kernel.so") :
    Sys.isapple()  ? joinpath(@__DIR__, "..", "..", "rust-kernel", "target", "release", "librust_kernel.dylib") :
                     joinpath(@__DIR__, "..", "..", "rust-kernel", "target", "release", "rust_kernel.dll"))

"""
    induced_channel_and_hashing_bound(H, Lx, Lz, G, p_tuple) -> (pbar, hashing_bound)

Compute induced p̄(a′,b′) and hashing bound under an i.i.d. Pauli channel with
single-qubit probabilities p_tuple = (pI,pX,pZ,pY). Fast path:
- bit-pack all precomputed spans:
  Ttab = span(H) over t ∈ {0,1}^r,
  AL   = span(Lx) over a ∈ {0,1}^k,
  BL   = span(Lz) over b ∈ {0,1}^k,
  SG   = span(G)  over s ∈ {0,1}^r,
- call Rust `compute_pbar_and_hashing_bound` to do heavy inner loops + argmax + accumulation.
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
    W = size(HU,2)

    # patterns
    patt_r = falses(S, r)
    for i in 0:S-1
        for b in 1:r
            patt_r[i+1, r-b+1] = isodd((i >> (b-1)) & 0x1)
        end
    end
    patt_k = falses(A, k)
    for i in 0:A-1
        for b in 1:k
            patt_k[i+1, k-b+1] = isodd((i >> (b-1)) & 0x1)
        end
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

    # ccall into Rust
    lib = DEFAULT_RUST_LIB
    @assert isfile(lib) "Rust library not found at: $lib. Set QEC_RUST_LIB if needed."

    hashing_bound = ccall((:compute_pbar_and_hashing_bound, lib), Float64,
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

