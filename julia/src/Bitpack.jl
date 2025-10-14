module Bitpack

export to_bitpacked_uv, span_bitpacked, counts_by_type_bitpacked

"""
    to_bitpacked_uv(rows_uv::AbstractMatrix{Bool}) -> (U::Matrix{UInt64}, V::Matrix{UInt64}, n::Int)

Pack rows of (u|v) ∈ F₂^{2n} into two matrices of UInt64 words (one for u, one for v).
Each row becomes ceil(n/64) words; bit j maps to bit j of the word.
"""
function to_bitpacked_uv(rows_uv::AbstractMatrix{Bool})
    r, n2 = size(rows_uv)
    @assert iseven(n2) "need 2n columns"
    n = n2 >>> 1
    W = cld(n, 64)
    U = zeros(UInt64, r, W)
    V = zeros(UInt64, r, W)
    for i in 1:r
        for j in 1:n
            if rows_uv[i, j]
                U[i, (j-1)>>>6 + 1] |= (UInt64(1) << ((j-1) % 64))
            end
            if rows_uv[i, n+j]
                V[i, (j-1)>>>6 + 1] |= (UInt64(1) << ((j-1) % 64))
            end
        end
    end
    return U, V, n
end

"""
    span_bitpacked(patterns::AbstractMatrix{Bool}, basis::AbstractMatrix{UInt64}) -> Matrix{UInt64}

XOR-linear combination of `basis` rows selected by `patterns` (over F₂), in packed form.
- patterns: (B, r) Bool
- basis:    (r, W) UInt64
Returns:    (B, W) UInt64
"""
function span_bitpacked(patterns::AbstractMatrix{Bool}, basis::AbstractMatrix{UInt64})
    B, r = size(patterns)
    W = size(basis, 2)
    out = zeros(UInt64, B, W)
    r == 0 && return out
    for i in 1:r
        mask = patterns[:, i]
        if any(mask)
            out[mask, :] .= out[mask, :] .⊻ view(basis, i, :)
        end
    end
    out
end

"""
    counts_by_type_bitpacked(U::Matrix{UInt64}, V::Matrix{UInt64}, n::Int) -> (NI,NX,NZ,NY)::NTuple{4,Vector{Int}}

For each packed row (U[i,:], V[i,:]), compute counts:
  NX: X-only (1,0), NZ: Z-only (0,1), NY: Y (1,1), NI: I (0,0).
Uses hardware popcount per word and sums over words.
"""
function counts_by_type_bitpacked(U::AbstractMatrix{UInt64}, V::AbstractMatrix{UInt64}, n::Int)
    B = size(U, 1)
    NX = zeros(Int, B)
    NZ = zeros(Int, B)
    NY = zeros(Int, B)
    for i in 1:B
        for w in 1:size(U,2)
            xmask = U[i,w] & ~V[i,w]
            zmask = ~U[i,w] & V[i,w]
            ymask = U[i,w] & V[i,w]
            NX[i] += count_ones(xmask)
            NZ[i] += count_ones(zmask)
            NY[i] += count_ones(ymask)
        end
    end
    NI = n .- (NX .+ NZ .+ NY)
    return NI, NX, NZ, NY
end

end # module

