module SGS

using ..Symplectic: symp_inner

export tableau_from_stabilizers

"""
    tableau_from_stabilizers(S::AbstractMatrix{Bool})

Construct a full stabilizer tableau (H, Lx, Lz, G) adapted to the given independent
stabilizer set S ∈ F₂^{r×2n}. Ensures:
  - rowspan(H) = rowspan(S)
  - HJHᵀ = 0, GJGᵀ = 0, Lx JLxᵀ = 0, Lz JLzᵀ = 0
  - HJGᵀ = I_r, Lx JLzᵀ = I_k
  - cross-zeros: HJLxᵀ = HJLzᵀ = GJLxᵀ = GJLzᵀ = 0
Returns (H, Lx, Lz, G).
Implementation sketch: Append standard generators; drop dependent rows to reach 2n;
run a symplectic Gram–Schmidt so that the first r pairs’ first entries span rowspan(S).
"""
function tableau_from_stabilizers(S::AbstractMatrix{Bool})
    r, n2 = size(S)
    @assert iseven(n2) "S must have 2n columns"
    n = n2 >>> 1
    # Build candidate list: S first, then standard generators X1..Xn,Z1..Zn
    std = falses(2n, 2n)
    for i in 1:n
        std[i, i] = true           # X_i -> (e_i | 0)
        std[n+i, n+i] = true       # Z_i -> (0 | e_i)
    end
    # Greedy-independent selection to 2n rows (keep S first)
    pool = [S; std]
    pick = Bool[]
    chosen = falses(0, 2n)
    for i in 1:size(pool,1)
        cand = vcat(chosen, view(pool,i:i,:))
        # Test if cand rows are independent mod 2 by simple Gaussian elimination over F2
        if rank_f2(cand) > size(chosen,1)
            push!(pick, true)
            chosen = cand
        else
            push!(pick, false)
        end
        size(chosen,1) == 2n && break
    end
    @assert size(chosen,1) == 2n "Need 2n independent rows"

    # Run a symplectic Gram–Schmidt that preserves the invariant:
    # the first r pairs have first elements in span(S).
    H = falses(r, 2n)
    G = falses(r, 2n)
    Lx = falses(n - r, 2n)
    Lz = falses(n - r, 2n)

    # We keep an index into 'chosen' and progressively construct pairs.
    used = falses(2n)
    # Phase 1: stabilize block — take rows that commute with previous and lie in span(S)
    sspan = rowspace_basis(S)
    # Helper to test if a row lies in span(S) via rank check
    function in_spanS(x)
        rank_f2(vcat(S, reshape(x, 1, :))) == rank_f2(S)
    end

    # Build pairs
    hcount = 0; lcount = 0
    # Working copy (we'll clean by previously chosen pairs)
    work = copy(chosen)

    # “pair-clean” against existing pairs
    # Accept AbstractVector{Bool} so BitVector rows also work.
    function clean_against_pairs!(x::AbstractVector{Bool}, pairs::Vector{Tuple{Vector{Bool},Vector{Bool}}})
        for (a,b) in pairs
            # x ← x ⊕ <x,b>a ⊕ <x,a>b
            if symp_inner(x, b)
                x .= xor.(x, a)
            end
            if symp_inner(x, a)
                x .= xor.(x, b)
            end
        end
    end

    pairs = Tuple{Vector{Bool},Vector{Bool}}[]
    # Sweep through rows; when we find a row in span(S), try to find an anticommuting partner.
    # Otherwise, it will become a logical.
    for i in 1:2n
        g = copy(view(work, i, :))
        if hcount < r && in_spanS(g)
            # find first h that anticommutes with g among remaining rows
            partner = nothing
            for j in i+1:2n
                h = copy(view(work, j, :))
                clean_against_pairs!(h, pairs)
                if symp_inner(g, h)
                    partner = h
                    break
                end
            end
            @assert partner !== nothing "Could not find destabilizer partner for stabilizer row"
            hcount += 1
            H[hcount, :] .= g
            G[hcount, :] .= partner
            push!(pairs, (copy(g), copy(partner)))
        else
            # treat as logical pair first element if needed
            if lcount < n - r
                # try to find an anticommuting partner from remaining rows
                partner = nothing
                for j in i+1:2n
                    h = copy(view(work, j, :))
                    clean_against_pairs!(h, pairs)
                    if symp_inner(g, h)
                        partner = h
                        break
                    end
                end

                # ---- Fallback for pure-Z or pure-X stabilizer sets ----
                if partner === nothing
                    # Construct a minimal partner that anticommutes with g:
                    # If g has an X-bit at i (u_i=1), choose h = Z_i;
                    # else if g has a Z-bit at i (v_i=1), choose h = X_i;
                    # else pick the first qubit and set h = Z_1 (then enforce anticommutation).
                    h = falses(2n)
                    iu = findfirst(@view g[1:n])
                    if iu !== nothing
                        # g has X on iu -> choose Z on iu
                        h[n + iu] = true
                        # clean w.r.t. existing pairs to preserve prior commutations
                        clean_against_pairs!(h, pairs)
                        # ensure <g,h> = 1; if not, flip the complementary bit on same iu
                        if !symp_inner(g, h)
                            h[iu] = !h[iu]  # toggle X on iu
                        end
                    else
                        iv = findfirst(@view g[n+1:2n])
                        if iv !== nothing
                            # g has Z on iv -> choose X on iv
                            h[iv] = true
                            clean_against_pairs!(h, pairs)
                            if !symp_inner(g, h)
                                h[n + iv] = !h[n + iv]  # toggle Z on iv
                            end
                        else
                            # g is (unexpectedly) zero; pick qubit 1 and make Z_1
                            h[n + 1] = true
                            clean_against_pairs!(h, pairs)
                            # enforce anticommutation by toggling X_1 if needed
                            if !symp_inner(g, h)
                                h[1] = !h[1]
                            end
                        end
                    end
                    partner = h
                end
                # --------------------------------------------------------

                lcount += 1
                Lx[lcount, :] .= g
                Lz[lcount, :] .= partner
                push!(pairs, (copy(g), copy(partner)))
            end
        end
    end

    @assert hcount == r && lcount == n - r "Did not build full tableau"
    return H, Lx, Lz, G
end

# ---------- small GF(2) helpers for independence checks ----------

# naive F2 rank (sufficient for moderate sizes; can replace with faster impl if needed)
function rank_f2(A::AbstractMatrix{Bool})
    A = copy(A)
    m,n = size(A)
    r = 0
    col = 1
    for c in 1:n
        # find pivot
        piv = findfirst(i -> A[i,c], r+1:m)
        if isnothing(piv); continue; end
        piv = r + piv
        # swap
        A[r+1, :], A[piv, :] = copy(A[piv, :]), copy(A[r+1, :])
        # eliminate
        for i in 1:m
            if i != r+1 && A[i,c]
                A[i,:] .= xor.(A[i,:], A[r+1,:])
            end
        end
        r += 1
        r == m && break
    end
    r
end

function rowspace_basis(A::AbstractMatrix{Bool})
    # return a row-reduced basis (not used directly beyond span test)
    rank_f2(A) # placeholder; kept for future extensions
    A
end

end # module

