include("src/Symplectic.jl")
include("src/SGS.jl")

using Random
using QECInduced, .Symplectic, .SGS

# Build an invertible L×L binary scrambler matrix A over GF(2),
# returned explicitly as Matrix{Bool}.
function build_scrambler(L::Int; rng=Random.default_rng())
    # Step 1: make random upper-triangular U with ones on diag
    # We force Matrix{Bool} here (NOT BitMatrix) by using Array{Bool}
    U = Array{Bool}(undef, L, L)
    @inbounds for i in 1:L
        # fill row i
        # below diagonal = 0
        for j in 1:(i-1)
            U[i,j] = false
        end
        # diagonal = 1
        U[i,i] = true
        # above diagonal = random Bool
        for j in (i+1):L
            U[i,j] = rand(rng, Bool)
        end
    end

    # Step 2: random row/col permutations
    rowperm = randperm(rng, L)
    colperm = randperm(rng, L)

    # A will also be Matrix{Bool}
    A = Array{Bool}(undef, L, L)
    @inbounds for new_i in 1:L
        i = rowperm[new_i]
        for new_j in 1:L
            j = colperm[new_j]
            A[new_i, new_j] = U[i,j]
        end
    end

    return A  # Matrix{Bool}, guaranteed invertible mod 2
end


# Multiply A * inbits (mod 2) over GF(2).
# Accept any boolean matrix type just in case.
function apply_scrambler!(
    outbits::Vector{Bool},
    A::AbstractMatrix{Bool},
    inbits::Vector{Bool},
)
    L = length(inbits)
    @inbounds for i in 1:L
        acc = false
        # acc = XOR_j (A[i,j] & inbits[j])
        for j in 1:L
            if A[i,j] && inbits[j]
                acc = !acc
            end
        end
        outbits[i] = acc
    end
    return outbits
end


function TestAllNK_scrambled(n::Int, r::Int)
    # Optional reproducibility for the scrambler:
    # Random.seed!(12345)

    totalBits = 2 * n * r  # L
    L = totalBits

    # Make the scrambler once
    A = build_scrambler(L)

    # Work buffers
    inbits_vec  = Vector{Bool}(undef, L)
    outbits_vec = Vector{Bool}(undef, L)

    hb_best = -1.0e9
    S_best  = falses(r, 2n)  # BitMatrix is fine here

    # whatever this is in your codebase
    pz = findZeroRate(f, 0, 0.5; maxiter=1000, ChannelType="Independent")

    j = 0

    # Enumerate all 2^(2nr) binary strings of length L
    for bits_tuple in Iterators.product(fill(0:1, L)...)

        # tuple -> Vector{Bool}
        @inbounds @simd for idx in 1:L
            inbits_vec[idx] = (bits_tuple[idx] == 1)
        end

        # Scramble: outbits_vec = A * inbits_vec (mod 2)
        apply_scrambler!(outbits_vec, A, inbits_vec)

        # Reshape scrambled flat vector into r × (2n) candidate stabilizer matrix S
        # Note: reshape(outbits_vec, r, 2n) will give Array{Bool,2} if outbits_vec
        # is Vector{Bool}, which is good (Matrix{Bool}).
        S = reshape(outbits_vec, r, 2n)

        # rank over F₂ must be r
        if SGS.rank_f2(S) == r
            # commutation / valid stabilizer check
            if Symplectic.valid_code(S)
                hb_temp = QECInduced.check_induced_channel(S,pz)
                if hb_temp >= hb_best
                    hb_best = hb_temp
                    S_best  = copy(S)
                end
            end
        end

        # heartbeat
        j += 1
        if j % 100000 == 0
            println("j = ", j)
            println("pz = ", pz)
            println("hb_best = ", hb_best)
            println("S_best = ")
            println(S_best)
        end
    end

    println("DONE.")
    println("Best hb = ", hb_best)
    println("Best S = ")
    println(S_best)
end

# call
TestAllNK_scrambled(6, 5)

