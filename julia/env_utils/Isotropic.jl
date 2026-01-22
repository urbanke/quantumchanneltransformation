module Isotropic
export random_isotropic_basis_with_structure, All_Codes_Random_SGS


include("../src/Symplectic.jl")
include("../src/SGS.jl")


using .Symplectic, .SGS
using QECInduced
using Base.Threads
using Plots
using LinearAlgebra
using Random




# utils 



# ---- GF(2) row operations on BitMatrix (safe, no slicing pitfalls) ----

@inline function xor_row!(A::BitMatrix, dst::Int, src::Int)
    n = size(A, 2)
    @inbounds for j in 1:n
        A[dst, j] ⊻= A[src, j]
    end
end

@inline function swap_rows!(A::BitMatrix, r1::Int, r2::Int)
    r1 == r2 && return
    n = size(A, 2)
    @inbounds for j in 1:n
        A[r1, j], A[r2, j] = A[r2, j], A[r1, j]
    end
end

"""
    gf2_nullspace(H::BitMatrix) -> BitMatrix

Returns a matrix N whose rows form a basis for the nullspace of H over GF(2),
i.e. H * N' = 0 (mod 2).
"""
function gf2_nullspace(H::BitMatrix)
    m, n = size(H)
    A = copy(H)

    pivot_cols = Int[]
    pivot_rows = Int[]
    row = 1

    # Reduced row echelon form over GF(2)
    for col in 1:n
        row > m && break

        # find pivot row at/below `row` with a 1 in this column
        pivot = 0
        @inbounds for r in row:m
            if A[r, col]
                pivot = r
                break
            end
        end
        pivot == 0 && continue

        swap_rows!(A, row, pivot)

        push!(pivot_cols, col)
        push!(pivot_rows, row)

        # eliminate from all other rows
        @inbounds for r in 1:m
            if r != row && A[r, col]
                xor_row!(A, r, row)
            end
        end

        row += 1
    end



    free_cols = setdiff(1:n, pivot_cols)
    N = BitMatrix(undef, length(free_cols), n)

    # Build nullspace basis vectors
    for (i, fc) in enumerate(free_cols)
        v = falses(n)
        v[fc] = true
        # pivot var = sum(A[pivot_row, free_col]*free_var)
        for (pr, pc) in zip(pivot_rows, pivot_cols)
            if A[pr, fc]
                v[pc] = true
            end
        end
        N[i, :] = v
    end

    return N
end



"""
Return the index of the first `true` bit in v, or 0 if all false.
(We use the leftmost pivot convention.)
"""
function first_one(v::BitVector)
    for i in 1:length(v)
        if v[i]
            return i
        end
    end
    return 0
end

"""
Reduce v in-place using an echelon basis over GF(2).
basis maps pivot_index -> pivot_row (BitVector).
"""
function gf2_reduce!(v::BitVector, basis::Dict{Int, BitVector})
    while true
        p = first_one(v)
        p == 0 && return v
        if haskey(basis, p)
            v .⊻= basis[p]   # XOR eliminate pivot
        else
            return v
        end
    end
end

"""
Try to add v to the echelon basis (over GF(2)).
Returns true iff v is linearly independent of the current span.
Stores a reduced pivot row in the basis if independent.
"""
function gf2_try_add!(basis::Dict{Int, BitVector}, v::BitVector)
    w = copy(v)
    gf2_reduce!(w, basis)
    p = first_one(w)
    p == 0 && return false
    basis[p] = w
    return true
end



"""
Randomly mix the rows of N (BitMatrix) using invertible GF(2) row operations.
This preserves the row span and row independence while making the basis "random looking".
"""
function gf2_random_row_mix!(N::BitMatrix, rng)
    d, n = size(N)
    if d <= 1
        return N
    end

    # Some random row swaps
    for _ in 1:(2d)
        i = rand(rng, 1:d)
        j = rand(rng, 1:d)
        if i != j
            swap_rows!(N, i, j)
        end
    end

    # Some random row XOR operations: row_i ^= row_j
    for _ in 1:(4d)
        i = rand(rng, 1:d)
        j = rand(rng, 1:d)
        if i != j
            xor_row!(N, i, j)
        end
    end

    return N
end


"""
    solve_linear_system_f2_with_random_gf2(A::BitMatrix, b::BitVector, rng) -> BitVector

Solve Ax = b (mod 2) with randomization of free variables.
Returns a solution vector, or nothing if no solution exists.
"""
function solve_linear_system_f2_with_random_gf2(A::BitMatrix, b::BitVector, rng)
    m, n = size(A)
    
    # Create augmented matrix [A | b]
    Aug = hcat(copy(A), reshape(b, m, 1))
    
    # Gaussian elimination (row echelon form)
    pivot_cols = Int[]
    current_row = 1
    
    for col in 1:n
        # Find pivot in current column
        pivot_row = findfirst(i -> Aug[i, col], current_row:m)
        
        if pivot_row === nothing
            continue  # No pivot in this column - free variable
        end
        
        pivot_row += current_row - 1
        
        # Swap rows if needed
        if pivot_row != current_row
            Aug[current_row, :], Aug[pivot_row, :] = Aug[pivot_row, :], Aug[current_row, :]
        end
        
        # Eliminate below and above
        for i in 1:m
            if i != current_row && Aug[i, col]
                Aug[i, :] .= xor.(Aug[i, :], Aug[current_row, :])
            end
        end
        
        push!(pivot_cols, col)
        current_row += 1
        
        if current_row > m
            break
        end
    end
    
    # Check for inconsistency (row like [0 0 ... 0 | 1])
    for i in 1:m
        if all(.!Aug[i, 1:n]) && Aug[i, n+1]
            return nothing  # No solution
        end
    end
    
    # Back substitution with random free variables
    x = falses(n)
    free_vars = setdiff(1:n, pivot_cols)
    
    # Randomize free variables
    for var in free_vars
        x[var] = rand(rng, Bool)
    end
    
    # Solve for pivot variables in reverse order
    for col in reverse(pivot_cols)
        row = findfirst(i -> Aug[i, col], 1:m)
        
        # x[col] should satisfy: x[col] + sum(Aug[row, j] * x[j] for j > col) = Aug[row, n+1]
        val = Aug[row, n+1]
        for j in (col+1):n
            if Aug[row, j] && x[j]
                val = !val
            end
        end
        x[col] = val
    end
    
    return x
end

# test

function assert_isotropic(S::BitMatrix, n::Int)
    m = size(S, 1)
    for i in 1:m, j in i+1:m
        u = BitVector(S[i, :])
        v = BitVector(S[j, :])
        @assert !symplectic_dot(u, v, n) "Rows $i and $j anticommute!"
    end
    return true
end

@inline function symplectic_dot(u::BitVector, v::BitVector, n::Int)
    parity = false
    @inbounds for i in 1:n
        # add mod 2: (x_i & z'_i) + (z_i & x'_i)
        parity ⊻= (u[i] & v[n+i]) ⊻ (u[n+i] & v[i])
    end
    return parity
end

# main 

function random_isotropic_basis_with_structure(n::Int, s::Int, r::Int; rng = Random.default_rng())
    @assert s ≤ n "Maximum isotropic dimension is n"
    @assert r ≤ s "Must have r ≤ s"
    @assert r ≤ n "Must have r ≤ n (need r qubits for identity)"

    rows  = BitVector[]
    basis = Dict{Int, BitVector}()  # GF(2) echelon basis for independence checks

    # -------------------------------
    # First r rows: enforce I in X-part
    # -------------------------------
    for i in 1:r
        success = false
        attempts = 0

        while !success
            attempts += 1
            attempts > 2000 && error("Could not generate independent structured row $i after many attempts")

            v = falses(2n)

            # X-part: identity bit + random A bits
            v[i] = true
            for j in (r+1):n
                v[j] = rand(rng, Bool)
            end

            # Choose Z-part to satisfy isotropy with existing rows
            if !isempty(rows)
                H_z = BitMatrix(undef, length(rows), n)
                b_z = falses(length(rows))

                for (eq, u) in enumerate(rows)
                    # coefficients are u_x
                    H_z[eq, :] .= view(u, 1:n)

                    # RHS = v_x · u_z
                    rhs = false
                    for k in 1:n
                        if v[k] && u[n + k]
                            rhs = !rhs
                        end
                    end
                    b_z[eq] = rhs
                end

                v_z = solve_linear_system_f2_with_random_gf2(H_z, b_z, rng)
                v_z === nothing && continue  # resample A and try again

                v[n+1:2n] .= v_z
            else
                # first row: free Z-part
                for j in (n+1):2n
                    v[j] = rand(rng, Bool)
                end
            end

            # Independence check (full 2n vector)
            if gf2_try_add!(basis, v)
                push!(rows, v)
                success = true
            end
        end
    end
    # -----------------------------------------
    # Remaining s-r rows: pure-Z only (X=0)
    # (Compute nullspace ONCE, randomize basis, and take independent rows)
    # -----------------------------------------
    if s > r
        # Build H_z from the X parts of the rows (these are the only nonzero X parts)
        H_z = BitMatrix(undef, length(rows), n)
        for (eq, u) in enumerate(rows)
            H_z[eq, :] .= view(u, 1:n)
        end

        # Compute nullspace once
        N_z = gf2_nullspace(H_z)   # d × n, rows are a basis for the nullspace
        d = size(N_z, 1)
        d == 0 && error("No remaining pure-Z isotropic directions")

        # Make sure there is enough room
        @assert (s - r) <= d "Not enough independent pure-Z directions: need $(s-r), have $d"

        # Optional: randomize the basis for more randomness
        gf2_random_row_mix!(N_z, rng)

        # Take first (s-r) basis vectors (each is independent by construction)
        for t in 1:(s - r)
            z_part = BitVector(N_z[t, :])
            v = vcat(falses(n), z_part)
            # We do not need gf2_try_add! here because rows of N_z are independent
            push!(rows, v)
        end
    end

    # Convert to matrix
    S = BitMatrix(undef, s, 2n)
    for i in 1:s
        S[i, :] = rows[i]
    end

    return S
end

function All_Codes_Random_SGS(channelParamFunc, n, k, r, p_range; newBest = nothing, trials = 1e7, rng = MersenneTwister(2025), concated = nothing, placement = "inner") 
    s = n - k  # Number of rows in the (n-k) × (2n) matrix
    
    # Initialize best trackers for each grid point
    points = length(p_range)
    S_best = [falses(s, 2n) for _ in 1:points]  # Best matrix at each grid point
    

    if newBest === nothing 
        hb_best = QECInduced.sweep_hashing_grid(p_range, channelParamFunc)
    else 
        hb_best = newBest
    end 

    println("=" ^ 70)
    println("Generating binary matrices ($s × $(2*n)) in standard block form")
    println("Parameters: n=$n, k=$k, s=$s, r=$r, grid_points=$points")
    println("pz range: [$(p_range[1]), $(p_range[end])]")
    println("=" ^ 70)
    
    println("Total matrices to be randomly searched: $trials")    
    
    count = 0
    last_print_count = 0
    print_interval = max(1, div(trials, 20))  # Print ~20 times
    
    println("\nStarting enumeration with orthogonality constraints...\n")
    
    for i in 1:trials
        try 
            M = random_isotropic_basis_with_structure(n, s, r; rng = rng)
            count += 1
            
            # Convert to Bool matrix
            S = Matrix{Bool}(M)
            
            if !isnothing(concated) 
                if placement == "inner"
                    S = concat_stabilizers_bool(S, concated)
                else 
                    S = concat_stabilizers_bool(concated, S)
                end
            end
            # Check the induced channel at all grid points
            hb_grid = QECInduced.check_induced_channel(S, 0, channelParamFunc; sweep=true, ps=p_range)
            
            # Find which grid points improved
            improved_indices = findall(hb_grid .> (hb_best .+ eps()))
            
            # Update best for each improved point
            if !isempty(improved_indices)
                for idx in improved_indices
                    hb_best[idx] = hb_grid[idx]
                    S_best[idx] = copy(S)
                end
                
                println("\n" * "=" ^ 70)
                println("NEW BEST FOUND! (Matrix #$count)")
                println("Improved at $(length(improved_indices)) grid point(s): $improved_indices")
                println("\nGrid point details:")
                for idx in improved_indices
                    println("  Point $idx: pz=$(round(p_range[idx], digits=4)), hb=$(round(hb_best[idx], digits=6))")
                end
                println("\nS_best (showing first improved point) =")
                println(Symplectic.build_from_bits(S_best[improved_indices[1]]))
                println("=" ^ 70 * "\n")
            end
        catch e 
            continue
        end
    end
    
    println("\n" * "=" ^ 70)
    println("SEARCH COMPLETE")
    println("=" ^ 70)
    println("Valid matrices found (satisfying orthogonality): $count")
        
    return hb_best, S_best
end



end #module

