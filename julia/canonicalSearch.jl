using Combinatorics

# Modular inverse in Z4
function invmod4(x)
    if x == 1
        return 1
    elseif x == 3
        return 3
    else
        error("No modular inverse for $x in Z4")
    end
end

# Compute RREF over Z4
function rref_mod4(M)
    M = copy(M)
    rows, cols = size(M)
    lead = 1
    for r in 1:rows
        if lead > cols
            return M
        end
        i = r
        while M[i, lead] == 0
            i += 1
            if i > rows
                i = r
                lead += 1
                if lead > cols
                    return M
                end
            end
        end
        # Swap rows
        M[i, :], M[r, :] = M[r, :], M[i, :]
        pivot = M[r, lead]
        if pivot != 0 && pivot != 2  # Only normalize if invertible
            inv = invmod4(pivot)
            M[r, :] .= (M[r, :] .* inv) .% 4
        end
        for j in 1:rows
            if j != r
                factor = M[j, lead]
                M[j, :] .= (M[j, :] .- factor .* M[r, :]) .% 4
            end
        end
        lead += 1
    end
    return M
end

# Canonical form: RREF + sorted rows/cols
function canonical_form(M)
    M = rref_mod4(M)
    rows_sorted = sort(collect(eachrow(M)))
    cols_sorted = sort(collect(eachcol(hcat(rows_sorted...))))
    return join(vec(cols_sorted), ",")
end

function unique_matrices(K, N)
    all_rows = collect(Iterators.product(fill(0:3, N)...))
    seen = Set{String}()
    for rows in combinations(all_rows, K)
        M = hcat(rows...)
        cf = canonical_form(M)
        push!(seen, cf)
    end
    return seen
end

# Example: 2x3
unique_set = unique_matrices(2, 3)
println("Number of unique canonical forms: ", length(unique_set))