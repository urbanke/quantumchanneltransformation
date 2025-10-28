include("src/Symplectic.jl")
include("src/SGS.jl")

using .Symplectic, .SGS


"""
Iterate through all possible binary matrices of size (n, r) one row at a time,
with early termination based on row compatibility checks.

Parameters:
- n: number of rows
- r: number of columns
- verify_fn: function(matrix, current_row) that returns true if rows 1:current_row are compatible

The verify function receives the partial matrix and the index of the row just added.
Return false to skip all matrices that extend this partial configuration.
"""
function iterate_binary_matrices_with_check(n::Int, r::Int, verify_fn::Function)
    Channel() do ch
        # Build matrix recursively, row by row
        matrix = zeros(Int, n, r)
        
        function build_rows(row_idx::Int)
            if row_idx > n
                # All rows filled and verified - yield complete matrix
                put!(ch, copy(matrix))
                return
            end
            
            # Try all 2^r possible values for this row
            for i in 0:(2^r - 1)
                # Fill current row
                temp = i
                for col in r:-1:1
                    matrix[row_idx, col] = temp & 1
                    temp >>= 1
                end
                
                # Verify compatibility with previous rows
                if verify_fn(matrix, row_idx)
                    # If valid, continue to next row
                    build_rows(row_idx + 1)
                end
                # If invalid, skip entire subtree (all extensions of this partial matrix)
            end
        end
        
        build_rows(1)
    end
end

# Example usage with verification
function demo_with_verification()
    n, r = 4, 10
    
    # Example verification: code should stabilize up to current row 
    function good_code(matrix, current_row)
        S = Matrix{Bool}(matrix[1:current_row,:])
        if current_row == 1
            return true  # First row always valid
        end
        return (Symplectic.valid_code(S)) & (SGS.rank_f2(S) == current_row)
    end
    
    println("Generating binary matrices ($n × $r) in a depth-first stabilizing approach:\n")
    
    count = 0
    total_possible = 2^(n*r)
    println("Total possible: $total_possible")
    for matrix in iterate_binary_matrices_with_check(n, r, good_code)
        count += 1
        if count <= 5  # Show first few
            println("Valid matrix $count:")
            display(matrix)
            println()
        end
    end
    
    println("Valid matrices found: $count")
    println("Efficiency gain: $(round((1 - count/total_possible)*100, digits=1))% pruned")
end

demo_with_verification()