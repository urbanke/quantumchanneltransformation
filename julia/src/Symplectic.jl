module Symplectic

export symp_inner, check_commutes, Jmul, xorrows

"""
    symp_inner(x::AbstractVector{Bool}, y::AbstractVector{Bool}) -> Bool

Symplectic inner product ⟨x,y⟩ = x J yᵀ over F₂ for 2n-bit rows x=(u|v), y=(u'|v').
This equals (u⋅v' ⊕ v⋅u') mod 2.
"""
function symp_inner(x::AbstractVector{Bool}, y::AbstractVector{Bool})
    n2 = length(x); @assert n2 == length(y) "length mismatch"
    @assert iseven(n2)
    n = n2 >>> 1
    u  = view(x, 1:n)
    v  = view(x, n+1:n2)
    up = view(y, 1:n)
    vp = view(y, n+1:n2)
    # dot over F2:
    d1 = isodd(count(==(true), u .& vp))
    d2 = isodd(count(==(true), v .& up))
    return xor(d1, d2)
end

"""
    check_commutes(A::AbstractMatrix{Bool}) -> Bool

Check that all rows of A mutually commute (A J Aᵀ = 0).
"""
function check_commutes(A::AbstractMatrix{Bool})
    m = size(A,1)
    for i in 1:m, j in 1:m
        if symp_inner(view(A,i,:), view(A,j,:))
            return false
        end
    end
    true
end

"""
    Jmul(x::AbstractVector{Bool}) -> Vector{Bool}

Multiply a 2n-bit row x=(u|v) by the standard J (right-multiplication), i.e.,
x J = (v | u). (Sufficient for syndrome maps like H J gᵀ.)
"""
function Jmul(x::AbstractVector{Bool})
    n2 = length(x); @assert iseven(n2); n = n2>>>1
    u = view(x,1:n); v=view(x,n+1:n2)
    [v; u]
end

"""
    xorrows(pattern::AbstractVector{Bool}, basis::AbstractMatrix{Bool}) -> Vector{Bool}

GF(2) linear combination of rows in `basis` selected by `pattern`.
"""
function xorrows(pattern::AbstractVector{Bool}, basis::AbstractMatrix{Bool})
    @assert length(pattern) == size(basis,1)
    out = falses(size(basis,2))
    for (i,b) in enumerate(pattern)
        if b
            @inbounds out .= xor.(out, view(basis,i,:))
        end
    end
    out
end


function sanity_check(H::AbstractMatrix{Bool},Lx::AbstractMatrix{Bool},Lz::AbstractMatrix{Bool},G::AbstractMatrix{Bool})
    T = vcat(H,Lx,G,Lz) 
    m = size(T,1)
    n = div(size(T,2),2)
    failure = 0 
    for i in 1:m
        for j in i:m
            if symp_inner(view(T,i,:), view(T,j,:)) & ((j!==i+n)) # this means it anticommutes at something that is not a pair(this is pair) 
               return false  
            end
            if !(symp_inner(view(T,i,:), view(T,j,:))) & ((j==i+n)) # this means it commutes at the opposite pair (this is bad)
                return false
            end
        end
    end
    true
end 

function build_from_stabs(StabString::Vector{String}) #  
    k = size(StabString,1)
    n = length(StabString[1])
    S = falses(k, 2*n)  # k stabilizers, 2n columns   
    m = size(StabString,1) 
    for i in 1:m
        stab = StabString[i]
        for j in 1:n
            s = stab[j]
            print(s)
            if s == 'X' || s == 'Y'
                S[i,j] = true 
            end 
            if s == 'Z' || s == 'Y' 
                S[i,j+n] = true 
            end 
        end
        println()
    end
    return S
end



end # module

