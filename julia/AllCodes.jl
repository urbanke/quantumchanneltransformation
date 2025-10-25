# 5-qubit code, r = 4; k = 1
# Pauli vector format is (u | v) with length 2n
#  - X on qubit i -> u_i = 1, v_i = 0
#  - Z on qubit i -> u_i = 0, v_i = 1

include("src/Symplectic.jl")


using QECInduced, .Symplectic

function dec2bin(int; n = 8) 
    quot = int 
    rem = 1 
    seq = zeros((1,n))
    i = 0 
    while(quot != 0)
        rem = quot%2 
        quot = div(quot,2)
        seq[n-i] = rem 
        i += 1 
    end
    return seq
end 


function TestAllNK(n,r) #  
    S = falses(r, 2*n)  # r = n-k stabilizers, 2n columns   
    totalBits = 2*n*r # it will be r rows of 2n each 
    for bits in 0:(2^totalBits-1)
        s = dec2bin(bits; n = totalBits)
        S = Matrix{Bool}(reshape(s,r,2*n))
        if Symplectic.valid_code(S)
            if QECInduced.check_induced_channel(S)
                println(S)
            end
        end
    end
end


TestAllNK(5,4)






end  

