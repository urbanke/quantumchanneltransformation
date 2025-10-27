# 5-qubit code, r = 4; k = 1
# Pauli vector format is (u | v) with length 2n
#  - X on qubit i -> u_i = 1, v_i = 0
#  - Z on qubit i -> u_i = 0, v_i = 1

include("src/Symplectic.jl")
include("src/SGS.jl")

using QECInduced, .Symplectic, .SGS




function TestAllNK(n,r) #  
    pz = findZeroRate(f, 0, .5;maxiter=1000, ChannelType = "Depolarizing")
    S_best = falses(r, 2*n)  # r = n-k stabilizers, 2n columns   
    totalBits = 2*n*r # it will be r rows of 2n each 
    j = 0 
    hb_best = -10
    for bits in Iterators.product(fill(0:1, r*n*2)...)
        S = Matrix{Bool}(reshape(collect(bits),r,2*n))
        if SGS.rank_f2(S) == r
            if Symplectic.valid_code(S)
                hb_temp = QECInduced.check_induced_channel(S)
                if hb_temp >= hb_best
                    hb_best = hb_temp
                    S_best = S 
                end
            end
        end
        j += 1 
        if j % 100000 == 0 
            println(j)
            println(pz)
            println(hb_best)
            println(S_best)
        end
    end
end


TestAllNK(6,5)


