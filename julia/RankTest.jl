include("src/Symplectic.jl")
include("src/SGS.jl")

using QECInduced, .Symplectic, .SGS

m, n = 3, 4
S = Matrix{Bool}(rand(0:1, m, n))
@show S
@show SGS.rank_f2(S) 
