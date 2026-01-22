include("../src/Symplectic.jl")
using .Symplectic


# this is doing the entire ML decoding hashing bound calculator 
# it first goes through all 4^n paulies and groups by syndromes 
# then it goes through each syndrome and finds the ML decoder logical it should apply 
# then it goes through all 4^n paulies, applies the syndrome offset and logical offset, and gets induced distribution (for each k logical) 

# chat gpt code, cannot vouch for it  (this block only)

function logical_pauli_from_bits(bits::Vector{Bool})
    # bits length = 2*L, first L are X flags, next L are Z flags
    L = div(length(bits), 2)
    xs = bits[1:L]
    zs = bits[L+1:2L]
    s = IOBuffer()
    for j in 1:L
        if xs[j] & zs[j]
            print(s, "Y")
        elseif xs[j]
            print(s, "X")
        elseif zs[j]
            print(s, "Z")
        else
            print(s, "I")
        end
    end
    return String(take!(s))
end

function logical_pauli_from_index(idx::Int, lilK::Int)
    bits = decimal_to_bin(idx, lilK)
    return logical_pauli_from_bits(bits)
end


function print_induced_channel(inducedDict, PS::Real, lilK::Int; top::Int=16)
    # Normalize
    norm = Dict(k => v/PS for (k,v) in inducedDict)

    # Sort by probability
    items = collect(norm)
    sort!(items, by = x -> -x[2])

    println("Logical induced channel P(L | s):")
    println("  (showing top $(min(top, length(items))) of $(length(items)))")
    for (i,(k,p)) in enumerate(items[1:min(top, length(items))])
        println(rpad("  " * logical_pauli_from_index(k, lilK), 12),
                "  idx=", lpad(string(k), 4),
                "  p=", p)
    end

    # Optional: sanity check sums to ~1
    s = sum(values(norm))
    println("  sum p = ", s)
end

function logical_marginal_one(inducedDict, PS::Real, lilK::Int, q::Int)
    L = div(lilK, 2)
    @assert 1 ≤ q ≤ L

    pI = 0.0; pX = 0.0; pZ = 0.0; pY = 0.0

    for (idx, w) in inducedDict
        p = w / PS
        bits = decimal_to_bin(idx, lilK)

        x = bits[q]
        z = bits[L + q]

        if x && z
            pY += p
        elseif x
            pX += p
        elseif z
            pZ += p
        else
            pI += p
        end
    end

    # optional sanity normalization against tiny FP drift
    s = pI + pX + pZ + pY
    return Dict("I"=>pI/s, "X"=>pX/s, "Z"=>pZ/s, "Y"=>pY/s)
end

function print_logical_marginals(inducedDict, PS::Real, lilK::Int)
    L = div(lilK, 2)
    for q in 1:L
        m = logical_marginal_one(inducedDict, PS, lilK, q)
        println("Logical qubit $q marginal: ",
                "I=$(m["I"]) | X=$(m["X"]) | Z=$(m["Z"]) | Y=$(m["Y"])")
    end
end

##### end of gpt code 

function printNice(S)
	print("[")
	for i in 1:size(S)[1] - 1
		print("\"")
	    print(build_from_bits(S[i,:]))
	    print("\", ")
	end 
	i = size(S)[1] 
	println("\"",build_from_bits(S[i,:]),"\"]")
end

function build_from_bits(S)
    n = div(length(S),2) 
    Xs = S[1:n]
    Zs = S[n+1:(2*n)]
        tempString = ""
        for j in 1:n 
            if Xs[j] & Zs[j]
                tempString = tempString*"Y" 
            elseif Xs[j] 
                tempString = tempString*"X" 
            elseif Zs[j] 
                tempString = tempString*"Z" 
            else 
                tempString = tempString*"I" 
            end
        end
    return tempString
end

function printDict(dict) 
	for k in sort(collect(keys(dict)))  
		print("$k => $(dict[k])")
		print("|")
	end
	println()
end

function calculateP(noise) # this function works (no need to look at)
	# calculate num of errors 
	I = 0; X = 0; Z = 0; Y = 0
	for j in 1:n  
		if !noise[j] & !noise[j+n] # 0,0 means I 
			I += 1
		elseif !noise[j+n] # 1,0 means X 
			X += 1
		elseif !noise[j] # 0,1 means Z 
			Z += 1
		else 
			Y += 1 
		end 
	end
	return pchannel[1]^I * pchannel[2]^X * pchannel[3]^Z * pchannel[4]^Y # this is the probability of the syndrome 
end 



function decimal_to_bin(number, width)
    out = Vector{Bool}(undef, width)
    for i in 1:width
        out[i] = ((number >> (width - i)) & 1) == 1
    end
    return out
end

function bin_to_dec(numberVec)
    out = 0
    width = length(numberVec)
    mult = 1
    for i in width:-1:1
        out += numberVec[i]*mult
        mult *= 2 
    end
    return Int(out)
end

# You need to construct the whole stabilizer tableau 
#=
Stabilizers = ["ZZ", "IZIIX", "IIZXI"]
#Stabilizers = ["ZXI", "IXX"]
logicalX = ["IIIXI","IIIIX"]
logicalZ = ["XIXZI","XXIIZ"]
G = ["XIIII", "IXIII", "IIXII"]
=# 



#= 
Stabilizers = ["XIX", "IXX"]
logicalX = ["IIX"]
logicalZ = ["ZZZ"]
G = ["ZII", "IZI"]
=# 

Stabilizers = ["ZIIIZ", "IZIIZ", "IIZIZ", "IIIZZ"] 
logicalX = ["IIIIZ"]
logicalZ = ["XXXXX"]
G = ["XIIII", "IXIII", "IIXII", "IIIXI"]

#Stabilizers = ["ZZZZZZ"]
#logicalX = ["IZZZZZ", "IIZZZZ", "IIIZZZ", "IIIIZZ", "IIIIIZ"]
#logicalZ = ["XXIIII", "IXXIII", "IIXXII", "IIIXXI", "IIIIXX"]
#G = ["XIIIII"]

AllLogicalString = vcat(logicalX, logicalZ)

S = Symplectic.build_from_stabs(Stabilizers)
G = Symplectic.build_from_stabs(G)


AllLogical = Symplectic.build_from_stabs(AllLogicalString)




global n = div(size(S)[2],2) 
global r = 2^size(Stabilizers)[1] 
global k = 2^size(AllLogicalString)[1] # logicalX and Z 
global m = size(Stabilizers)[1]
global lilK = size(AllLogical)[1]


function trueErrorRepair(G, S) 
	synds = zeros(Int,size(S)[1])
	idx = [1:size(S)[1];]
	GLUT = zeros(Int,size(S)[1])
	trueErrorRepair = Dict() 

	for j in 1:r
		idxSelector = Int.(decimal_to_bin(j, m))
		#println(idxSelector)
		#println(idx)
		idxPos = idxSelector .* idx
		idxPos = idxPos[idxPos .> 0]
		g = Bool.(vec(sum(Int.(G[idxPos,:]), dims = 1) .% 2)) # this gives all combinations of G
		for i in 1:size(S)[1]
			synds[i] = Int(Symplectic.symp_inner(g, S[i,:]))
		end
		dictNum = bin_to_dec(synds)
		trueErrorRepair[dictNum] = idxPos 
	end 
	return trueErrorRepair
end 

global trueRepairs = trueErrorRepair(G,S)

# The tableau shows us the two logicals are IIX and XZY 

# Code is [Z, X , I] [I, X, X]

p = 0.11002786443835955
px = 0.235
pz = px/9
p = 0.1905
p = 0.1904775

#Any["IZXXX", "ZIXIX", "IIXXI"]


#pchannel = [(1-p)*(1-p), p*(1-p), p*(1-p), p*p]
#pchannel = [(1 - px) * (1 - pz), px * (1 - pz), pz * (1 - px), px * pz] 
pchannel = [(1-p), p/3, p/3, p/3]
# calculate the probability of the syndromes 

SyndromeDict = Dict() # making a dict where the 4 syndromes currently have no prob 
AllRecieved = Dict()


for i in 0:r-1
	SyndromeDict[i] = 0.0 
	AllRecieved[i] = []
end


for i in 0:(4^n - 1) # there are 3 quaterary symbols so i will say there are 2^6 options 
	noise = decimal_to_bin(i, 2*n)
	synds = zeros(Int,size(S)[1])
	for i in 1:size(S)[1]
		synds[i] = Int(Symplectic.symp_inner(noise, S[i,:]))
	end
	dictNum = bin_to_dec(synds)

	# calculate num of errors 
	prob = calculateP(noise) 
	SyndromeDict[dictNum] += prob # this is the probability of the syndrome 
	push!(AllRecieved[dictNum], noise) # this gives all vectors relating to a certain syndrome 

	#println(SyndromeDict)
end 



function errorBasedOnSynd(recieved,PS, pchannel,AllLogical)

	#AllLogical = vcat(logicalX, logicalZ)
	logicalprob = 0 
	logicalDict = Dict() 

	for i in 0:k-1
		logicalDict[i] = 0.0
	end

	logicals = zeros(Int, lilK)

	for error in recieved 
		#println(error)
		for i in 1:lilK
			logicals[i] = Int(Symplectic.symp_inner(error, AllLogical[i,:])) # for every error get the logical 
		end
		dictNum = bin_to_dec(logicals)
		prob = calculateP(error) 
		#println(dictNum)
		logicalDict[dictNum] += prob # the total probability of each logical
	end
	maxVal = 0 
	maxKey = 0
	for i in 0:k-1
		logicalDict[i] = logicalDict[i]/PS
		if logicalDict[i] >= maxKey
			maxKey = logicalDict[i]
			maxVal = i 
		end
	end
	#println(logicalDict)
 	return maxVal # return the logical correction that has the most probability  a(s), b(s)
end


syndromeRepair = Dict() 
for i in 0:k-1
	syndromeRepair[i] = Int(0) 
end 

for i in 0:r-1
#	println("="^50,i,"="^50)
	syndromeRepair[i] = errorBasedOnSynd(AllRecieved[i],SyndromeDict[i],pchannel,AllLogical)
#	println()
end

function inducedChannel(recieved,PS, pchannel,AllLogical, repair)
	logicalprob = 0 
	repairs = decimal_to_bin(repair, lilK)
	inducedDict = Dict()
	for i in 0:k-1 
		inducedDict[i] = 0
	end 

	residuals = zeros(Int, lilK)
	for error in recieved

		synds = zeros(Int,size(S)[1])
		for i in 1:size(S)[1]
			synds[i] = Int(Symplectic.symp_inner(error, S[i,:])) # get which syndrome it will give 
		end
		dictNum = bin_to_dec(synds)
		idxPos = trueRepairs[dictNum]
		g = Bool.(vec(sum(Int.(G[idxPos,:]), dims = 1) .% 2)) # this gives all combinations of G
		#error = Bool.(vec((error .+ g) .% 2))

		for i in 1:lilK
			residuals[i] = Int(Symplectic.symp_inner(error, AllLogical[i,:])) ⊻ repairs[i] # the residual when we apply the ML correction 
		end 
		#print(build_from_bits(error), " || ")
		dictNum = bin_to_dec(residuals)
		#println("Residual Error Number: ", dictNum)
		# calculate num of errors 
		prob = calculateP(error) 
		inducedDict[dictNum] += prob
	end
	h = 0 
	inducedDictPrePS = copy(inducedDict)
	for i in 0:k-1
		inducedDict[i] = inducedDict[i]/PS
		h -= inducedDict[i]*log2(inducedDict[i])
	end

	#printDict(inducedDict)

	println("Syrdome: (",repairs,"), (k - H(p(a,b|s))/n = ",(div(lilK,2)-h)/n)
 	return h*PS, inducedDictPrePS
end

# total prob 
global totalProb = Dict()
for i in 0:k-1 
	totalProb[i] = 0.0
end 


global H = 0 
for i in 0:r-1
	println("="^50,i,"="^50)
	println("Syndrome Measured: ", i)
	println("Probability of Measuring Syndrome: ",  SyndromeDict[i])
	println("Logical Operator Applied: ", syndromeRepair[i])
	idxPos = trueRepairs[i]
	trueErrorRep = bin_to_dec(idxPos) 
	println("True Error Operator Applied: ", trueErrorRep)
	h_temp, p_temp = inducedChannel(AllRecieved[i],SyndromeDict[i], pchannel,AllLogical, syndromeRepair[i])
	print_induced_channel(p_temp, SyndromeDict[i], lilK; top=32)
	print_logical_marginals(p_temp, SyndromeDict[i], lilK)
	global H += h_temp
	global totalProb = mergewith(+, totalProb, p_temp)  # Dict(:a => 1.4, :b => 1.4)
	println()
end

global h_orig = 0 
for i in 1:4 
	global h_orig -= pchannel[i]*log2(pchannel[i])
end 

println("Original Channel: ", 1- h_orig)
println("Induced Channel: ", (div(lilK,2)-H)/n)
println(pchannel)
#printDict(totalProb)




function errorCheck(error, AllLogical, syndromeRepair, S; stringMode = true)
	if stringMode 
		error = vec(Symplectic.build_from_stabs(error))
	end 
	synds = zeros(Int,size(S)[1])
	for i in 1:size(S)[1]
		synds[i] = Int(Symplectic.symp_inner(error, S[i,:])) # get which syndrome it will give 
	end
	dictNum = bin_to_dec(synds)
	error = Bool.(vec((error .+ trueRepairs[dictNum]) .% 2))
	synds = zeros(Int,size(S)[1])
	for i in 1:size(S)[1]
		synds[i] = Int(Symplectic.symp_inner(error, S[i,:])) # get which syndrome it will give 
	end
	logicals = zeros(Int, lilK) 
	for i in 1:lilK
		logicals[i] = Int(Symplectic.symp_inner(error, AllLogical[i,:])) # get which logicals (if any) it turns on 
	end
	degen = 0
	if sum(logicals) == 0 
		degen = 1 
	#	println("Degenerate Error")
	end 
	#println(logicals)
	repair = syndromeRepair[dictNum]
	repairs = decimal_to_bin(repair, lilK)
	residuals = zeros(Int, lilK)
	logical = 0 
	for i in 1:lilK
		residuals[i] = Int(Symplectic.symp_inner(error, AllLogical[i,:])) ⊻ repairs[i] # the residual when we apply the ML correction 
	end 
	if sum(residuals) > 0 
		logical = 1 
	end
	#println(residuals)
	return degen, logical 
end 




############################### rudi calculations  #################
 

function inducedChannelFromRepCode(m,PI,PX,PZ,PY)
	pizp = PI + PZ
	pizm = PI - PZ 
	pxyp = PX + PY 
	pxym = PX - PY 
	pib = 0
	pzb = 0 
	for i in 0:Int(ceil(m/2))
		mci = binomial(m,i) 
		pib += mci*pizp^(m-i)*pxyp^(i) 
		pib += mci*pizm^(m-i)*pxym^(i) 

		pzb += mci*pizp^(m-i)*pxyp^(i) 
		pzb -= mci*pizm^(m-i)*pxym^(i) 
	end 
	pib = pib/2 
	pzb = pzb/2 
	pxb = .5*(1 + (pizm + pxym)^m) - pib
	pyb = .5*(1 - (pizm + pxym)^m) - pzb

	return pib, pxb, pzb, pyb 
end 




#errorCheck(["XIIII"], AllLogical, syndromeRepair, S)

#=
for i in 0:4^n-1
	error = decimal_to_bin(i,2n)
	degen, logical = errorCheck(error, AllLogical, syndromeRepair, S; stringMode = false)
	if logical == 1 
			println(Symplectic.build_from_bits(error'))
	end 
end
=#




