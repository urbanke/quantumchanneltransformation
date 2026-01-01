include("src/Symplectic.jl")
using .Symplectic


#using QECInduced


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



Stabilizers = ["ZIIXX", "IZIIX", "IIZXI"]
#Stabilizers = ["ZXI", "IXX"]
logicalX = ["IIIXI","IIIIX"]
logicalZ = ["XIXZI","XXIIZ"]
G = ["XIIII", "IXIII", "IIXII"]

#logicalX = ["IIX"]
#logicalZ = ["XYY"]

#logicalX = Bool[0,0,1,0,0,0] # IIX
#logicalZ = Bool[1,0,0,0,1,1] # XYY

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

#Any["IZXXX", "ZIXIX", "IIXXI"]


pchannel = [(1-p)*(1-p), p*(1-p), p*(1-p), p*p]

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
		for i in 1:lilK
			logicals[i] = Int(Symplectic.symp_inner(error, AllLogical[i,:])) # for every error get the logical 
		end
		dictNum = bin_to_dec(logicals)
		prob = calculateP(error) 

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
		dictNum = bin_to_dec(residuals)
		# calculate num of errors 
		prob = calculateP(error) 
		inducedDict[dictNum] += prob
	end
	h = 0 
	for i in 0:k-1
		inducedDict[i] = inducedDict[i]/PS
		h -= inducedDict[i]*log2(inducedDict[i])
	end
	#println(inducedDict)
	println("Syrdome: (",repairs,"), (k - H(p(a,b|s))/n = ",(div(lilK,2)-h)/n)
 	return h*PS
end

global H = 0 
for i in 0:r-1
	println("="^50,i,"="^50)
	println("Syndrome Measured: ", i)
	println("Logical Operator Applied: ", syndromeRepair[i])
	idxPos = trueRepairs[i]
	trueErrorRep = bin_to_dec(idxPos) 
	println("True Error Operator Applied: ", trueErrorRep)
	global H += inducedChannel(AllRecieved[i],SyndromeDict[i], pchannel,AllLogical, syndromeRepair[i])
	println()
end
println((div(lilK,2)-H)/n)


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
	println(synds)
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




