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

Stabilizers = ["ZXI", "IXX"]
S = Symplectic.build_from_stabs(Stabilizers)

function decimal_to_bin(number, width)
    out = Vector{Bool}(undef, width)
    for i in 1:width
        out[i] = ((number >> (width - i)) & 1) == 1
    end
    return out
end

# The tableau shows us the two logicals are IIX and XZY 

# Code is [Z, X , I] [I, X, X]

p = 0.11002786443835955

#Any["IZXXX", "ZIXIX", "IIXXI"]


pchannel = [(1-p)*(1-p), p*(1-p), p*(1-p), p*p]

# calculate the probability of the syndromes 

SyndromeDict = Dict(0 => 0.0, 1 => 0.0, 2 => 0.0, 3 => 0.0) # making a dict where the 4 syndromes currently have no prob 
AllRecieved = Dict(0 => [], 1 => [], 2 => [], 3 => [])


for i in 0:(2^6 - 1) # there are 3 quaterary symbols so i will say there are 2^6 options 
	noise = decimal_to_bin(i, 6)
	synd1 = Int(Symplectic.symp_inner(noise, S[1,:]))
	synd2 = Int(Symplectic.symp_inner(noise, S[2,:]))

	# calculate num of errors 
	I = 0; X = 0; Z = 0; Y = 0
	for j in 1:3 
		if !noise[j] & !noise[j+3] # 0,0 means I 
			I += 1
		elseif !noise[j+3] # 1,0 means X 
			X += 1
		elseif !noise[j] # 0,1 means Z 
			Z += 1
		else 
			Y += 1 
		end 
	end
	SyndromeDict[synd1 + 2*synd2] += pchannel[1]^I * pchannel[2]^X * pchannel[3]^Z * pchannel[4]^Y # this is the probability of the syndrome 
	push!(AllRecieved[synd1 + 2*synd2], noise) # this gives all vectors relating to a certain syndrome 
end 


function errorBasedOnSynd(recieved,PS, pchannel,logicalX, logicalZ)

	logicalprob = 0 

	logicalDict = Dict(0 => 0.0, 1 => 0.0, 2 => 0.0, 3 => 0.0)

	for error in recieved 
		logicalXest = Int(Symplectic.symp_inner(error, logicalX)) # for every error get the logical X 
		logicalZest = Int(Symplectic.symp_inner(error, logicalZ)) # or logical Z (if there are any) 
		# calculate num of errors 
		I = 0; X = 0; Z = 0; Y = 0 
		for j in 1:3 
			if !error[j] & !error[j+3] # 0,0 means I 
				I += 1
			elseif !error[j+3] # 1,0 means X 
				X += 1
			elseif !error[j] # 0,1 means Z 
				Z += 1
			else 
				Y += 1 
			end 
		end
		logicalDict[logicalXest + 2*logicalZest] += pchannel[1]^I * pchannel[2]^X * pchannel[3]^Z * pchannel[4]^Y # the total probability of each logical
	end
	maxVal = 0 
	maxKey = 0
	for i in 0:3
		logicalDict[i] = logicalDict[i]/PS
		if logicalDict[i] >= maxKey
			maxKey = logicalDict[i]
			maxVal = i 
		end
	end
	println(logicalDict)
 	return maxVal # return the logical correction that has the most probability  a(s), b(s)
end

logicalX = Bool[0,0,1,0,0,0] # IIX
logicalZ = Bool[1,0,0,0,1,1] # XYY
syndromeRepair = [0,0,0,0]

for i in 0:3 
	println("="^50,i,"="^50)
	syndromeRepair[i+1] = errorBasedOnSynd(AllRecieved[i],SyndromeDict[i],pchannel,logicalX, logicalZ)
	println()
end

function inducedChannel(recieved,PS, pchannel,logicalX, logicalZ, repair)
	logicalprob = 0 
	if repair == 0 
		xrepair = 0; zrepair = 0
	elseif repair == 1 
		xrepair = 1; zrepair = 0 
	elseif repair == 2
		xrepair = 0; zrepair = 1; 
	else 
		xrepair = 1; zrepair = 1;
	end 
	inducedDict = Dict(0 => 0.0, 1 => 0.0, 2 => 0.0, 3 => 0.0)

	for error in recieved 
		residualX = Int(Symplectic.symp_inner(error, logicalX)) ⊻ xrepair # the residual when we apply the ML correction 
		residualZ = Int(Symplectic.symp_inner(error, logicalZ)) ⊻ zrepair # a',b' = a ⊕ a(s), b ⊕ b(s) 
		# calculate num of errors 
		I = 0; X = 0; Z = 0; Y = 0
		for j in 1:3 
			if !error[j] & !error[j+3] # 0,0 means I 
				I += 1
			elseif !error[j+3] # 1,0 means X 
				X += 1
			elseif !error[j] # 0,1 means Z 
				Z += 1
			else 
				Y += 1 
			end 
		end
		inducedDict[residualX + 2*residualZ] += pchannel[1]^I * pchannel[2]^X * pchannel[3]^Z * pchannel[4]^Y
	end
	h = 0 
	for i in 0:3
		inducedDict[i] = inducedDict[i]/PS
		h -= inducedDict[i]*log2(inducedDict[i])
	end
	println(inducedDict)
	println("Syrdome: (",xrepair,",",zrepair,"), (1 - H(p(a,b|s))/3 = ",(1-h)/3)
 	return h*PS
end

global H = 0 
for i in 0:3 
	println("="^50,i,"="^50)
	global H += inducedChannel(AllRecieved[i],SyndromeDict[i], pchannel,logicalX, logicalZ, syndromeRepair[i+1])
	println()
end
println((1-H)/3)








