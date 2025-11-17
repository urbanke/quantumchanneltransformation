

H = Bool[0 1 1 0 0 0; 0 0 1 1 0 0]
Lx = Bool[0 0 1 0 0 0]
Lz = Bool[1 0 0 0 1 1]
G = Bool[0 0 0 0 1 0; 1 0 0 0 0 0]


p = 0.11002786443835955
global pchannel = [(1-p)*(1-p), p*(1-p), p*(1-p), p*p]


function calculateP(noise) # this function works (no need to look at)
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
	return pchannel[1]^I * pchannel[2]^X * pchannel[3]^Z * pchannel[4]^Y # this is the probability of the syndrome 
end 


# H(t|s)
zerob = Bool[0 0 0 0 0 0]
global T = vcat(zerob, vcat(H, (H[1,:] .⊻ H[2,:])')) # this means we can easily iterate over all combinations of T, A, B, S
global A = vcat(zerob, Lx)
global B = vcat(zerob, Lz)
global S = vcat(zerob, vcat(G, (G[1,:] .⊻ G[2,:])'))


function calculatePS() # get the probability of the syndrome 
	pdict = Dict(1 => 0.0, 2 => 0.0, 3 => 0.0, 4 => 0.0)
	for s in 1:4
		for t in 1:4
			for a in 1:2 
				for b in 1:2 
					noise = vec(T[t,:]) .⊻ vec(A[a,:]) .⊻ vec(B[b,:]) .⊻ vec(S[s,:])
					pdict[s] += calculateP(noise) # p(t,a,b,s) 
				end
				# p(t,a,s)
			end
			# p(t,s)
		end
		#p(s) 
	end
	return pdict
end 

global ps = calculatePS() # P(s) 

function calculateHT_givenSpecificS(s) 
	pdict = Dict(1 => 0.0, 2 => 0.0, 3 => 0.0, 4 => 0.0)
	for t in 1:4 # go over all t 
		for a in 1:2 # go over all a 
			for b in 1:2 # go over all b 
				noise = vec(T[t,:]) .⊻ vec(A[a,:]) .⊻ vec(B[b,:]) .⊻ vec(S[s,:]) # create E 
				pdict[t] += calculateP(noise) # get the probability of E, p(t,a,b,s = s) 
			end
		end
		# sum_a,b p(t,a,b,s) = p(t,s = s) 
	end
	# pdict now has the probability of P(t,s = s)
	# p(t|s = s) = p(t,s = s)/p(s)  
	hts = 0 
	for i in 1:4
		pdict[i] = pdict[i]/ps[i]
		hts -= pdict[i]*log2(pdict[i]) 
	end
	return hts # H(t|S=s) 
end 

function calculateHT_givenS()
	h = 0
	for i in 1:4 
		h += ps[i]*calculateHT_givenSpecificS(i) # H(t|s) = sum_s P(s)*H(t|S=s)
	end 
	return h
end 



function calculateHS() 
	pdict = Dict(1 => 0.0, 2 => 0.0, 3 => 0.0, 4 => 0.0)
	for s in 1:4
		for t in 1:4 
			for a in 1:2 
				for b in 1:2 
					noise = vec(T[t,:]) .⊻ vec(A[a,:]) .⊻ vec(B[b,:]) .⊻ vec(S[s,:])
					pdict[s] += calculateP(noise) # p(t,a,b,s) 
				end 
				# p(t,a,s)
			end
			# p (t,s)
		end
		#p(s) 
	end
	hs = 0 
	for i in 1:4
		hs -= pdict[i]*log2(pdict[i])
	end
	return hs
end 

println("Entropy of〚3,1〛Code found from DFS")
println("H(s) = ", calculateHS())
println("H(t|s) = ", calculateHT_givenS())
println("H(t,s) = ", calculateHS() + calculateHT_givenS())


