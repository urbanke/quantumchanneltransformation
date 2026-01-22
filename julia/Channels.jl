module Channels
export ninexz, depolar, twoPauli, indy 


"""  
functions must be of this form 

function ChannelName(x; tuple = false, plot = false) 
	pI, pX, pZ, pY = f(x) 
    if tuple 
        return (pI, pX, pZ, pY)
    end
    if plot # this is what you want plotted (most likely 1 - pI)
        return 1-pI 
    end 
    return [pI, pX, pZ, pY]
"""


function ninexz(x; tuple = false, plot = false) # this is an example of customP, which gives the same one smith did 
    z = x/9
    pI = (1-z)*(1-x)  
    pX = x*(1-z) 
    pZ = z*(1-x)
    pY = z*x
    if tuple # this should always be here, do not touch 
        return (pI, pX, pZ, pY)
    end
    if plot # this is to plot different things (for example, smith plots 1-pI instead of pX despite working with pX)
        return 1-pI 
    end 
    return [pI, pX, pZ, pY]
end 


function depolar(p; tuple = false, plot = false) # this is an example of customP, which gives the same one smith did 
    pI = (1-p)
    pX = p/3
    pZ = p/3
    pY = p/3
    if tuple # this should always be here, do not touch 
        return (pI, pX, pZ, pY)
    end
    if plot # this is to plot different things (for example, smith plots 1-pI instead of pX despite working with pX)
        return p 
    end 
    return [pI, pX, pZ, pY]
end 

function twoPauli(p; tuple = false, plot = false) # this is an example of customP, which gives the same one smith did 
    pI = (1-p)
    pX = p/2
    pZ = p/2
    pY = 0.0 
    if tuple # this should always be here, do not touch 
        return (pI, pX, pZ, pY)
    end
    if plot # this is to plot different things (for example, smith plots 1-pI instead of pX despite working with pX)
        return p
    end 
    return [pI, pX, pZ, pY]
end 

function indy(x; tuple = false, plot = false)
    z = x 
    pI = (1-z)*(1-x) 
    pX = x*(1-z) 
    pZ = z*(1-x)
    pY = z*x
    if tuple # this should always be here, do not touch 
        return (pI, pX, pZ, pY)
    end
    if plot # this is to plot different things (for example, smith plots 1-pI instead of pX despite working with pX)
        return 1-pI 
    end 
    return [pI, pX, pZ, pY]
end 

end # module
