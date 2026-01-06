module ParallelSweep

using Base.Threads: @threads, nthreads
using ..Induced

export sweep_depolarizing_grid


function H1(p::AbstractVector{<:Real})
    s = 0.0
    @inbounds for v in p
        if v > 0
            s -= v * log2(v)
        end
    end
    return s
end


"""
    sweep_depolarizing_grid(H, Lx, Lz, G; p_min=0.0, p_max=1.0, step=0.01, threads=nthreads())

Parallel sweep over depolarizing parameter p in [p_min, p_max] with step size `step`.
Each worker calls the Rust-accelerated induced channel computation. Returns a matrix
with three columns: [p  hashing_bound_induced 1-H(pd)].
"""
function sweep_depolarizing_grid(H, Lx, Lz, G; p_min=0.0, p_max=1.0, step=0.01, threads::Int=nthreads())
    ps = collect(range(p_min, p_max; step=step))
    m = length(ps)
    out = zeros(Float64, m, 3)
    @threads for i in 1:m
    p = ps[i]
	pd = [1-p, p/3,p/3,p/3]
	c = 1-H1(pd)
    #=pbar,=# hb = Induced.induced_channel_and_hashing_bound(H, Lx, Lz, G, (1-p, p/3, p/3, p/3))
    out[i,1] = p
    out[i,2] = hb
	out[i,3] = c
    end
    out
end

function sweep_independent_grid(H, Lx, Lz, G; p_min=0.0, p_max=1.0, step=0.01, threads::Int=nthreads())
    ps = collect(range(p_min, p_max; step=step))
    m = length(ps)
    out = zeros(Float64, m, 3)
    @threads for i in 1:m
    p = ps[i]
        pd = [(1-p)*(1-p), (1-p)*p,p*(1-p),p*p]
        c = 1-H1(pd)
    #=pbar,=# hb = Induced.induced_channel_and_hashing_bound(H, Lx, Lz, G, ((1-p)*(1-p), (1-p)*p, p*(1-p), p*p))
    out[i,1] = p
    out[i,2] = hb
        out[i,3] = c
    end
    out
end



function sweep_custom_grid(H, Lx, Lz, G, customP; p_min=0.0, p_max=1.0, step=0.01, threads::Int=nthreads())
    ps = collect(range(p_min, p_max; step=step))
    m = length(ps)
    out = zeros(Float64, m, 3)
    @threads for i in 1:m
        p = ps[i]
        pd = customP(p)#[customP[1], customP[2], customP[3], customP[4]]
        c = 1-H1(pd)
    #=pbar,=# hb = Induced.induced_channel_and_hashing_bound(H, Lx, Lz, G, customP(p; tuple=true))
    out[i,1] = p
    out[i,2] = hb
        out[i,3] = c
    end
    out
end


function sweep_custom_grid_exact(H, Lx, Lz, G, ps, customP; depolarizing=false, independent=false, threads::Int=nthreads())
    ps_vec = collect(ps)  # Ensure it's a vector
    m = length(ps_vec)
    out = zeros(Float64, m, 3)
    
    @threads for i in 1:m
        p = ps_vec[i]
        
        # Determine the channel probabilities
        if depolarizing
            pd = [1-p, p/3, p/3, p/3]
            pd_tuple = (1-p, p/3, p/3, p/3)
        elseif independent
            pd = [(1-p)*(1-p), p*(1-p), p*(1-p), p*p]
            pd_tuple = ((1-p)*(1-p), p*(1-p), p*(1-p), p*p)
        else
            pd = customP(p)
            pd_tuple = customP(p; tuple=true)
        end
        
        c = 1 - H1(pd)
        hb = Induced.induced_channel_and_hashing_bound(H, Lx, Lz, G, pd_tuple)
        
        out[i,1] = p
        out[i,2] = hb
        out[i,3] = c
    end
    
    return out
end


function sweep_hashing_grid(ps, ChannelType; customP = nothing)
    ps_vec = collect(ps)                 # Convert to vector if needed
    m = length(ps_vec)                   # Number of points
    out = zeros(Float64, m)              # Change to 1D vector instead of (m, 1)
    
    for i in 1:m
        p = ps_vec[i]
        if ChannelType == "Depolarizing"
            pd = [1 - p, p/3, p/3, p/3]
        elseif ChannelType == "Independent"
            pd = [(1 - p) * (1 - p), p * (1 - p), p * (1 - p), p * p]
        else
            pd = customP(p)
        end
        c = 1 - H1(pd)
        out[i] = c                       # Assign directly to vector element
    end
    
    return out                            # Return 1D vector
end



end # module

