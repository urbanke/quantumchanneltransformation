module ParallelSweep

using Base.Threads: @threads, nthreads
using ..Induced

export sweep_depolarizing_grid

"""
    sweep_depolarizing_grid(H, Lx, Lz, G; p_min=0.0, p_max=1.0, step=0.01, threads=nthreads())

Parallel sweep over depolarizing parameter p in [p_min, p_max] with step size `step`.
Each worker calls the Rust-accelerated induced channel computation. Returns a matrix
with two columns: [p  hashing_bound].
"""
function sweep_depolarizing_grid(H, Lx, Lz, G; p_min=0.0, p_max=1.0, step=0.01, threads::Int=nthreads())
    ps = collect(range(p_min, p_max; step=step))
    m = length(ps)
    out = zeros(Float64, m, 2)
    @threads for i in 1:m
        p = ps[i]
        pbar, hb = Induced.induced_channel_and_hashing_bound(H, Lx, Lz, G, (1-p, p/3, p/3, p/3))
        out[i,1] = p
        out[i,2] = hb
    end
    out
end

end # module

