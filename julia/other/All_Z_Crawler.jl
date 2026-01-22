# ============================================
# Enumerate all binary (n-k) × (2n) matrices
# in the standard sub-block form, for all r
# ============================================
include("../src/Symplectic.jl")
include("../src/SGS.jl")
include("../env_utils/Channels.jl")

using .Symplectic, .SGS, .Channels
using QECInduced
using Base.Threads
using Plots
using LinearAlgebra
using Random
using Printf 


function hashing_calc(p, pfunc) 
    channel = pfunc(p) 
    h = 0 
    for i in 1:4 
        h -= channel[i]*log2(channel[i])
    end 
    return 1 - h 
end 

function all_z_code_check(n, pz, channelParamFunc) 
    S = falses(1,2n)
    S[1,(n+1):end] .= true 
    S = Matrix{Bool}(S)
    hb_ind = QECInduced.check_induced_channel(S, pz, channelParamFunc; sweep=false, threads = threads)
    return hb_ind
end 

# helper: compute hb_induced - hashing
function diff_f(n, p, channelParamFunc)
    hb = all_z_code_check(n, p, channelParamFunc)
    h  = hashing_calc(p, channelParamFunc)
    return hb - h
end

# helper: average a few evaluations to reduce sign jitter
function diff_mean(n, p, channelParamFunc; reps::Int=3)
    s = 0.0
    for _ in 1:reps
        s += diff_f(n, p, channelParamFunc)
    end
    return s / reps
end

# helper: estimate noise (peak-to-peak) and mean at a point
function estimate_noise(n, p, channelParamFunc; reps::Int=7)
    vals = Vector{Float64}(undef, reps)
    for i in 1:reps
        vals[i] = diff_f( n, p, channelParamFunc)
    end
    return (maximum(vals) - minimum(vals)), sum(vals) / reps
end

# bracket a sign change around p0 by scanning outward
function bracket_root(n, p0, channelParamFunc;
                      step::Float64=1e-3, max_steps::Int=2000, reps::Int=3)

    f0 = diff_mean(n, p0, channelParamFunc; reps=reps)

    # Scan upward from p0
    p_prev, f_prev = p0, f0
    for k in 1:max_steps
        p = p0 + k*step
        f = diff_mean(n, p, channelParamFunc; reps=reps)
        if sign(f) != sign(f_prev)
            return (p_prev, p)  # bracket [p_prev, p]
        end
        p_prev, f_prev = p, f
    end

    # If not found upward, scan downward (ensure p stays >= 0)
    p_prev, f_prev = p0, f0
    for k in 1:max_steps
        p = p0 - k*step
        if p < 0
            break
        end
        f = diff_mean(n, p, channelParamFunc; reps=reps)
        if sign(f) != sign(f_prev)
            return (p, p_prev)  # bracket [p, p_prev]
        end
        p_prev, f_prev = p, f
    end

    error("Could not bracket a sign change for n=$n starting at p0=$p0")
end

function bracket_root_local(n, p0, channelParamFunc;
                            step::Float64=1e-3,
                            max_steps::Int=500,
                            reps::Int=3,
                            p_min::Float64=0.0,
                            p_max::Float64=0.5)

    f0 = safe_diff_mean(n, p0, channelParamFunc; reps=reps)
    if !isfinite(f0)
        error("f(p0) is not finite for n=$n at p0=$p0")
    end

    # Search outward symmetrically and return the *closest* valid sign change
    pL = p0
    fL = f0
    pR = p0
    fR = f0

    for k in 1:max_steps
        # Right step
        pR_new = p0 + k*step
        if pR_new <= p_max
            fR_new = safe_diff_mean(n, pR_new, channelParamFunc; reps=reps)
            if isfinite(fR_new) && sign(fR_new) != sign(fR)
                return (pR, pR_new)
            end
            if isfinite(fR_new)
                pR, fR = pR_new, fR_new
            end
        end

        # Left step
        pL_new = p0 - k*step
        if pL_new >= p_min
            fL_new = safe_diff_mean(n, pL_new, channelParamFunc; reps=reps)
            if isfinite(fL_new) && sign(fL_new) != sign(fL)
                return (pL_new, pL)
            end
            if isfinite(fL_new)
                pL, fL = pL_new, fL_new
            end
        end
    end

    error("Could not bracket near p0=$p0 for n=$n within [$p_min,$p_max]")
end


# bisection inside a bracket; returns smallest p with diff >= 0
function bisect_threshold(n, p_lo, p_hi, channelParamFunc;
                          p_tol::Float64=1e-6, reps::Int=7)

    # We will enforce invariant: f(lo) < 0, f(hi) > 0
    noise_lo, f_lo = estimate_noise(n, p_lo, channelParamFunc; reps=reps)
    noise_hi, f_hi = estimate_noise(n, p_hi, channelParamFunc; reps=reps)

    # If bracket orientation is flipped, swap
    if f_lo > 0 && f_hi < 0
        p_lo, p_hi = p_hi, p_lo
        f_lo, f_hi = f_hi, f_lo
        noise_lo, noise_hi = noise_hi, noise_lo
    end

    if !(f_lo < 0 && f_hi > 0)
        error("Bracket does not straddle crossing for n=$n: f(lo)=$f_lo, f(hi)=$f_hi")
    end

    while (p_hi - p_lo) > p_tol
        p_mid = (p_lo + p_hi)/2

        noise_mid, f_mid = estimate_noise(n, p_mid, channelParamFunc; reps=reps)

        # Because hb_induced and hashing are ~0.5, use a *relative* noise stop too:
        # stop if mean is smaller than half the measured jitter (or very tiny floor)
        stop_eps = max(1e-12, 0.5*noise_mid)

        if abs(f_mid) <= stop_eps
            # We're at the noise floor; best "smallest p with diff >= 0" is hi
            return p_hi
        end

        if f_mid > 0
            p_hi, f_hi = p_mid, f_mid
        else
            p_lo, f_lo = p_mid, f_mid
        end
    end

    return p_hi  # smallest p in bracket with diff >= 0
end

function safe_diff(n, p, channelParamFunc)
    hb = all_z_code_check(n, p, channelParamFunc)
    h  = hashing_calc(p, channelParamFunc)
    d  = hb - h
    if !isfinite(d)
        return NaN
    end
    return d
end

function safe_diff_mean(n, p, channelParamFunc; reps::Int=3)
    s = 0.0
    cnt = 0
    for _ in 1:reps
        d = safe_diff(n, p, channelParamFunc)
        if isfinite(d)
            s += d
            cnt += 1
        end
    end
    return cnt == 0 ? NaN : s / cnt
end


function root_finder(n_max, channelParamFunc; pz=0.23,
                         scan_step=1e-3, scan_reps=1,
                         bisect_reps=1, p_tol=1e-6,
                         p_min=0.0, p_max=0.35)  # <- tighten this!

    smallest_p = Dict{Int, Float64}()

    for n in 2:n_max
        println("\n==== n = $n ====")
        println("Starting scan at p0 = $pz")

        elapsed_total = @elapsed begin
            p_lo, p_hi = bracket_root_local(n, pz, channelParamFunc;
                                            step=scan_step, max_steps=800,
                                            reps=scan_reps,
                                            p_min=p_min, p_max=p_max)

            p_star = bisect_threshold(n, p_lo, p_hi, channelParamFunc;
                                      p_tol=p_tol, reps=bisect_reps)

            smallest_p[n] = p_star

            println("Bracket: [$p_lo, $p_hi]")
            println("Threshold: $p_star")
        end

        println("Elapsed total for n=$n: $elapsed_total seconds")
    end

    return [smallest_p[n] for n in 2:n_max]
end




function main()
    n_max = 16
    hashing, base_grid = root_finder(n_max, Channels.ninexz)
end

# Run the main function

main()
