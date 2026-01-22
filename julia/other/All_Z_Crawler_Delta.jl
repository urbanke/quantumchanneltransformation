# ============================================
# Enumerate all binary (n-k) × (2n) matrices
# in the standard sub-block form, for all r
# ============================================
include("../src/Symplectic.jl")
include("../src/SGS.jl")
include("../env_utils/Channels.jl")

using .Symplectic, .SGS
using QECInduced, .Channels
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

function all_z_code_checkn(pz, customP) 
    S = falses(1,2n)
    S[1,(n+1):end] .= true 
    S = Matrix{Bool}(S)
    hb_induced = QECInduced.check_induced_channel(S, pz, customP;  sweep=false, threads = 0)
    return hb_induced
end 

# robust margin
const DEFAULT_DELTA = 1e-8

# compute g(p) = (hb - h) - delta
function diff_marginn(p, customP, delta)
    hb = all_z_code_checkn(p, customP)
    h  = hashing_calc(p, customP)
    d  = (hb - h) - delta
    return isfinite(d) ? d : NaN
end

function diff_margin_meann(p, customP, delta; reps::Int=3)
    s = 0.0
    cnt = 0
    for _ in 1:reps
        d = diff_marginn(p, customP, delta)
        if isfinite(d)
            s += d
            cnt += 1
        end
    end
    return cnt == 0 ? NaN : s / cnt
end

function estimate_noise_marginn(p, customP, delta; reps::Int=7)
    vals = Vector{Float64}(undef, reps)
    ok = 0
    for i in 1:reps
        v = diff_marginn(p, customP, delta)
        vals[i] = v
        ok += isfinite(v) ? 1 : 0
    end
    if ok == 0
        return (Inf, NaN)
    end
    finite_vals = filter(isfinite, vals)
    return (maximum(finite_vals) - minimum(finite_vals)), sum(finite_vals) / length(finite_vals)
end

# helper: estimate noise (peak-to-peak) and mean at a point
function estimate_noisen(p, customP; reps::Int=7)
    vals = Vector{Float64}(undef, reps)
    for i in 1:reps
        vals[i] = diff_fn(p, customP)
    end
    return (maximum(vals) - minimum(vals)), sum(vals) / reps
end

# bracket a sign change around p0 by scanning outward
function bracket_rootn, p0, customP;
                      step::Float64=1e-3, max_steps::Int=2000, reps::Int=3)

    f0 = diff_meann(p0, customP; reps=reps)

    # Scan upward from p0
    p_prev, f_prev = p0, f0
    for k in 1:max_steps
        p = p0 + k*step
        f = diff_meann(p, customP; reps=reps)
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
        f = diff_meann(p, customP; reps=reps)
        if sign(f) != sign(f_prev)
            return (p, p_prev)  # bracket [p, p_prev]
        end
        p_prev, f_prev = p, f
    end

    error("Could not bracket a sign change for n=$n starting at p0=$p0")
end

function bracket_root_local_marginn, p0, customP, delta;
                                   step::Float64=1e-3,
                                   max_steps::Int=800,
                                   reps::Int=3,
                                   p_min::Float64=0.0,
                                   p_max::Float64=0.35)

    f0 = diff_margin_meann(p0, customP, delta; reps=reps)
    if !isfinite(f0)
        error("g(p0) is not finite for n=$n at p0=$p0")
    end

    pL, fL = p0, f0
    pR, fR = p0, f0

    for k in 1:max_steps
        # Right
        pR_new = p0 + k*step
        if pR_new <= p_max
            fR_new = diff_margin_meann(pR_new, customP, delta; reps=reps)
            if isfinite(fR_new) && sign(fR_new) != sign(fR)
                return (pR, pR_new)
            end
            if isfinite(fR_new)
                pR, fR = pR_new, fR_new
            end
        end

        # Left
        pL_new = p0 - k*step
        if pL_new >= p_min
            fL_new = diff_margin_meann(pL_new, customP, delta; reps=reps)
            if isfinite(fL_new) && sign(fL_new) != sign(fL)
                return (pL_new, pL)
            end
            if isfinite(fL_new)
                pL, fL = pL_new, fL_new
            end
        end
    end

    error("Could not bracket a sign change for margin delta=$delta near p0=$p0 for n=$n within [$p_min,$p_max]")
end
function bisect_threshold_marginn, p_lo, p_hi, customP, delta;
                                 p_tol::Float64=1e-6, reps::Int=7)

    noise_lo, f_lo = estimate_noise_marginn(p_lo, customP, delta; reps=reps)
    noise_hi, f_hi = estimate_noise_marginn(p_hi, customP, delta; reps=reps)

    # swap if orientation flipped
    if f_lo > 0 && f_hi < 0
        p_lo, p_hi = p_hi, p_lo
        f_lo, f_hi = f_hi, f_lo
        noise_lo, noise_hi = noise_hi, noise_lo
    end

    if !(f_lo < 0 && f_hi > 0)
        error("Margin bracket does not straddle for n=$n: g(lo)=$f_lo, g(hi)=$f_hi (delta=$delta)")
    end

    while (p_hi - p_lo) > p_tol
        p_mid = (p_lo + p_hi)/2
        noise_mid, f_mid = estimate_noise_marginn(p_mid, customP, delta; reps=reps)

        # If we're noise-limited, we still return p_hi (conservative smallest p with g>=0)
        stop_eps = max(1e-12, 0.5*noise_mid)
        if abs(f_mid) <= stop_eps
            return p_hi
        end

        if f_mid > 0
            p_hi = p_mid
        else
            p_lo = p_mid
        end
    end

    return p_hi
end


function safe_diffn(p, customP)
    hb = all_z_code_checkn(p, customP)
    h  = hashing_calc(p, customP)
    d  = hb - h
    if !isfinite(d)
        return NaN
    end
    return d
end

function safe_diff_meann(p, customP; reps::Int=3)
    s = 0.0
    cnt = 0
    for _ in 1:reps
        d = safe_diffn(p, customP)
        if isfinite(d)
            s += d
            cnt += 1
        end
    end
    return cnt == 0 ? NaN : s / cnt
end


function root_finder_margin(n_max, customP;
                                pz=0.23,
                                delta=DEFAULT_DELTA,
                                scan_step=1e-3, scan_reps=3,
                                bisect_reps=2, p_tol=1e-6,
                                p_min=0.0, p_max=0.35)

    smallest_p = Dict{Int, Float64}()

    for n in 2:n_max
        println("\n==== n = $n (margin >= $delta) ====")
        println("Starting scan at p0 = $pz")

        elapsed_total = @elapsed begin
            p_lo, p_hi = bracket_root_local_marginn, pz, customP, delta;
                                                   step=scan_step, max_steps=800,
                                                   reps=scan_reps, p_min=p_min, p_max=p_max)

            p_star = bisect_threshold_marginn, p_lo, p_hi, customP, delta;
                                             p_tol=p_tol, reps=bisect_reps)

            smallest_p[n] = p_star
            println("Bracket: [$p_lo, $p_hi]")
            println("Threshold (margin): $p_star")

        end

        println("Elapsed total for n=$n: $elapsed_total seconds")
    end

    return [smallest_p[n] for n in 2:n_max]
end



function main()
    n_max = 16
    hashing, base_grid = root_finder_margin(n_max, Channels.ninexz)
end

# Run the main function

main()
