//! rust-kernel: fast inner loops for induced channel computation.
//!
//! C-ABI export computes the induced logical distribution with ML-centering
//! and the per-syndrome hashing bound, i.e.
//!   H_bar = sum_s p(s) * H( p(a',b'|s) ),
//! and returns (k - H_bar)/n.
//!
//! Inputs (bit-packed):
//!   - Ttab (2^r × W)  packed (U,V): all stabilizer-span error cosets over t ∈ F2^r
//!   - AL   (A   × W)  packed (U,V): logical-X span over a ∈ F2^k
//!   - BL   (A   × W)  packed (U,V): logical-Z span over b ∈ F2^k
//!   - SG   (2^r × W)  packed (U,V): pure-error block (syndrome reps) over s ∈ F2^r
//! Channel parameters: pI, pX, pZ, pY
//! Size params: n (physical qubits), r (checks), k (logical qubits), words = ceil(n/64)
//!
//! Output:
//!   - pbar_out (A×A) row-major: ML-centered marginal over (a',b') (Σ_s P_s shifted & normalized)
//!   - return value: hashing bound = (k - Σ_s p(s) H(p(a',b'|s)))/n
//!
//! Implementation notes:
//!   For each syndrome s, we compute P_s(a,b) = Σ_t P(E) where
//!       base = AL[a] ⊕ BL[b] ⊕ SG[s],   E = base ⊕ Ttab[t].
//!   We then find ML (a*,b*) for this s, define shifted coordinates
//!       a' = a ⊕ a*,  b' = b ⊕ b*,
//!   accumulate the unnormalized histogram over (a',b'), and simultaneously
//!   compute p(s) (= sum over a,b of P_s) and H(p(a',b'|s)).
use rayon::prelude::*;
use std::cmp::Ordering;
use std::slice;

#[inline(always)]
fn prob_from_counts(pI: f64, pX: f64, pZ: f64, pY: f64, ni: u32, nx: u32, nz: u32, ny: u32) -> f64 {
    pI.powi(ni as i32) * pX.powi(nx as i32) * pZ.powi(nz as i32) * pY.powi(ny as i32)
}

#[inline(always)]
fn shannon_entropy_base2(p: &[f64]) -> f64 {
    p.iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| -x * x.log2())
        .sum()
}

#[inline(always)]
fn popcnt_u64(x: u64) -> u32 { x.count_ones() }

#[inline(always)]
fn count_ni_nx_nz_ny(al_u: &[u64], al_v: &[u64],
                     bl_u: &[u64], bl_v: &[u64],
                     sg_u: &[u64], sg_v: &[u64],
                     tt_u: &[u64], tt_v: &[u64]) -> (u32,u32,u32,u32)
{
    // E = (AL ⊕ BL ⊕ SG) ⊕ Ttab   on the (u|v) halves
    let words = tt_u.len();
    let mut nx = 0u32;
    let mut nz = 0u32;
    let mut ny = 0u32;
    for w in 0..words {
        let u = al_u[w] ^ bl_u[w] ^ sg_u[w] ^ tt_u[w];
        let v = al_v[w] ^ bl_v[w] ^ sg_v[w] ^ tt_v[w];
        let xmask = u & !v;
        let zmask = !u & v;
        let ymask = u & v;
        nx += popcnt_u64(xmask);
        nz += popcnt_u64(zmask);
        ny += popcnt_u64(ymask);
    }
    // ni inferred from total sites n = 64*words at most; we don’t have n here,
    // so caller provides n and computes ni = n - (nx+nz+ny)
    (0, nx, nz, ny)
}

#[no_mangle]
pub extern "C" fn compute_pbar_and_hashing_bound(
    // Ttab packed: rows 0..(2^r - 1), words per row = words
    ttab_u_ptr: *const u64,
    ttab_v_ptr: *const u64,
    t_rows: usize,
    words: usize,

    // AL packed: A rows
    al_u_ptr: *const u64,
    al_v_ptr: *const u64,
    a_rows: usize,

    // BL packed: B rows (= A)
    bl_u_ptr: *const u64,
    bl_v_ptr: *const u64,
    b_rows: usize,

    // SG packed: S rows (= 2^r)
    sg_u_ptr: *const u64,
    sg_v_ptr: *const u64,
    s_rows: usize,

    // sizes / channel
    n_sites: u32,
    r_checks: u32,
    k_log: u32,
    pI: f64,
    pX: f64,
    pZ: f64,
    pY: f64,

    // outputs
    pbar_out_ptr: *mut f64, // length A*B  (A==B==2^k)
) -> f64 {
    let ttab_u = unsafe { slice::from_raw_parts(ttab_u_ptr, t_rows * words) };
    let ttab_v = unsafe { slice::from_raw_parts(ttab_v_ptr, t_rows * words) };
    let al_u = unsafe { slice::from_raw_parts(al_u_ptr, a_rows * words) };
    let al_v = unsafe { slice::from_raw_parts(al_v_ptr, a_rows * words) };
    let bl_u = unsafe { slice::from_raw_parts(bl_u_ptr, b_rows * words) };
    let bl_v = unsafe { slice::from_raw_parts(bl_v_ptr, b_rows * words) };
    let sg_u = unsafe { slice::from_raw_parts(sg_u_ptr, s_rows * words) };
    let sg_v = unsafe { slice::from_raw_parts(sg_v_ptr, s_rows * words) };

    let a_dim = a_rows; // A = 2^k
    let b_dim = b_rows; // B = 2^k (same)
    debug_assert_eq!(a_dim, b_dim);
    debug_assert_eq!(s_rows, (1usize << r_checks));

    // Accumulator for marginal ML-centered p̄(a',b') (unnormalized until the end).
    let mut pbar_accum = vec![0.0f64; a_dim * b_dim];

    // We also need Σ_s p(s)*H(p(a',b'|s)).
    // Compute this in parallel across s and reduce.
    let word_stride = words; // words per row in each packed table
    let t_rows_us = t_rows;
    let a_dim_us = a_dim;
    let b_dim_us = b_dim;
    let n_sites_f = n_sites as u32;

    // Per-s contributions (p_s, p_s * H_s) and we also gather the shifted table to add to pbar_accum.
    // We’ll do a two-pass scheme: parallel compute per-s vectors and ML coords,
    // then sequentially accumulate to pbar_accum to avoid atomics.
    #[derive(Clone)]
    struct SRes {
        ml_a: usize,
        ml_b: usize,
        p_sum: f64,
        p_weighted_entropy: f64,
        // shifted, unnormalized table over (a',b') for this s  (size A×B)
        shifted: Vec<f64>,
    }

    let per_s_results: Vec<SRes> = (0..s_rows).into_par_iter().map(|s_idx| {
        // views for SG row s
        let sg_u_row = &sg_u[s_idx * word_stride .. (s_idx + 1) * word_stride];
        let sg_v_row = &sg_v[s_idx * word_stride .. (s_idx + 1) * word_stride];

        // We will build P_s(a,b) into a temporary buffer of size A×B
        let mut ps = vec![0.0f64; a_dim_us * b_dim_us];

        // For each (a,b), accumulate over t
        for a in 0..a_dim_us {
            let al_u_row = &al_u[a * word_stride .. (a + 1) * word_stride];
            let al_v_row = &al_v[a * word_stride .. (a + 1) * word_stride];
            for b in 0..b_dim_us {
                let bl_u_row = &bl_u[b * word_stride .. (b + 1) * word_stride];
                let bl_v_row = &bl_v[b * word_stride .. (b + 1) * word_stride];

                let mut total = 0.0f64;
                for t in 0..t_rows_us {
                    let tt_u_row = &ttab_u[t * word_stride .. (t + 1) * word_stride];
                    let tt_v_row = &ttab_v[t * word_stride .. (t + 1) * word_stride];

                    // count types
                    let mut nx = 0u32;
                    let mut nz = 0u32;
                    let mut ny = 0u32;
                    for w in 0..word_stride {
                        let u = al_u_row[w] ^ bl_u_row[w] ^ sg_u_row[w] ^ tt_u_row[w];
                        let v = al_v_row[w] ^ bl_v_row[w] ^ sg_v_row[w] ^ tt_v_row[w];
                        let xmask = u & !v;
                        let zmask = !u & v;
                        let ymask = u & v;
                        nx += xmask.count_ones();
                        nz += zmask.count_ones();
                        ny += ymask.count_ones();
                    }
                    let ni = n_sites_f - (nx + nz + ny);
                    total += prob_from_counts(pI, pX, pZ, pY, ni, nx, nz, ny);
                }
                ps[a * b_dim_us + b] = total;
            }
        }

        // Probability of the syndrome p(s) = sum_{a,b} P_s(a,b)
        let p_s: f64 = ps.iter().copied().sum();

        // ML (a*,b*)
        let mut ml_a = 0usize;
        let mut ml_b = 0usize;
        if p_s > 0.0 {
            let mut best = 0.0f64;
            for a in 0..a_dim_us {
                for b in 0..b_dim_us {
                    let val = ps[a * b_dim_us + b];
                    if val > best {
                        best = val;
                        ml_a = a;
                        ml_b = b;
                    }
                }
            }
        }

        // Produce shifted table q_s(a',b') = P_s(a'⊕a*, b'⊕b*) / p_s
        let mut shifted = vec![0.0f64; a_dim_us * b_dim_us];
        let h_s: f64;
        if p_s > 0.0 {
            for a_prime in 0..a_dim_us {
                for b_prime in 0..b_dim_us {
                    let a = a_prime ^ ml_a;
                    let b = b_prime ^ ml_b;
                    shifted[a_prime * b_dim_us + b_prime] = ps[a * b_dim_us + b];
                }
            }
            // Normalize to conditional and compute H(q_s)
            let mut tmp = shifted.clone();
            for x in &mut tmp {
                *x /= p_s;
            }
            h_s = shannon_entropy_base2(&tmp);
        } else {
            h_s = 0.0;
        }

        SRes {
            ml_a,
            ml_b,
            p_sum: p_s,
            p_weighted_entropy: p_s * h_s,
            shifted, // still unnormalized; used to build marginal p̄ later
        }
    }).collect();

    // Accumulate p̄(a',b') = Σ_s shifted(a',b'), then normalize it.
    for sr in &per_s_results {
        for (i, v) in sr.shifted.iter().enumerate() {
            pbar_accum[i] += *v;
        }
    }
    let total_mass: f64 = pbar_accum.iter().copied().sum();
    let pbar_norm: Vec<f64> = if total_mass > 0.0 {
        pbar_accum.iter().map(|&x| x / total_mass).collect()
    } else {
        // degenerate channel; keep uniform to avoid NaNs
        vec![1.0 / (a_dim as f64 * b_dim as f64); a_dim * b_dim]
    };

    // New: H_bar = Σ_s p(s) H(p(a',b'|s))
    let h_bar: f64 = per_s_results.iter().map(|sr| sr.p_weighted_entropy).sum::<f64>();

    // Hashing bound with the *conditional* entropy
    let k = k_log as f64;
    let n_f = (n_sites as f64);
    let hashing_bound = (k - h_bar) / n_f;

    // Write p̄ to output
    let out = unsafe { slice::from_raw_parts_mut(pbar_out_ptr, pbar_norm.len()) };
    out.copy_from_slice(&pbar_norm);

    hashing_bound
}

