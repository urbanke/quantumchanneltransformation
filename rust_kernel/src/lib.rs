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
fn prob_from_counts_log(
    li: f64, lx: f64, lz: f64, ly: f64,
    ni: u32, nx: u32, nz: u32, ny: u32
) -> f64 {
    // Same real-number value as pI^ni * pX^nx * pZ^nz * pY^ny, but faster.
    // Not bit-for-bit identical to powi due to floating rounding.
    //
    // Avoid 0 * (-inf) => NaN when some p*=0 but the corresponding count is 0.
    let mut s = 0.0f64;
    if ni != 0 { s += (ni as f64) * li; }
    if nx != 0 { s += (nx as f64) * lx; }
    if nz != 0 { s += (nz as f64) * lz; }
    if ny != 0 { s += (ny as f64) * ly; }
    s.exp()
}

const INV_LN_2: f64 = 1.0 / std::f64::consts::LN_2;
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
    //pbar_out_ptr: *mut f64, // length A*B  (A==B==2^k)
) -> f64 {
    let li = pI.ln();
    let lx = pX.ln();
    let lz = pZ.ln();
    let ly = pY.ln();

    // Precompute probability lookup table for all (nx,nz,ny) with 0<=nx,nz,ny<=n.
    // This moves ln/exp work out of the innermost loops.
    // Indexing: idx = nx*(n+1)^2 + nz*(n+1) + ny.
    let n_us = n_sites as usize;
    let stride1 = n_us + 1;
    let stride2 = stride1 * stride1;
    let mut prob_lut = vec![0.0f64; stride1 * stride1 * stride1];
    for nx in 0..=n_us {
        for nz in 0..=n_us {
            for ny in 0..=n_us {
                let sum = nx + nz + ny;
                let p = if sum > n_us {
                    0.0
                } else {
                    let ni = n_us - sum;
                    // Same as prob_from_counts_log(li,lx,lz,ly,ni,nx,nz,ny)
                    let mut s = 0.0f64;
                    if ni != 0 { s += (ni as f64) * li; }
                    if nx != 0 { s += (nx as f64) * lx; }
                    if nz != 0 { s += (nz as f64) * lz; }
                    if ny != 0 { s += (ny as f64) * ly; }
                    s.exp()
                };
                prob_lut[nx * stride2 + nz * stride1 + ny] = p;
            }
        }
    }
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

    // We need H_bar = Σ_s p(s) * H(p(a,b|s)).
    //
    // Parallelization strategy (adaptive):
    //   - If s_rows is large enough, parallelize across syndromes s.
    //   - If s_rows is small (e.g. small r) but k is large (huge A,B), we instead
    //     parallelize across (s,a) work units and reduce per-s accumulators.
    //
    // Rationale: avoid "not enough tasks" when 2^r is small, while also avoiding
    // heavy nested parallelism.
    let word_stride = words; // words per row in each packed table
    let t_rows_us = t_rows;
    let a_dim_us = a_dim;
    let b_dim_us = b_dim;
    let n_sites_f = n_sites as u32;

    // Per-s contributions (p_s, p_s * H_s).
    #[derive(Clone)]
    struct SRes {
     //   ml_a: usize,
     //   ml_b: usize,
        p_sum: f64,
        p_weighted_entropy: f64,
        // shifted, unnormalized table over (a',b') for this s  (size A×B)
     //   shifted: Vec<f64>,
    }

    #[inline(always)]
    fn compute_syndrome_contrib(
        s_idx: usize,
        sg_u: &[u64],
        sg_v: &[u64],
        al_u: &[u64],
        al_v: &[u64],
        bl_u: &[u64],
        bl_v: &[u64],
        ttab_u: &[u64],
        ttab_v: &[u64],
        word_stride: usize,
        t_rows_us: usize,
        a_dim_us: usize,
        b_dim_us: usize,
        n_sites_f: u32,
        prob_lut: &[f64],
        stride1: usize,
        stride2: usize,
    ) -> (f64, f64) {
        let sg_u_row = &sg_u[s_idx * word_stride..(s_idx + 1) * word_stride];
        let sg_v_row = &sg_v[s_idx * word_stride..(s_idx + 1) * word_stride];
        let mut p_s = 0.0f64;
        let mut sum_p_ln_p = 0.0f64;

        // Reusable buffers to avoid recomputing (AL ⊕ BL ⊕ SG) inside the t-loop.
        // This cuts the per-word XORs in the hottest loop roughly in half.
        let mut base_u_buf: Vec<u64> = vec![0u64; word_stride];
        let mut base_v_buf: Vec<u64> = vec![0u64; word_stride];

        for a in 0..a_dim_us {
            let al_u_row = &al_u[a * word_stride..(a + 1) * word_stride];
            let al_v_row = &al_v[a * word_stride..(a + 1) * word_stride];
            for b in 0..b_dim_us {
                let bl_u_row = &bl_u[b * word_stride..(b + 1) * word_stride];
                let bl_v_row = &bl_v[b * word_stride..(b + 1) * word_stride];

                // base = AL[a] ⊕ BL[b] ⊕ SG[s]
                // Precompute once per (a,b,s), then just xor with Ttab in the t-loop.
                for w in 0..word_stride {
                    base_u_buf[w] = al_u_row[w] ^ bl_u_row[w] ^ sg_u_row[w];
                    base_v_buf[w] = al_v_row[w] ^ bl_v_row[w] ^ sg_v_row[w];
                }

                let mut total = 0.0f64;
                for t in 0..t_rows_us {
                    let tt_u_row = &ttab_u[t * word_stride..(t + 1) * word_stride];
                    let tt_v_row = &ttab_v[t * word_stride..(t + 1) * word_stride];

                    let mut nx = 0u32;
                    let mut nz = 0u32;
                    let mut ny = 0u32;
                    for w in 0..word_stride {
                        let u = base_u_buf[w] ^ tt_u_row[w];
                        let v = base_v_buf[w] ^ tt_v_row[w];
                        nx += (u & !v).count_ones();
                        nz += (!u & v).count_ones();
                        ny += (u & v).count_ones();
                    }
                    // Look up probability by (nx,nz,ny). If padding bits exist in the
                    // last word, they should be zeroed in the input tables; otherwise
                    // counts could exceed n_sites.
                    let sum = nx + nz + ny;
                    if sum <= n_sites_f {
                        let idx = (nx as usize) * stride2 + (nz as usize) * stride1 + (ny as usize);
                        total += prob_lut[idx];
                    }
                }

                p_s += total;
                if total > 0.0 {
                    sum_p_ln_p += total * total.ln();
                }
            }
        }
        (p_s, sum_p_ln_p)
    }

    // Decide whether we have enough syndromes to parallelize at the top level.
    let threads = rayon::current_num_threads().max(1);
    let par_over_s = s_rows >= 2 * threads;

    let per_s_results: Vec<SRes> = if par_over_s {
        // Parallelize across syndromes (good when 2^r is large enough).
        (0..s_rows)
            .into_par_iter()
            .map(|s_idx| {
                let (p_s, sum_p_ln_p) = compute_syndrome_contrib(
                    s_idx,
                    sg_u,
                    sg_v,
                    al_u,
                    al_v,
                    bl_u,
                    bl_v,
                    ttab_u,
                    ttab_v,
                    word_stride,
                    t_rows_us,
                    a_dim_us,
                    b_dim_us,
                    n_sites_f,
                    &prob_lut,
                    stride1,
                    stride2,
                );

                let p_weighted_entropy = if p_s > 0.0 {
                    (-sum_p_ln_p + p_s * p_s.ln()) * INV_LN_2
                } else {
                    0.0
                };

                SRes {
                    p_sum: p_s,
                    p_weighted_entropy,
                }
            })
            .collect()
    } else {
        // Small r => few syndromes. Parallelize over (s,a) tiles and reduce per syndrome.
        // This avoids underutilizing the thread pool when k is large.
        let total_tiles = s_rows * a_dim_us;

        // Reduce into per-s accumulators: (p_s, sum_p_ln_p).
        #[derive(Clone)]
        struct TileAcc {
            per_s: Vec<(f64, f64)>,
            base_u: Vec<u64>,
            base_v: Vec<u64>,
        }

        let reduced: Vec<(f64, f64)> = (0..total_tiles)
            .into_par_iter()
            .fold(
                || TileAcc {
                    per_s: vec![(0.0f64, 0.0f64); s_rows],
                    base_u: vec![0u64; word_stride],
                    base_v: vec![0u64; word_stride],
                },
                |mut acc, idx| {
                    let s_idx = idx / a_dim_us;
                    let a = idx % a_dim_us;

                    let sg_u_row = &sg_u[s_idx * word_stride..(s_idx + 1) * word_stride];
                    let sg_v_row = &sg_v[s_idx * word_stride..(s_idx + 1) * word_stride];
                    let al_u_row = &al_u[a * word_stride..(a + 1) * word_stride];
                    let al_v_row = &al_v[a * word_stride..(a + 1) * word_stride];

                    let mut p_s_partial = 0.0f64;
                    let mut sum_p_ln_p_partial = 0.0f64;

                    for b in 0..b_dim_us {
                        let bl_u_row = &bl_u[b * word_stride..(b + 1) * word_stride];
                        let bl_v_row = &bl_v[b * word_stride..(b + 1) * word_stride];

                        // base = AL[a] ⊕ BL[b] ⊕ SG[s]
                        for w in 0..word_stride {
                            acc.base_u[w] = al_u_row[w] ^ bl_u_row[w] ^ sg_u_row[w];
                            acc.base_v[w] = al_v_row[w] ^ bl_v_row[w] ^ sg_v_row[w];
                        }

                        let mut total = 0.0f64;
                        for t in 0..t_rows_us {
                            let tt_u_row = &ttab_u[t * word_stride..(t + 1) * word_stride];
                            let tt_v_row = &ttab_v[t * word_stride..(t + 1) * word_stride];

                            let mut nx = 0u32;
                            let mut nz = 0u32;
                            let mut ny = 0u32;
                            for w in 0..word_stride {
                                let u = acc.base_u[w] ^ tt_u_row[w];
                                let v = acc.base_v[w] ^ tt_v_row[w];
                                nx += (u & !v).count_ones();
                                nz += (!u & v).count_ones();
                                ny += (u & v).count_ones();
                            }
                            let sum = nx + nz + ny;
                            if sum <= n_sites_f {
                                let idx = (nx as usize) * stride2 + (nz as usize) * stride1 + (ny as usize);
                                total += prob_lut[idx];
                            }
                        }

                        p_s_partial += total;
                        if total > 0.0 {
                            sum_p_ln_p_partial += total * total.ln();
                        }
                    }

                    let (p_acc, s_acc) = acc.per_s[s_idx];
                    acc.per_s[s_idx] = (p_acc + p_s_partial, s_acc + sum_p_ln_p_partial);
                    acc
                },
            )
            .reduce(
                || TileAcc {
                    per_s: vec![(0.0f64, 0.0f64); s_rows],
                    base_u: vec![0u64; word_stride],
                    base_v: vec![0u64; word_stride],
                },
                |mut a, b| {
                    for i in 0..s_rows {
                        a.per_s[i].0 += b.per_s[i].0;
                        a.per_s[i].1 += b.per_s[i].1;
                    }
                    a
                },
            )
            .per_s;

        reduced
            .into_iter()
            .map(|(p_s, sum_p_ln_p)| {
                let p_weighted_entropy = if p_s > 0.0 {
                    (-sum_p_ln_p + p_s * p_s.ln()) * INV_LN_2
                } else {
                    0.0
                };
                SRes {
                    p_sum: p_s,
                    p_weighted_entropy,
                }
            })
            .collect()
    };

    // New: H_bar = Σ_s p(s) H(p(a',b'|s))
    let h_bar: f64 = per_s_results.iter().map(|sr| sr.p_weighted_entropy).sum::<f64>();

    // Hashing bound with the *conditional* entropy
    let k = k_log as f64;
    let n_f = n_sites as f64;
    let hashing_bound = (k - h_bar) / n_f;


    hashing_bound
}

