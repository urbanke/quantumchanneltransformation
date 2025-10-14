//! rust-kernel: fast inner loops for induced channel computation.
//!
//! Exports a single C-ABI function that:
//! - takes bit-packed (U,V) tables: Ttab (2^r × W), AL (A × W), BL (A × W), SG (S × W)
//! - computes P_s(a,b) = sum_t P(E) with E = base ⊕ T[t], base = AL[a] ⊕ BL[b] ⊕ SG[s],
//! - finds ML (a*(s), b*(s)),
//! - accumulates p̄(a′,b′) = Σ_s P_s(a′⊕a*(s), b′⊕b*(s)),
//! - writes p̄ into pbar_out (size A×A), returns hashing bound (k - H(p̄))/n.
//!
//! All rows are bit-packed UInt64 words representing (u|v) halves separately.
//! Popcount uses u64::count_ones() and sums.

use rayon::prelude::*;
use std::slice;

#[inline(always)]
fn popcount_u64_slice(xs: &[u64]) -> u64 {
    xs.iter().map(|w| w.count_ones() as u64).sum()
}

#[inline(always)]
fn counts_by_type_bitpacked(u: &[u64], v: &[u64], n_sites: usize) -> (u64, u64, u64, u64) {
    // X = U & !V, Z = !U & V, Y = U & V, I = n - (X+Z+Y).
    let mut nx = 0u64;
    let mut nz = 0u64;
    let mut ny = 0u64;

    for (uu, vv) in u.iter().zip(v.iter()) {
        let xmask = uu & !vv;
        let zmask = !uu & *vv;
        let ymask = uu & *vv;
        nx += xmask.count_ones() as u64;
        nz += zmask.count_ones() as u64;
        ny += ymask.count_ones() as u64;
    }
    let ni = (n_sites as u64) - (nx + nz + ny);
    (ni, nx, nz, ny)
}

#[inline(always)]
fn p_of_counts(ni: u64, nx: u64, nz: u64, ny: u64, pI: f64, pX: f64, pZ: f64, pY: f64) -> f64 {
    pI.powi(ni as i32) * pX.powi(nx as i32) * pZ.powi(nz as i32) * pY.powi(ny as i32)
}

#[inline(always)]
fn shannon_entropy_base2(p: &[f64]) -> f64 {
    p.iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| -x * x.log2())
        .sum()
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

    // sizes & channel
    n_sites: u32,
    r_log: u32,
    k_log: u32,
    pI: f64,
    pX: f64,
    pZ: f64,
    pY: f64,

    // outputs
    pbar_out_ptr: *mut f64, // length A*B
) -> f64 {
    // Safety: Julia passes valid pointers/sizes.
    let ttab_u = unsafe { slice::from_raw_parts(ttab_u_ptr, t_rows * words) };
    let ttab_v = unsafe { slice::from_raw_parts(ttab_v_ptr, t_rows * words) };
    let al_u = unsafe { slice::from_raw_parts(al_u_ptr, a_rows * words) };
    let al_v = unsafe { slice::from_raw_parts(al_v_ptr, a_rows * words) };
    let bl_u = unsafe { slice::from_raw_parts(bl_u_ptr, b_rows * words) };
    let bl_v = unsafe { slice::from_raw_parts(bl_v_ptr, b_rows * words) };
    let sg_u = unsafe { slice::from_raw_parts(sg_u_ptr, s_rows * words) };
    let sg_v = unsafe { slice::from_raw_parts(sg_v_ptr, s_rows * words) };

    let n = n_sites as usize;
    let a_dim = a_rows;
    let b_dim = b_rows;
    let s_dim = s_rows;
    let t_dim = t_rows;

    // P_s table (S × A × B) is not stored explicitly to save memory.
    // Instead, we compute argmax (a*(s), b*(s)) on the fly by scanning (a,b),
    // then add shifted contributions into pbar.

    let mut pbar = vec![0.0f64; a_dim * b_dim];

    // Iterate s in parallel (Rayon), each thread keeps a local pbar and reduces at the end.
    let pbar_sum = (0..s_dim)
        .into_par_iter()
        .map(|si| {
            // Access SG row
            let sg_u_row = &sg_u[si * words..(si + 1) * words];
            let sg_v_row = &sg_v[si * words..(si + 1) * words];

            // Find argmax (a*, b*)
            let mut best_val = -1.0f64;
            let mut best_ai = 0usize;
            let mut best_bi = 0usize;

            // We also collect, for argmax shift later, the full P_s grid in a scratch buffer
            // to avoid recomputing. If memory is tight, compute twice; here we keep it.
            let mut ps_grid = vec![0.0f64; a_dim * b_dim];

            for ai in 0..a_dim {
                let al_u_row = &al_u[ai * words..(ai + 1) * words];
                let al_v_row = &al_v[ai * words..(ai + 1) * words];

                // a_s = AL[a] ⊕ SG[s]
                let mut a_s_u = vec![0u64; words];
                let mut a_s_v = vec![0u64; words];
                for w in 0..words {
                    a_s_u[w] = al_u_row[w] ^ sg_u_row[w];
                    a_s_v[w] = al_v_row[w] ^ sg_v_row[w];
                }

                for bi in 0..b_dim {
                    let bl_u_row = &bl_u[bi * words..(bi + 1) * words];
                    let bl_v_row = &bl_v[bi * words..(bi + 1) * words];

                    // base = a_s ⊕ BL[b]
                    let mut base_u = vec![0u64; words];
                    let mut base_v = vec![0u64; words];
                    for w in 0..words {
                        base_u[w] = a_s_u[w] ^ bl_u_row[w];
                        base_v[w] = a_s_v[w] ^ bl_v_row[w];
                    }

                    // Sum over all t: E = base ⊕ T[t]
                    let mut acc = 0.0f64;
                    for ti in 0..t_dim {
                        let t_u_row = &ttab_u[ti * words..(ti + 1) * words];
                        let t_v_row = &ttab_v[ti * words..(ti + 1) * words];

                        // E = base ⊕ T[t]
                        // counts by type
                        let mut e_u = vec![0u64; words];
                        let mut e_v = vec![0u64; words];
                        for w in 0..words {
                            e_u[w] = base_u[w] ^ t_u_row[w];
                            e_v[w] = base_v[w] ^ t_v_row[w];
                        }
                        let (ni, nx, nz, ny) = counts_by_type_bitpacked(&e_u, &e_v, n);
                        let p = p_of_counts(ni, nx, nz, ny, pI, pX, pZ, pY);
                        acc += p;
                    }

                    let idx = ai * b_dim + bi;
                    ps_grid[idx] = acc;
                    if acc > best_val {
                        best_val = acc;
                        best_ai = ai;
                        best_bi = bi;
                    }
                }
            }

            // Shift-add into a local pbar_s via XOR on indices:
            // p̄(a′,b′) += P_s(a′⊕a*(s), b′⊕b*(s))
            // XOR on indices is bitwise XOR on the k-bit representations 0..A-1.
            let mut pbar_local = vec![0.0f64; a_dim * b_dim];
            for a_prime in 0..a_dim {
                let src_a = a_prime ^ best_ai;
                for b_prime in 0..b_dim {
                    let src_b = b_prime ^ best_bi;
                    pbar_local[a_prime * b_dim + b_prime] += ps_grid[src_a * b_dim + src_b];
                }
            }
            pbar_local
        })
        .reduce(
            || vec![0.0f64; a_dim * b_dim],
            |mut acc, local| {
                for i in 0..acc.len() {
                    acc[i] += local[i];
                }
                acc
            },
        );

    // Normalize pbar and compute hashing bound
    let total: f64 = pbar_sum.iter().sum();
    let mut pbar_norm = vec![0.0f64; pbar_sum.len()];
    if total > 0.0 {
        for i in 0..pbar_sum.len() {
            pbar_norm[i] = pbar_sum[i] / total;
        }
    }

    let h = shannon_entropy_base2(&pbar_norm);
    let k = k_log as f64;
    let n_f = (n_sites as f64);
    let hashing_bound = (k - h) / n_f;

    // Write pbar to output
    let out = unsafe { slice::from_raw_parts_mut(pbar_out_ptr, pbar_norm.len()) };
    out.copy_from_slice(&pbar_norm);

    hashing_bound
}

