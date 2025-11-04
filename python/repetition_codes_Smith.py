import numpy as np
from math import comb
import matplotlib.pyplot as plt

# =========================
# Basic helpers: entropy H
# =========================
def H_base2(p):
    """
    Shannon entropy H(p) with base-2 logarithm.
    p: 1D iterable of probabilities that (approximately) sum to 1.
    """
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    nz = p > 0
    return -np.sum(p[nz] * np.log2(p[nz]))

# ============================================
# Channel: independent X/Z with skew z = x/9
# ============================================
def pauli_probs_independent(x):
    """
    Independent X/Z-component channel with z = x/9.

    Returns:
      (pI, pX, pZ, pY): Pauli letter probs
      perr: per-qubit physical error rate = 1 - pI
    """
    z = x / 9.0
    pI = (1.0 - x) * (1.0 - z)
    pX = x * (1.0 - z)
    pZ = z * (1.0 - x)
    pY = x * z

    s = pI + pX + pZ + pY
    if s <= 0:
        return (1.0, 0.0, 0.0, 0.0), 0.0
    pI, pX, pZ, pY = pI / s, pX / s, pZ / s, pY / s
    perr = 1.0 - pI
    return (pI, pX, pZ, pY), perr

def raw_hashing_bound_from_x(x):
    """
    Raw hashing bound 1 - H(pI,pX,pY,pZ) for the independent channel with z = x/9.
    """
    (pI, pX, pZ, pY), _ = pauli_probs_independent(x)
    return 1.0 - H_base2([pI, pX, pY, pZ])

# =======================================================
# Smith specialization: repetition code induced bound
# =======================================================
def joint_one_pattern_smith(u, v, r, k, pI, pX, pZ, pY):
    """
    Smith formula for a single syndrome pattern (of weight r):
      P = 0.5 * [ (pX+pY)^t (pI+pZ)^s + (-1)^v (pX-pY)^t (pI-pZ)^s ],
    where t = u*(k-2r)+r and s = (1-u)*(k-2r)+r.
    """
    t = u * (k - 2 * r) + r
    s = (1 - u) * (k - 2 * r) + r
    A = (pX + pY) ** t * (pI + pZ) ** s
    B = (pX - pY) ** t * (pI - pZ) ** s
    return 0.5 * (A + ((-1) ** v) * B)

def syndrome_weight_mass(r, k, pI, pX, pZ, pY):
    """
    P(r) = [#patterns with weight r] * sum_{u,v in {0,1}} P(u,v,r for one pattern).
    """
    mult = comb(k - 1, r)
    total = 0.0
    for u in (0, 1):
        for v in (0, 1):
            total += joint_one_pattern_smith(u, v, r, k, pI, pX, pZ, pY)
    return mult * total

def conditional_logical_given_r(r, k, pI, pX, pZ, pY):
    """
    \bar p(u,v | r) computed from the per-pattern joint (identical across patterns of same weight).
    Returns a 2x2 array over (u,v) ∈ {0,1}^2.
    """
    J = np.zeros((2, 2), dtype=float)
    for u in (0, 1):
        for v in (0, 1):
            J[u, v] = joint_one_pattern_smith(u, v, r, k, pI, pX, pZ, pY)
    Z = J.sum()
    if Z <= 0:
        # Numerically degenerate; return uniform to avoid NaNs
        return np.full((2, 2), 0.25, dtype=float)
    return J / Z

def induced_hashing_bound_repetition(k, pI, pX, pZ, pY):
    """
    Induced hashing bound for a repetition code of length k under Pauli law (pI,pX,pZ,pY).

    hashing_induced = sum_r P(r) * [max(0, 1 - H( \bar p(·|r) ))] / k, with P(r) normalized.
    Returns:
      hashing_induced (float)
      Pr (np.array length k): P(r)
      hashing_r (np.array length k): per-r contribution (already /k and max(0,.))
    """
    # P(r) for r = 0..k-1 (including multiplicity)
    Pr = np.array([syndrome_weight_mass(r, k, pI, pX, pZ, pY) for r in range(k)], dtype=float)
    S = Pr.sum()
    if S > 0:
        Pr /= S
    else:
        Pr[:] = 0.0
        Pr[0] = 1.0

    hashing_r = np.zeros(k, dtype=float)
    for r in range(k):
        pbar = conditional_logical_given_r(r, k, pI, pX, pZ, pY)  # 2x2 over (u,v)
        hashing_r[r] = (1.0 - H_base2(pbar.flatten())) / k

    hashing_induced = float(np.dot(Pr, hashing_r))
    return hashing_induced, Pr, hashing_r

# =========================
# Demo / Plotting
# =========================
if __name__ == "__main__":
    # ---- Sweep the channel parameter (x), map to perr ----
    # x := X-flip probability; Z-flip uses z = x/9
    x_grid = np.linspace(0.2, 0.3, 200)

    perr_grid = np.empty_like(x_grid)
    orig_vals = np.empty_like(x_grid)
    for i, x in enumerate(x_grid):
        (pI, pX, pZ, pY), p = pauli_probs_independent(x)
        perr_grid[i] = p
        orig_vals[i] = 1.0 - H_base2([pI, pX, pY, pZ])

    # ---- Range of repetition lengths to consider ----
    k_min, k_max = 3, 33
    ks = list(range(k_min, k_max + 1))

    # Compute induced bounds for each k, at every x
    all_ind_vals = np.zeros((len(ks), len(x_grid)), dtype=float)
    for ki, k in enumerate(ks):
        for xi, x in enumerate(x_grid):
            (pI, pX, pZ, pY), _ = pauli_probs_independent(x)
            val, _, _ = induced_hashing_bound_repetition(k, pI, pX, pZ, pY)
            all_ind_vals[ki, xi] = val

    # ---- Pointwise "envelope": best k at each channel param ----
    # best_over_k[xi] = max_k all_ind_vals[k, xi]
    best_over_k = np.max(all_ind_vals, axis=0)
    argmax_k_idx = np.argmax(all_ind_vals, axis=0)     # indices into ks
    k_star_per_x = np.array([ks[i] for i in argmax_k_idx], dtype=int)

    # ---- Plot: (1) raw hashing bound; (2) some faint k-curves; (3) best-over-k envelope ----
    plt.figure(figsize=(10, 6))

    # (1) Raw hashing bound for context
    plt.plot(perr_grid, orig_vals, linewidth=1.0, alpha=0.9,
             label="Original hashing bound (indep., z=x/9)")

    # (2) Optional: show a handful of k-curves faintly
    sample_ks_to_plot = [3, 5, 7, 11, 17, 25, 33]  # edit as you like
    for k, yk in zip(ks, all_ind_vals):
        if k in sample_ks_to_plot:
            plt.plot(perr_grid, yk, linewidth=0.9, alpha=0.30, label=f"Induced (k={k})")
        else:
            plt.plot(perr_grid, yk, linewidth=0.5, alpha=0.10)

    # (3) Best (envelope) over k
    plt.plot(perr_grid, best_over_k, linewidth=2.2, linestyle="-",
             label="Envelope = max over k")

    plt.xlabel("Per-qubit physical error rate  perr = 1 - pI")
    plt.ylabel("Achievable rate (bits per qubit)")
    plt.title(f"Induced vs Original Hashing Bound; Envelope is pointwise max over k ∈ [{k_min},{k_max}]")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    plt.show()
