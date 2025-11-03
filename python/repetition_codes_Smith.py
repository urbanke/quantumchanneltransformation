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
    """Independent X/Z-component channel with z = x/9.
       Returns (pI, pX, pZ, pY) and per-qubit physical error rate perr = 1 - pI.
    """
    z = x/9.0
    # Single-qubit Pauli letter probabilities under independent flips:
    pI = (1.0 - x) * (1.0 - z)
    pX = x * (1.0 - z)
    pZ = z * (1.0 - x)
    pY = x * z
    # Normalize for safety (floating roundoff)
    s = pI + pX + pZ + pY
    if s <= 0:
        # degenerate corner; return something safe
        return (1.0, 0.0, 0.0, 0.0), 0.0
    pI, pX, pZ, pY = pI / s, pX / s, pZ / s, pY / s
    perr = 1.0 - pI
    return (pI, pX, pZ, pY), perr

def raw_hashing_bound_from_x(x):
    """Raw hashing bound 1 - H(pI,pX,pY,pZ) for the independent channel with z = x/9."""
    (pI, pX, pZ, pY), _ = pauli_probs_independent(x)
    return 1.0 - H_base2([pI, pX, pY, pZ])

# =======================================================
# Smith's joint P( X^u Z^v (bar), r ) for one pattern
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
    #print("Z:= ", Z)
    if Z <= 0:
        # Numerically degenerate; return uniform to avoid NaNs
        return np.full((2, 2), 0.25, dtype=float)
    return J / Z

def induced_hashing_bound_repetition(k, pI, pX, pZ, pY):
    """
    Final induced hashing bound for the repetition code of length k
    under the Pauli law (pI,pX,pZ,pY) (Smith specialization).
    hashing_induced = sum_r P(r) * [1 - H( \bar p(·|r) )], with P(r) normalized to 1.
    Returns (hashing_induced, P_r, hashing_r_per_syndrome).
    """
    # P(r) (including multiplicity) for r=0..k-1
    Pr = np.array([syndrome_weight_mass(r, k, pI, pX, pZ, pY) for r in range(k)], dtype=float)
    #Pr = np.clip(Pr, 0.0, None)
    S = Pr.sum()
    #print("S:= ", S)
    #assert S == 1.0, "Pr(r) sums to 1"
    if S > 0:
        Pr /= S
    else:
        Pr[:] = 0.0
        Pr[0] = 1.0

    hashing_r = np.zeros(k, dtype=float)
    for r in range(k):
        pbar = conditional_logical_given_r(r, k, pI, pX, pZ, pY)  # 2x2 over (u,v)
        hashing_r[r] = max(0,(1.0 - H_base2(pbar.flatten()))/k) #(1.0 - H_base2(pbar.flatten()))/k

    hashing_induced = float(np.dot(Pr, hashing_r))
    return hashing_induced, Pr, hashing_r

# =========================
# Demo / Plotting
# =========================
if __name__ == "__main__":
    # Repetition code length
    plt.figure()
    for k in range(3,34):  

        # Sweep x (X-flip prob). 
        x_grid = np.linspace(0.2, 0.5, 200)
    
        orig_vals = []
        ind_vals = []
        perr = []
    
        for x in x_grid:
            # Independent channel with skew z = x/9
            (pI, pX, pZ, pY), p = pauli_probs_independent(x)
            perr.append(p)
    
            # Raw hashing bound
            hashing_orig = 1.0 - H_base2([pI, pX, pY, pZ])
    
            # Induced hashing bound for repetition code (Smith)
            hashing_ind, Pr, hashing_r = induced_hashing_bound_repetition(k, pI, pX, pZ, pY)
    
            orig_vals.append(hashing_orig)
            ind_vals.append(hashing_ind)
    
        orig_vals = np.array(orig_vals, dtype=float)
        ind_vals = np.array(ind_vals, dtype=float)
    
        # Plot both vs x
        plt.plot(perr, orig_vals, label="Original hashing bound (indep., z=x/9)")
        plt.plot(perr, ind_vals, label=f"Induced hashing bound (repetition k={k})")
        plt.xlabel("x  (X-flip probability)")
        plt.ylabel("Hashing bound (bits per qubit)")
        plt.title("Induced vs Original Hashing Bound (independent channel, z = x/9)")
        plt.grid(True)
        #plt.legend()
        plt.tight_layout()
        plt.show()
    
        # Optional: quick text sanity for a few x
        # for x_test in [0.255]:
        #     (pI, pX, pZ, pY), perr = pauli_probs_independent(x_test)
        #     hashing_orig = 1.0 - H_base2([pI, pX, pY, pZ])
        #     hashing_ind, Pr, hashing_r = induced_hashing_bound_repetition(k, pI, pX, pZ, pY)
        #     print(f"\nx={x_test:.3f}, z={x_test/9:.5f}")
        #     print(f"pI={pI:.6f}, pX={pX:.6f}, pY={pY:.6f}, pZ={pZ:.6f}, perr={perr:.6f}")
        #     print(f"Original bound:     {hashing_orig:.6f}")
        #     print(f"Induced bound: {hashing_ind:.6f}")
