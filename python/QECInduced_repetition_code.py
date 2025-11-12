#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full repetition (CSS/Shor) induced channel via two-stage maps:
   p  --T^X_m-->  p^(m)  --T^Z_ell-->  p_bar
where T^Z_ell is implemented by swapping X<->Z, applying T^X_ell, then swapping back.

We support:
  - Generic input p = (pI, pX, pZ, pY)
  - Independent X/Z channel with parameter p in [0,1]: U,V ~ Bern(p),  X^U Z^V
  - Depolarizing channel with parameter p in [0,1]: (pX,pY,pZ) = (p/3,p/3,p/3), pI=1-p

We compute:
  Hashing(original) = 1 - H(p)               (rate per physical qubit)
  Hashing(induced)  = (1 - H(p_bar)) / (m*l) (rate per physical qubit for the inner [[m*l,1]] code)

We also sweep p and plot both bounds for independent and depolarizing families.

Author: (you)
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Utilities
# ---------------------------

def entropy_base2(pvec):
    """
    Shannon entropy H(p) = - sum_i p_i log2 p_i, ignoring zero terms.
    pvec: iterable of probabilities (should sum ~ 1)
    """
    h = 0.0
    for x in pvec:
        if x > 0.0:
            h -= x * math.log2(x)
    return h

def normalize4(p):
    """
    Ensure numerical normalization of a 4-tuple.
    """
    s = sum(p)
    if s == 0.0:
        # pathological, return uniform to avoid NaNs
        return (0.25, 0.25, 0.25, 0.25)
    return tuple(x / s for x in p)


# ---------------------------
# Channels
# ---------------------------

def pauli_channel_independent(p):
    """
    Independent X/Z channel with parameter p in [0,1]:
       U,V ~ Bernoulli(p),  sigma = X^U Z^V.
    Then:
      pI = (1-p)^2
      pX = p(1-p)
      pZ = p(1-p)
      pY = p^2
    Returns 4-tuple (pI, pX, pZ, pY).
    """
    pI = (1 - p) ** 2
    pX = p * (1 - p)
    pZ = p * (1 - p)
    pY = p ** 2
    return normalize4((pI, pX, pZ, pY))

def pauli_channel_depolarizing(p):
    """
    Depolarizing channel with parameter p in [0,1]:
      pX = pY = pZ = p/3, pI = 1 - p
    Returns 4-tuple (pI, pX, pZ, pY).
    """
    pI = 1 - p
    r = p / 3.0
    return normalize4((pI, r, r, r))


# ---------------------------
# Swap X <-> Z
# ---------------------------

def swap_XZ(p):
    """
    Swap labels X and Z in a 4-tuple p = (pI, pX, pZ, pY).
    I and Y remain the same. Returns (pI, pZ, pX, pY).
    """
    pI, pX, pZ, pY = p
    return (pI, pZ, pX, pY)


# ---------------------------
# Single repetition-induced map T^X_m
# ---------------------------

def T_X_m(p, m):
    """
    Single repetition that protects X flips (Z-type checks), of odd length m.
    This is the closed form from our notes:
      Define:
        pIZ_plus  = pI + pZ
        pIZ_minus = pI - pZ
        pXY_plus  = pX + pY
        pXY_minus = pX - pY
      Then for odd m:
        pI_bar = 1/2 [ sum_{i=0..floor(m/2)} C(m,i) (pIZ_plus)^{m-i} (pXY_plus)^i
                        + sum_{i=0..floor(m/2)} C(m,i) (pIZ_minus)^{m-i} (pXY_minus)^i ]
        pZ_bar = 1/2 [ same first sum  - same second sum ]
        pX_bar = 1/2 [ 1 + (pIZ_minus + pXY_minus)^m ] - pI_bar
        pY_bar = 1/2 [ 1 - (pIZ_minus + pXY_minus)^m ] - pZ_bar

    Note:
      - m must be odd for this majority logic (as in the standard derivation).
      - The result is a 4-tuple (pI, pX, pZ, pY).
    """
    if m % 2 != 1:
        raise ValueError("T_X_m: m must be odd for majority-based repetition.")

    pI, pX, pZ, pY = p
    pIZ_plus  = pI + pZ
    pIZ_minus = pI - pZ
    pXY_plus  = pX + pY
    pXY_minus = pX - pY

    half = m // 2

    # Sum over i=0..floor(m/2) of C(m,i) a^{m-i} b^i
    def truncated_binomial(a, b):
        s = 0.0
        for i in range(half + 1):
            s += math.comb(m, i) * (a ** (m - i)) * (b ** i)
        return s

    S_plus  = truncated_binomial(pIZ_plus,  pXY_plus)
    S_minus = truncated_binomial(pIZ_minus, pXY_minus)

    pI_bar = 0.5 * (S_plus + S_minus)
    pZ_bar = 0.5 * (S_plus - S_minus)

    parity_term = (pIZ_minus + pXY_minus) ** m
    pX_bar = 0.5 * (1.0 + parity_term) - pI_bar
    pY_bar = 0.5 * (1.0 - parity_term) - pZ_bar

    return normalize4((pI_bar, pX_bar, pZ_bar, pY_bar))


# ---------------------------
# Two-stage full repetition (CSS/Shor): T^Z_ell ◦ T^X_m
# ---------------------------

def full_repetition_induced(p, m, ell):
    """
    Two-stage (CSS/Shor) repetition induced channel on one logical qubit:
       p ---> p^(m) = T^X_m(p)
         ---> p_bar = T^Z_ell(p^(m)).

    We compute T^Z_ell via:
       T^Z_ell = S^{-1} ◦ T^X_ell ◦ S,
    i.e., swap X<->Z, apply T^X_ell, swap back.
    Both m and ell must be odd.

    Returns: p_bar = (pI_bar, pX_bar, pZ_bar, pY_bar).
    """
    if m % 2 != 1 or ell % 2 != 1:
        raise ValueError("full_repetition_induced: m and ell must be odd.")
    # First stage: X-protection (Z checks)
    p_stage1 = T_X_m(p, m)
    # Second stage: Z-protection (X checks) via swaps
    p_swapped_in  = swap_XZ(p_stage1)
    q             = T_X_m(p_swapped_in, ell)  # apply X-formula
    p_bar         = swap_XZ(q)                # swap back
    #p_bar = p_stage1
    print("Induced channel after X protection: ", p, normalize4(p_stage1))
    return normalize4(p_bar)


# ---------------------------
# Hashing bounds
# ---------------------------

def hashing_bound_original(p):
    """
    Hashing bound (per physical qubit) of the original single-qubit Pauli channel:
      HB_orig = 1 - H(p)
    """
    return 1.0 - entropy_base2(p)

def hashing_bound_induced(p_bar, m, ell):
    """
    Hashing bound (per physical qubit) after the inner [[m*ell, 1]] code:
      HB_induced = (1 - H(p_bar)) / (m*ell)
    """
    n = m * ell
    return (1.0 - entropy_base2(p_bar)) / float(n)


# ---------------------------
# Demonstration / Sweep and Plots
# ---------------------------

def sweep_and_plot(m=3, ell=3, p_min=0.0, p_max=0.3, num=61):
    """
    Sweep p over [p_min, p_max] (num samples) and plot:
      - Independent channel: HB(original) vs HB(induced)
      - Depolarizing channel: HB(original) vs HB(induced)

    Arguments:
      m, ell : odd integers (lengths of the two repetition stages)
      p_min, p_max : range of p
      num : number of sample points
    """
    ps = np.linspace(p_min, p_max, num=num)

    # Storage for curves
    HB_indep_orig = []
    HB_indep_ind  = []

    HB_depol_orig = []
    HB_depol_ind  = []

    tuples_indep = []  # (p, HB_orig, HB_induced)
    tuples_depol = []

    for p in ps:
        # Independent
        p_indep = pauli_channel_independent(p)
        hb_orig_indep = hashing_bound_original(p_indep)
        pbar_indep = full_repetition_induced(p_indep, m, ell)
        hb_ind_indep = hashing_bound_induced(pbar_indep, m, ell)

        HB_indep_orig.append(hb_orig_indep)
        HB_indep_ind.append(hb_ind_indep)
        tuples_indep.append((p, hb_orig_indep, hb_ind_indep))

        # Depolarizing
        p_depol = pauli_channel_depolarizing(p)
        hb_orig_depol = hashing_bound_original(p_depol)
        pbar_depol = full_repetition_induced(p_depol, m, ell)
        hb_ind_depol = hashing_bound_induced(pbar_depol, m, ell)

        HB_depol_orig.append(hb_orig_depol)
        HB_depol_ind.append(hb_ind_depol)
        tuples_depol.append((p, hb_orig_depol, hb_ind_depol))

    # Print a few sample tuples
    print("\nSample tuples (Independent channel): (p, HB_original, HB_induced)")
    for row in tuples_indep[::max(1, len(tuples_indep)//5)]:
        print(row)

    print("\nSample tuples (Depolarizing channel): (p, HB_original, HB_induced)")
    for row in tuples_depol[::max(1, len(tuples_depol)//5)]:
        print(row)

    # Plot: Independent
    plt.figure(figsize=(7, 4.5))
    plt.plot(ps, HB_indep_orig, label="Original hashing bound (indep)", linewidth=2)
    plt.plot(ps, HB_indep_ind,  label=f"Induced hashing bound (m={m}, ℓ={ell})", linewidth=2)
    plt.xlabel("p")
    plt.ylabel("Hashing bound per physical qubit")
    plt.title("Independent X/Z channel: original vs induced")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Plot: Depolarizing
    plt.figure(figsize=(7, 4.5))
    plt.plot(ps, HB_depol_orig, label="Original hashing bound (depol)", linewidth=2)
    plt.plot(ps, HB_depol_ind,  label=f"Induced hashing bound (m={m}, ℓ={ell})", linewidth=2)
    plt.xlabel("p")
    plt.ylabel("Hashing bound per physical qubit")
    plt.title("Depolarizing channel: original vs induced")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()


# ---------------------------
# Main (example)
# ---------------------------

if __name__ == "__main__":
    # Example: Shor-style two-stage repetition with m=3 (X-protection), ell=3 (Z-protection)
    m = 3
    ell = 1

    # Show a single p example for each channel family
    p_example = 0.11002786443835955
    p_indep = pauli_channel_independent(p_example)
    p_depol = pauli_channel_depolarizing(p_example)

    print("Single example at p = ", p_example)
    print("Independent channel p:", p_indep)
    pbar_indep = full_repetition_induced(p_indep, m, ell)
    print("Induced (indep) p̄:", pbar_indep)
    print("HB(original, indep) =", hashing_bound_original(p_indep))
    print("HB(induced,  indep) =", hashing_bound_induced(pbar_indep, m, ell))

    print("\nDepolarizing channel p:", p_depol)
    pbar_depol = full_repetition_induced(p_depol, m, ell)
    print("Induced (depol) p̄:", pbar_depol)
    print("HB(original, depol) =", hashing_bound_original(p_depol))
    print("HB(induced,  depol) =", hashing_bound_induced(pbar_depol, m, ell))

    # Sweep and plot
    #sweep_and_plot(m=m, ell=ell, p_min=0.0, p_max=0.3, num=61)
