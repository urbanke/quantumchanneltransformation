# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 10:16:36 2025

@author: kudekar
"""

import itertools
import math
from collections import defaultdict

# =========================
# Binary symplectic helpers
# =========================

# Single-qubit Pauli -> (x,z) bits; order: I,X,Y,Z
PZ = {'I': (0,0), 'X': (1,0), 'Z': (0,1), 'Y': (1,1)}
LETTERS = ['I','X','Y','Z']

def pauli_str_to_symp(s):
    """'ZIX' -> binary symplectic vector (x|z) in {0,1}^{2n} as a tuple of ints."""
    n = len(s)
    x = [0]*n
    z = [0]*n
    for i,ch in enumerate(s):
        xi, zi = PZ[ch]
        x[i] = xi
        z[i] = zi
    return tuple(x+z)

def symp_inner(a, b):
    """<a,b> = a_x·b_z + a_z·b_x (mod 2); both a,b are length 2n tuples."""
    n2 = len(a); n = n2//2
    ax, az = a[:n], a[n:]
    bx, bz = b[:n], b[n:]
    dot = 0
    for i in range(n):
        dot ^= (ax[i] & bz[i]) ^ (az[i] & bx[i])
    return dot  # 0 or 1

def commute(a, b):
    """Return True iff Pauli(a) and Pauli(b) commute."""
    return symp_inner(a,b) == 0

def multiply_letters(a, b):
    """Multiply single-qubit Paulis up to global phase (resulting letter only)."""
    mult = {
        ('I','I'):'I', ('I','X'):'X', ('I','Y'):'Y', ('I','Z'):'Z',
        ('X','I'):'X', ('X','X'):'I', ('X','Y'):'Z', ('X','Z'):'Y',
        ('Y','I'):'Y', ('Y','X'):'Z', ('Y','Y'):'I', ('Y','Z'):'X',
        ('Z','I'):'Z', ('Z','X'):'Y', ('Z','Y'):'X', ('Z','Z'):'I',
    }
    return mult[(a,b)]

def multiply_strings(p, q):
    """Multiply multi-qubit Pauli strings up to global phase (letterwise)."""
    return ''.join(multiply_letters(a,b) for a,b in zip(p,q))

# =========================
# Problem setup: n=3, k=1
# =========================

n = 3
k = 1
g1 = "ZIX"
g2 = "IXX"
G = [g1, g2]
G_symp = [pauli_str_to_symp(g) for g in G]

# Sanity: generators commute
assert commute(G_symp[0], G_symp[1]), "Stabilizer generators must commute."

# -------- Find a logical pair (Xbar, Zbar) --------
# We search S^perp \ S: elements commuting with both g1,g2 but not in the small span {I, g1, g2, g1*g2},
# then pick a pair with <Xbar, Zbar> = 1.
span_small = {"III", g1, g2, multiply_strings(g1,g2)}
all_paulis = [''.join(p) for p in itertools.product(LETTERS, repeat=n)]
symp_map = {s: pauli_str_to_symp(s) for s in all_paulis}

normalizer = []
for s in all_paulis:
    if s == "III":
        continue
    v = symp_map[s]
    if all(commute(v, gs) for gs in G_symp) and (s not in span_small):
        normalizer.append(s)

Xbar = Zbar = None
for s1 in normalizer:
    for s2 in normalizer:
        if s1 == s2:
            continue
        if symp_inner(symp_map[s1], symp_map[s2]) == 1:
            Xbar, Zbar = s1, s2
            break
    if Xbar is not None:
        break

if Xbar is None:
    raise RuntimeError("Failed to find a logical pair; please check generators.")

def print_stabilizers_and_logicals():
    print("======== Code summary ========")
    print(f"n = {n}, k = {k}")
    print("Stabilizer generators:")
    for i, g in enumerate(G, start=1):
        print(f"  g{i} = {g}")
    print("Chosen logical operators:")
    print(f"  Xbar = {Xbar}")
    print(f"  Zbar = {Zbar}")
    # Sanity checks
    okX = all(commute(symp_map[Xbar], gs) for gs in G_symp)
    okZ = all(commute(symp_map[Zbar], gs) for gs in G_symp)
    anti = symp_inner(symp_map[Xbar], symp_map[Zbar]) == 1
    print(f"Checks: Xbar commutes with S: {okX},  Zbar commutes with S: {okZ},  <Xbar,Zbar>=1: {anti}")
    print("====================================")

print_stabilizers_and_logicals()

# ---------------------------
# Channel model: independent
# ---------------------------
def pauli_probs_independent_component(p):
    """
    Independent X/Z-component channel, symmetric:
      P(I)=(1-p)^2,  P(X)=p(1-p),  P(Z)=p(1-p),  P(Y)=p^2.
    Returns dict for single-qubit letter probabilities.
    """
    return {
        'I': (1-p)*(1-p),
        'X': p*(1-p),
        'Z': (1-p)*p,
        'Y': p*p
    }

def pattern_prob(pattern, letter_probs):
    prob = 1.0
    for ch in pattern:
        prob *= letter_probs[ch]
    return prob

# ==============================================
# Enumerate all 64 errors; bucket by (u,v,r)
# ==============================================

def compute_tables(p=0.11, verbose=True):
    letter_probs = pauli_probs_independent_component(p)

    # Accumulators
    Pr = defaultdict(float)                # P(r)
    Puvr = defaultdict(float)              # P((u,v), r)
    # (optionally) collect a small view per syndrome for inspection
    sample_rows = { (0,0):[], (0,1):[], (1,0):[], (1,1):[] }

    # Enumerate all 4^3 errors E
    for pattern in all_paulis:
        vE = symp_map[pattern]
        # syndrome r: two bits for the two generators
        r0 = symp_inner(vE, G_symp[0])  # 0/1
        r1 = symp_inner(vE, G_symp[1])  # 0/1
        r = (r0, r1)
        # induced logical letter: X^u Z^v with
        # u = <E, Zbar>, v = <E, Xbar>
        u = symp_inner(vE, symp_map[Zbar])  # commutation with Zbar => X exponent
        v = symp_inner(vE, symp_map[Xbar])  # commutation with Xbar => Z exponent
        uv = (u, v)
        # probability
        pe = pattern_prob(pattern, letter_probs)
        # accumulate
        Pr[r] += pe
        Puvr[(uv, r)] += pe
        # keep a few rows per r to sanity check (optional)
        if verbose and len(sample_rows[r]) < 4:
            sample_rows[r].append((pattern, uv, pe))

    # Per-syndrome conditionals and entropies
    cond = {}   # r -> dict[(u,v)] = bar p(u,v|r)
    Hr = {}     # r -> entropy in bits
    for r in [(0,0),(0,1),(1,0),(1,1)]:
        Z = Pr[r]
        if Z == 0.0:
            cond[r] = {(u,v): 0.0 for u in (0,1) for v in (0,1)}
            Hr[r] = 0.0
            continue
        probs = []
        for u in (0,1):
            for v in (0,1):
                uv = (u,v)
                val = Puvr.get((uv, r), 0.0) / Z
                cond.setdefault(r, {})[uv] = val
                probs.append(val)
        # entropy H_r
        H = 0.0
        for q in probs:
            if q > 0:
                H -= q*math.log(q, 2)
        Hr[r] = H

    # Induced hashing bound (no clipping), per block
    induced_per_block = sum(Pr[r]*(1.0 - Hr[r]) for r in Hr)

    if verbose:
        print("\n--- Per-syndrome summaries (first few rows each) ---")
        for r in [(0,0),(0,1),(1,0),(1,1)]:
            print(f"Syndrome r={r}: P(r)={Pr[r]:.12f},  H_r={Hr[r]:.12f},  1-H_r={1.0-Hr[r]:.12f}")
            for row in sample_rows[r]:
                patt, uv, pe = row
                print(f"  E={patt}  -> (u,v)={uv}   p(E)={pe:.12e}")
            print("  cond (u,v | r):", {uv: f"{cond[r][uv]:.12f}" for uv in cond[r]})
        print("\n--- Final ---")
        print(f"Induced hashing (per inner block, no clipping) at p={p:.6f}: {induced_per_block:.12f}")
        print(f"Per physical qubit (divide by n=3): {induced_per_block/3.0:.16f}")

    return {
        "Pr": Pr,
        "Puvr": Puvr,
        "cond": cond,
        "Hr": Hr,
        "induced_per_block": induced_per_block,
        "induced_per_qubit": induced_per_block/3.0
    }

# Run once (change p as needed)
if __name__ == "__main__":
    p = 0.11002786443835955
    results = compute_tables(p=p, verbose=True)
