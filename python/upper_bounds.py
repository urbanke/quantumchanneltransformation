# [>-] Upper bounds for qubit Pauli channels + plotting
# Bounds included:
#  - EA/2 upper bound (all channels)
#  - Entanglement-breaking (EB) => Q=0 test (all channels; returns 0 or inf)
#  - SSC convex-envelope bounds (closed-form) for:
#       * BB84 / independent symmetric px=pz=p
#       * depolarizing
#  - AD bounds (optional) using epsilon from an SDP (cvxpy):
#       * BB84 symmetric
#       * depolarizing
# Plotting:
#  - Plots all available bounds for each channel on same figure.

import numpy as np
import math
import matplotlib.pyplot as plt

# --------------------------
# Basic entropy utilities
# --------------------------
def h2(x):
    """Binary entropy h2(x) in bits. Vectorized."""
    x = np.asarray(x, dtype=float)
    eps = 1e-300
    x = np.clip(x, eps, 1 - eps)
    return -(x*np.log2(x) + (1-x)*np.log2(1-x))

def H4(pI, pX, pY, pZ):
    """Shannon entropy of 4-outcome distribution in bits. Vectorized."""
    P = np.vstack([pI, pX, pY, pZ]).T
    P = np.clip(P, 1e-300, 1.0)
    return -np.sum(P * np.log2(P), axis=1)

# --------------------------
# Pauli-channel families
# --------------------------
def pauli_probs_independent(px, pz):
    """
    Independent X/Z flips: apply X with prob px and Z with prob pz independently.
    Then Y occurs when both happen.
    """
    pI = (1 - px) * (1 - pz)
    pX = px * (1 - pz)
    pZ = pz * (1 - px)
    pY = px * pz
    return pI, pX, pY, pZ

def pauli_probs_depolarizing(p):
    """Depolarizing with total non-identity probability p."""
    pI = 1 - p
    pX = p / 3
    pY = p / 3
    pZ = p / 3
    return pI, pX, pY, pZ

def probs_case_independent_sym(p):
    """Independent symmetric: px=pz=p."""
    return pauli_probs_independent(p, p)

def probs_case_skewed(p):
    """Skewed independent: px=p, pz=p/9."""
    return pauli_probs_independent(p, p/9)

# --------------------------
# General bounds
# --------------------------
def ub_entanglement_assisted_half(pI, pX, pY, pZ):
    """
    EA upper bound:
      Q <= 1 - 1/2 * H(pI,pX,pY,pZ)
    """
    return 1.0 - 0.5 * H4(pI, pX, pY, pZ)

def pauli_eigenvalues_from_probs(pI, pX, pY, pZ):
    """
    Pauli eigenvalues (Bloch contractions) for a qubit Pauli channel:
      λX = pI + pX - pY - pZ
      λY = pI - pX + pY - pZ
      λZ = pI - pX - pY + pZ
    Vectorized.
    """
    lamX = pI + pX - pY - pZ
    lamY = pI - pX + pY - pZ
    lamZ = pI - pX - pY + pZ
    return lamX, lamY, lamZ

def ub_entanglement_breaking_cutoff(pI, pX, pY, pZ):
    """
    For unital qubit channels (incl. Pauli): if |λX|+|λY|+|λZ| <= 1 then channel is EB => Q=0.
    Returns 0.0 when EB condition holds, else +inf (meaning "no numeric upper bound from EB").
    """
    lamX, lamY, lamZ = pauli_eigenvalues_from_probs(pI, pX, pY, pZ)
    s = np.abs(lamX) + np.abs(lamY) + np.abs(lamZ)
    return np.where(s <= 1.0 + 1e-12, 0.0, np.inf)

# --------------------------
# Convex envelope utilities
# --------------------------
def convex_minorant_piecewise_linear(x, y):
    """
    Lower convex hull / convex minorant (envelope from below), enforcing nondecreasing slopes.
    Returns indices of hull points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert np.all(np.diff(x) > 0)

    hull = []

    def slope(i, j):
        return (y[j] - y[i]) / (x[j] - x[i])

    for j in range(len(x)):
        hull.append(j)
        while len(hull) >= 3:
            i1, i2, i3 = hull[-3], hull[-2], hull[-1]
            # convex => slope(i1,i2) <= slope(i2,i3)
            if slope(i1, i2) > slope(i2, i3) + 1e-15:
                hull.pop(-2)
            else:
                break
    return np.array(hull, dtype=int)

def eval_piecewise_linear(x, y, hull_idx, xq):
    """Evaluate hull-defined piecewise-linear function at query points xq."""
    x = np.asarray(x); y = np.asarray(y); xq = np.asarray(xq)
    hx = x[hull_idx]; hy = y[hull_idx]

    seg = np.searchsorted(hx, xq, side='right') - 1
    seg = np.clip(seg, 0, len(hx) - 2)

    x0, x1 = hx[seg], hx[seg + 1]
    y0, y1 = hy[seg], hy[seg + 1]
    t = (xq - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def convex_envelope_of_min(x, curves):
    """
    curves: list of y arrays sampled on same x grid.
    We take pointwise min, then compute the lower convex envelope (convex minorant).
    """
    y_min = np.minimum.reduce(curves)
    hull = convex_minorant_piecewise_linear(x, y_min)
    return y_min, hull

# --------------------------
# SSC bounds (closed forms)
# --------------------------
def ub_bb84_ssc_sym(p_grid):
    """
    BB84 / independent symmetric SSC bound (convex envelope of two curves):
      conv{ 1 - h(2p(1-p)),  h(1/2 - 2p(1-p)) - h(2p(1-p)) }.
    Applies to independent symmetric px=pz=p.
    """
    p = np.asarray(p_grid, dtype=float)
    q = 2 * p * (1 - p)
    f1 = 1 - h2(q)
    f2 = h2(0.5 - q) - h2(q)
    y_min, hull = convex_envelope_of_min(p, [f1, f2])
    return eval_piecewise_linear(p, y_min, hull, p)

def gamma_depolarizing(p):
    """
    Gamma function used in a common depolarizing SSC convex-envelope form.
    NOTE: Literature has multiple equivalent parameterizations; keep consistent with your paper.
    Here:
      γ(p) = 4 * sqrt(1-p) * (1 - sqrt(1-p)).
    """
    p = np.asarray(p, dtype=float)
    t = np.sqrt(np.clip(1.0 - p, 0.0, 1.0))
    return 4.0 * t * (1.0 - t)

def ub_depolarizing_ssc(p_grid):
    """
    Depolarizing SSC bound (convex envelope of three curves):
      conv{ 1-h(p),  h((1+γ)/2)-h(γ/2),  1-4p }.
    """
    p = np.asarray(p_grid, dtype=float)
    f1 = 1 - h2(p)
    g = gamma_depolarizing(p)
    f2 = h2((1 + g) / 2) - h2(g / 2)
    f3 = 1 - 4 * p
    y_min, hull = convex_envelope_of_min(p, [f1, f2, f3])
    return eval_piecewise_linear(p, y_min, hull, p)

# --------------------------
# Approximate degradability via SDP (optional)
# --------------------------
def perm_matrix_for_choi_to_transfer(d_in, d_out):
    """
    Permutation P s.t. vec(T) = P vec(J), where:
      J indices (i,k ; j,l) with i,j in out and k,l in in
      T indices (i,j ; k,l)
    """
    N = d_out * d_in
    P = np.zeros((d_out*d_out*d_in*d_in, d_out*d_in*d_out*d_in), dtype=float)
    for i in range(d_out):
        for k in range(d_in):
            for j in range(d_out):
                for l in range(d_in):
                    rowJ = i*d_in + k
                    colJ = j*d_in + l
                    idxJ = rowJ + N * colJ

                    rowT = i*d_out + j
                    colT = k*d_in + l
                    idxT = rowT + (d_out*d_out) * colT
                    P[idxT, idxJ] = 1.0
    return P

def choi_from_kraus(kraus, d_in=2, d_out=2):
    """Choi matrix via J = sum_i vec(K_i) vec(K_i)†."""
    J = np.zeros((d_out * d_in, d_out * d_in), dtype=complex)
    for K in kraus:
        v = K.reshape(-1, order='F')  # column-major vec
        J += np.outer(v, np.conjugate(v))
    return J

def choi_complement_from_kraus(kraus, d_in=2):
    """
    Choi of complementary channel Φ^c for Kraus {K_i}:
      [Φ^c(ρ)]_{ij} = Tr(K_j† K_i ρ).
    Output dimension = number of Kraus ops.
    """
    r = len(kraus)
    dE = r
    J = np.zeros((dE * d_in, dE * d_in), dtype=complex)

    for m in range(d_in):
        for n in range(d_in):
            out = np.zeros((r, r), dtype=complex)
            for i in range(r):
                for j in range(r):
                    A = kraus[j].conj().T @ kraus[i]
                    out[i, j] = A[n, m]
            for i in range(r):
                for j in range(r):
                    J[i*d_in + m, j*d_in + n] = out[i, j]
    return J

def choi_to_transfer(J, d_in, d_out):
    """Convert Choi J (out ⊗ in) to transfer matrix T."""
    JJ = J.reshape(d_out, d_in, d_out, d_in)
    TT = np.transpose(JJ, (0, 2, 1, 3)).reshape(d_out*d_out, d_in*d_in)
    return TT

def epsilon_degradable_pauli(pI, pX, pY, pZ, solver="SCS", verbose=False):
    """
    Compute ε ≈ inf_D || Φ^c - D∘Φ ||_⋄ for a qubit Pauli channel via an SDP (cvxpy).
    This is expensive; use a coarse grid.

    Requires: cvxpy installed.
    """
    import cvxpy as cp

    # Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    probs = [float(pI), float(pX), float(pY), float(pZ)]
    probs = [max(0.0, pi) for pi in probs]
    paulis = [I, X, Y, Z]
    kraus = [math.sqrt(probs[i]) * paulis[i] for i in range(4)]

    dA, dB, dE = 2, 2, 4  # input, output, environment dims

    J_Phi = choi_from_kraus(kraus, d_in=dA, d_out=dB)
    J_Phic = choi_complement_from_kraus(kraus, d_in=dA)
    T_Phi = choi_to_transfer(J_Phi, d_in=dA, d_out=dB)

    # Variables: Choi of degrading map Xi: B->E (dimension dE x dB)
    J_Xi = cp.Variable((dE*dB, dE*dB), complex=True, hermitian=True)

    # Convert J_Xi -> transfer T_Xi via permutation
    P_J2T = perm_matrix_for_choi_to_transfer(d_in=dB, d_out=dE)
    vec_JXi = cp.reshape(J_Xi, (dE*dB*dE*dB, 1), order='F')
    vec_TXi = P_J2T @ vec_JXi
    T_Xi = cp.reshape(vec_TXi, (dE*dE, dB*dB), order='F')

    # Composition: T_comp = T_Xi T_Phi
    T_comp = T_Xi @ T_Phi

    # Convert transfer -> Choi via inverse permutation
    P_T2J = P_J2T.T
    vec_Tcomp = cp.reshape(T_comp, (dE*dE*dA*dA, 1), order='F')
    vec_Jcomp = P_T2J @ vec_Tcomp
    J_comp = cp.reshape(vec_Jcomp, (dE*dA, dE*dA), order='F')

    # Helper: partial trace over output (first subsystem) of Choi (out ⊗ in)
    def ptr_out_linear(J, d_out, d_in):
        J4 = cp.reshape(J, (d_out, d_in, d_out, d_in), order='F')
        blocks = []
        for a in range(d_in):
            row = []
            for b in range(d_in):
                s = 0
                for o in range(d_out):
                    s += J4[o, a, o, b]
                row.append(s)
            blocks.append(row)
        return cp.bmat(blocks)

    # CPTP constraints for Xi: J_Xi >= 0 and Tr_out(J_Xi) = I_in
    constraints = [
        J_Xi >> 0,
        ptr_out_linear(J_Xi, d_out=dE, d_in=dB) == np.eye(dB),
    ]

    # Diamond norm distance SDP:
    #   min 2*mu s.t. Z >= 0, Z >= J(Δ), Tr_out(Z) <= mu I
    Zvar = cp.Variable((dE*dA, dE*dA), complex=True, hermitian=True)
    mu = cp.Variable(nonneg=True)

    J_Delta = J_Phic - J_comp
    constraints += [
        Zvar >> 0,
        Zvar - J_Delta >> 0,
        ptr_out_linear(Zvar, d_out=dE, d_in=dA) << mu * np.eye(dA),
    ]

    prob = cp.Problem(cp.Minimize(2*mu), constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"epsilon SDP failed: status={prob.status}")
    return float(prob.value)

def ub_depolarizing_ad(p_grid, eps_grid):
    """
    AD depolarizing-style bound:
      conv{ 1-h(p) + eps*log2(3) + (1+eps/2) h(eps/(2+eps)), 1-4p }.
    """
    p = np.asarray(p_grid, dtype=float)
    eps = np.asarray(eps_grid, dtype=float)
    fA = 1 - h2(p) + eps*np.log2(3.0) + (1 + eps/2.0)*h2(eps/(2.0 + eps))
    fB = 1 - 4*p
    y_min, hull = convex_envelope_of_min(p, [fA, fB])
    return eval_piecewise_linear(p, y_min, hull, p)

def ub_bb84_ad_sym(p_grid, eps_grid):
    """
    AD BB84 symmetric-style bound:
      1 - h(2p(1-p)) + eps + (1+eps/2) h(eps/(2+eps)).
    """
    p = np.asarray(p_grid, dtype=float)
    eps = np.asarray(eps_grid, dtype=float)
    q = 2*p*(1-p)
    return 1 - h2(q) + eps + (1 + eps/2.0)*h2(eps/(2.0 + eps))

# --------------------------
# Evaluation on p-grid (step-based)
# --------------------------
def build_p_grid(p_min=0.0, p_max=0.25, p_step=0.01):
    p = np.arange(p_min, p_max + 0.5*p_step, p_step, dtype=float)
    # ensure strictly increasing unique (convex hull assumes increasing)
    p = np.unique(np.clip(p, 0.0, 1.0))
    p.sort()
    if len(p) < 3:
        raise ValueError("Need at least 3 grid points; decrease p_step or widen [p_min,p_max].")
    return p

def evaluate_bounds(p_min=0.0, p_max=0.25, p_step=0.01, compute_eps=False, eps_solver="SCS"):
    """
    Returns:
      p_grid
      bounds dict with arrays for each bound
    """
    p = build_p_grid(p_min, p_max, p_step)

    # Probabilities for the three channels
    pI_ind, pX_ind, pY_ind, pZ_ind = probs_case_independent_sym(p)
    pI_dep, pX_dep, pY_dep, pZ_dep = pauli_probs_depolarizing(p)
    pI_skw, pX_skw, pY_skw, pZ_skw = probs_case_skewed(p)

    bounds = {}

    # EA/2 (all)
    bounds["EA/2_ind"] = ub_entanglement_assisted_half(pI_ind, pX_ind, pY_ind, pZ_ind)
    bounds["EA/2_dep"] = ub_entanglement_assisted_half(pI_dep, pX_dep, pY_dep, pZ_dep)
    bounds["EA/2_skw"] = ub_entanglement_assisted_half(pI_skw, pX_skw, pY_skw, pZ_skw)

    # EB (all): returns 0 when EB triggers, else inf
    bounds["EB_ind"] = ub_entanglement_breaking_cutoff(pI_ind, pX_ind, pY_ind, pZ_ind)
    bounds["EB_dep"] = ub_entanglement_breaking_cutoff(pI_dep, pX_dep, pY_dep, pZ_dep)
    bounds["EB_skw"] = ub_entanglement_breaking_cutoff(pI_skw, pX_skw, pY_skw, pZ_skw)

    # SSC closed-form bounds (where available)
    bounds["SSC_ind"] = ub_bb84_ssc_sym(p)      # independent symmetric
    bounds["SSC_dep"] = ub_depolarizing_ssc(p)  # depolarizing
    bounds["SSC_skw"] = np.full_like(p, np.nan) # not implemented (skewed-specific closed form not in this script)

    # AD bounds (optional; heavy)
    if compute_eps:
        eps_ind = np.array([epsilon_degradable_pauli(*pauli_probs_independent(pi, pi), solver=eps_solver) for pi in p])
        eps_dep = np.array([epsilon_degradable_pauli(*pauli_probs_depolarizing(pi), solver=eps_solver) for pi in p])
        eps_skw = np.array([epsilon_degradable_pauli(*pauli_probs_independent(pi, pi/9), solver=eps_solver) for pi in p])

        bounds["eps_ind"] = eps_ind
        bounds["eps_dep"] = eps_dep
        bounds["eps_skw"] = eps_skw

        bounds["AD_ind"] = ub_bb84_ad_sym(p, eps_ind)
        bounds["AD_dep"] = ub_depolarizing_ad(p, eps_dep)
        bounds["AD_skw"] = np.full_like(p, np.nan)  # not specialized here
    else:
        bounds["AD_ind"] = np.full_like(p, np.nan)
        bounds["AD_dep"] = np.full_like(p, np.nan)
        bounds["AD_skw"] = np.full_like(p, np.nan)

    # Tightest among computed bounds (pointwise min ignoring nan/inf)
    def tightest(*arrs):
        stack = []
        for a in arrs:
            a = np.asarray(a, dtype=float)
            a = np.where(np.isfinite(a), a, np.inf)  # nan/inf won't win
            stack.append(a)
        return np.min(np.vstack(stack), axis=0)

    bounds["tight_ind"] = tightest(bounds["EA/2_ind"], bounds["EB_ind"], bounds["SSC_ind"], bounds["AD_ind"])
    bounds["tight_dep"] = tightest(bounds["EA/2_dep"], bounds["EB_dep"], bounds["SSC_dep"], bounds["AD_dep"])
    bounds["tight_skw"] = tightest(bounds["EA/2_skw"], bounds["EB_skw"])  # add more if you extend

    return p, bounds

# --------------------------
# Plotting
# --------------------------
def _mask_inf_nan(y):
    y = np.asarray(y, dtype=float)
    y = np.where(np.isfinite(y), y, np.nan)
    return y

def plot_bounds(p, B, title_prefix="Upper bounds"):
    """
    Makes 3 figures: independent-symmetric, depolarizing, skewed.
    Each figure plots available bounds and the tightest computed.
    """
    # Helper to plot one channel
    def plot_one(tag, name):
        plt.figure(figsize=(8, 5))
        plt.title(f"{title_prefix}: {name}")
        plt.xlabel("p")
        plt.ylabel("Upper bound on Q (qubits/use)")
        plt.grid(True)

        # EA/2
        plt.plot(p, _mask_inf_nan(B[f"EA/2_{tag}"]), label="EA/2")

        # EB (0 where EB else nan so it appears only when triggers)
        eb = B[f"EB_{tag}"]
        eb_plot = np.where(np.isfinite(eb), eb, np.nan)
        plt.plot(p, eb_plot, label="EB => 0")

        # SSC if available
        if np.any(np.isfinite(_mask_inf_nan(B.get(f"SSC_{tag}", np.nan)))):
            plt.plot(p, _mask_inf_nan(B[f"SSC_{tag}"]), label="SSC (conv envelope)")

        # AD if available
        if np.any(np.isfinite(_mask_inf_nan(B.get(f"AD_{tag}", np.nan)))):
            plt.plot(p, _mask_inf_nan(B[f"AD_{tag}"]), label="AD (via SDP eps)")

        # Tightest computed
        plt.plot(p, _mask_inf_nan(B[f"tight_{tag}"]), linestyle="--", label="tightest among computed")

        plt.ylim(bottom=0.0, top=1.05)
        plt.legend()
        plt.tight_layout()

    plot_one("ind", "Independent (px=pz=p)")
    plot_one("dep", "Depolarizing (p)")
    plot_one("skw", "Skewed independent (px=p, pz=p/9)")
    plt.show()

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # Change these as needed:
    p_min = 0.0
    p_max = 0.25
    p_step = 0.01

    # Turn on SDP-based AD bounds (slow; requires cvxpy):
    compute_eps = False
    eps_solver = "SCS"  # try "SCS" first

    p_grid, bounds = evaluate_bounds(
        p_min=p_min, p_max=p_max, p_step=p_step,
        compute_eps=compute_eps, eps_solver=eps_solver
    )

    plot_bounds(p_grid, bounds, title_prefix="Upper bounds (coarse grid)")
