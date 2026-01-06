# upper_bounds.py
# Bounds included:
#  - EA/2 upper bound (all channels)
#  - Entanglement-breaking (EB) => Q=0 test (all channels; returns 0 or inf)
#  - SSC analytic bounds:
#       * BB84 / independent symmetric px=pz=p (closed form)
#       * depolarizing (convex envelope)
#  - Theorem 3.4 (Approx. degradable) bound for SKEWED ONLY (optional; uses SDP epsilon via cvxpy)
# Plotting:
#  - Saves each plot as a PNG (default into ./plots/)
#  - Uses non-GUI backend by default (works on headless systems)

import os
import numpy as np
import math
import matplotlib

matplotlib.use("Agg")  # non-GUI backend (saves PNGs fine)
import matplotlib.pyplot as plt


# --------------------------
# Basic entropy utilities
# --------------------------
def h2(x):
    """Binary entropy h2(x) in bits. Vectorized."""
    x = np.asarray(x, dtype=float)
    eps = 1e-300
    x = np.clip(x, eps, 1 - eps)
    return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))


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


def probs_case_skewed(p, a):
    """Skewed independent: px=p, pz=a*p."""
    return pauli_probs_independent(p, a * p)


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
    For a qubit Pauli channel N(ρ)=Σ_P p_P P ρ P,
    in the Pauli operator basis {I,X,Y,Z}, it acts diagonally with eigenvalues:
      λX = pI + pX - pY - pZ
      λY = pI - pX + pY - pZ
      λZ = pI - pX - pY + pZ
    """
    pI = np.asarray(pI, dtype=float)
    pX = np.asarray(pX, dtype=float)
    pY = np.asarray(pY, dtype=float)
    pZ = np.asarray(pZ, dtype=float)
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
    x = np.asarray(x)
    y = np.asarray(y)
    xq = np.asarray(xq)
    hx = x[hull_idx]
    hy = y[hull_idx]

    seg = np.searchsorted(hx, xq, side="right") - 1
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
# SSC bounds (analytic forms)
# --------------------------
def ub_bb84_ssc_sym(p_grid):
    """
    BB84 / independent symmetric (px=pz=p) upper bound (Smith–Smolin / Sutter et al.):

        Q(B_{p,p}) <= h(1/2 - 2p(1-p)) - h(2p(1-p)).

    No convex-envelope is needed here because the cited bound is a single explicit expression.
    """
    p = np.asarray(p_grid, dtype=float)
    q = 2 * p * (1 - p)
    return h2(0.5 - q) - h2(q)


def gamma_depolarizing(p):
    """
    Gamma function used in depolarizing SSC convex-envelope form:

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
# Theorem 3.4 (Approx. degradable) bound
# --------------------------
def q1_bb84(px, pz):
    """
    Single-letter coherent info for BB84/independent channel B_{px,pz} as used in the paper:
      Q^(1)(B_{px,pz}) = 1 - h(px) - h(pz).
    """
    px = np.asarray(px, dtype=float)
    pz = np.asarray(pz, dtype=float)
    return 1.0 - h2(px) - h2(pz)


def ub_theorem34_from_q1(q1, eps, E_dim):
    """
    Theorem 3.4 (i) upper bound:
      Q(Φ) <= Q^(1)(Φ)
              + (ε/2) log2(|E|-1) + h2(ε/2)
              + ε log2 |E| + (1+ε/2) h2( ε/(2+ε) ).

    Here |E| is the dimension of the environment in a Stinespring dilation.
    For our Pauli Kraus representation with 4 Kraus ops, we take |E|=4.
    """
    eps = np.asarray(eps, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    term = (
        (eps / 2.0) * np.log2(max(E_dim - 1, 1))
        + h2(eps / 2.0)
        + eps * np.log2(E_dim)
        + (1.0 + eps / 2.0) * h2(eps / (2.0 + eps))
    )
    return q1 + term


# --------------------------
# Approximate degradability via SDP (needed for skewed Thm 3.4 plot)
# --------------------------
def _try_import_cvxpy():
    try:
        import cvxpy as cp  # noqa: F401
        return True
    except Exception:
        return False


def perm_matrix_for_choi_to_transfer(d_in, d_out):
    """
    Permutation P s.t. vec(T) = P vec(J), where:
      J indices (i,k ; j,l) with i,j in out and k,l in in
      T indices (i,j ; k,l)
    """
    N = d_out * d_in
    P = np.zeros((d_out * d_out * d_in * d_in, d_out * d_in * d_out * d_in), dtype=float)
    for i in range(d_out):
        for k in range(d_in):
            for j in range(d_out):
                for l in range(d_in):
                    rowJ = i * d_in + k
                    colJ = j * d_in + l
                    idxJ = rowJ + N * colJ

                    rowT = i * d_out + j
                    colT = k * d_in + l
                    idxT = rowT + (d_out * d_out) * colT
                    P[idxT, idxJ] = 1.0
    return P


def choi_from_kraus(kraus, d_in=2, d_out=2):
    """Choi matrix via J = sum_i vec(K_i) vec(K_i)†."""
    J = np.zeros((d_out * d_in, d_out * d_in), dtype=complex)
    for K in kraus:
        v = K.reshape(-1, order="F")  # column-major vec
        J += np.outer(v, np.conjugate(v))
    return J


def kraus_pauli_channel(pI, pX, pY, pZ):
    """Kraus ops for Pauli channel."""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [
        math.sqrt(max(pI, 0.0)) * I,
        math.sqrt(max(pX, 0.0)) * X,
        math.sqrt(max(pY, 0.0)) * Y,
        math.sqrt(max(pZ, 0.0)) * Z,
    ]


def choi_of_complement_from_kraus(kraus):
    """
    Build Choi of complementary map Φ^c from Kraus {K_i}.
    Environment dimension r = number of Kraus ops.
    Output dimension of complement is r.
    """
    d_in = kraus[0].shape[1]
    r = len(kraus)
    J = np.zeros((r * d_in, r * d_in), dtype=complex)

    for a in range(d_in):
        for b in range(d_in):
            M = np.zeros((r, r), dtype=complex)
            for i in range(r):
                for j in range(r):
                    M[j, i] = np.vdot(kraus[i][:, b], kraus[j][:, a])
            for j in range(r):
                for i in range(r):
                    J[j * d_in + a, i * d_in + b] = M[j, i]
    return J


def partial_trace_out_cvxpy(X, d_out, d_in):
    """
    Tr_out over the 'output' subsystem of dims (d_out x d_in) matrix.
    Here X is (d_out*d_in) x (d_out*d_in) Hermitian (cvxpy variable/expression).
    Returns (d_in x d_in) expression.
    """
    import cvxpy as cp

    blocks = []
    for i in range(d_out):
        blocks.append(X[i * d_in : (i + 1) * d_in, i * d_in : (i + 1) * d_in])
    return cp.sum(blocks)


def epsilon_degradable_pauli(pI, pX, pY, pZ, solver="SCS", verbose=False):
    """
    Compute ε = min_{degrading Ξ} || Φ^c - Ξ ∘ Φ ||_diamond via SDP (heavy).
    Returns SDP optimum value (float). Needs cvxpy installed.
    """
    import cvxpy as cp

    kraus = kraus_pauli_channel(pI, pX, pY, pZ)
    J_Phi = choi_from_kraus(kraus, d_in=2, d_out=2)  # (4x4)
    J_Phic = choi_of_complement_from_kraus(kraus)    # (8x8) since r=4

    dA = 2
    dB = 2
    dE = 4  # environment dim = number of Kraus ops

    J_Xi = cp.Variable((dE * dB, dE * dB), complex=True, hermitian=True)

    constraints = [J_Xi >> 0]
    TrE_JXi = partial_trace_out_cvxpy(J_Xi, d_out=dE, d_in=dB)
    constraints += [TrE_JXi == np.eye(dB)]

    P_BE = perm_matrix_for_choi_to_transfer(d_in=dB, d_out=dE)
    P_AB = perm_matrix_for_choi_to_transfer(d_in=dA, d_out=dB)

    T_Xi = P_BE @ cp.vec(J_Xi)
    T_Xi = cp.reshape(T_Xi, (dE * dE, dB * dB))

    T_Phi = P_AB @ J_Phi.reshape(-1, order="F")
    T_Phi = T_Phi.reshape((dB * dB, dA * dA), order="F")

    T_comp = T_Xi @ T_Phi

    P_AE = perm_matrix_for_choi_to_transfer(d_in=dA, d_out=dE)
    vecJ_comp = P_AE.T @ cp.vec(T_comp)
    J_comp = cp.reshape(vecJ_comp, (dE * dA, dE * dA))

    J_Delta = J_Phic - J_comp
    Zvar = cp.Variable((dE * dA, dE * dA), complex=True, hermitian=True)
    mu = cp.Variable(nonneg=True)

    constraints += [
        Zvar >> 0,
        Zvar - J_Delta >> 0,
        partial_trace_out_cvxpy(Zvar, d_out=dE, d_in=dA) << mu * np.eye(dA),
    ]

    prob = cp.Problem(cp.Minimize(2 * mu), constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"epsilon SDP failed: status={prob.status}")
    return float(prob.value)


# --------------------------
# Evaluation on p-grid (step-based)
# --------------------------
def build_p_grid(p_min=0.0, p_max=0.25, p_step=0.01):
    p = np.arange(p_min, p_max + 0.5 * p_step, p_step, dtype=float)
    p = np.unique(np.clip(p, 0.0, 1.0))
    p.sort()
    if len(p) < 3:
        raise ValueError("Need at least 3 grid points; decrease p_step or widen [p_min,p_max].")
    return p


def evaluate_bounds(
    p_min=0.0,
    p_max=0.25,
    p_step=0.01,
    compute_thm34_skw=False,
    eps_solver="SCS",
    skw_a=1.0 / 9.0,
):
    """
    Returns:
      p_grid
      bounds dict with arrays for each bound (for ind/dep/skw evaluated on same p_grid)
    """
    p = build_p_grid(p_min, p_max, p_step)

    pI_ind, pX_ind, pY_ind, pZ_ind = probs_case_independent_sym(p)
    pI_dep, pX_dep, pY_dep, pZ_dep = pauli_probs_depolarizing(p)
    pI_skw, pX_skw, pY_skw, pZ_skw = probs_case_skewed(p, skw_a)

    bounds = {}

    bounds["EA/2_ind"] = ub_entanglement_assisted_half(pI_ind, pX_ind, pY_ind, pZ_ind)
    bounds["EA/2_dep"] = ub_entanglement_assisted_half(pI_dep, pX_dep, pY_dep, pZ_dep)
    bounds["EA/2_skw"] = ub_entanglement_assisted_half(pI_skw, pX_skw, pY_skw, pZ_skw)

    bounds["EB_ind"] = ub_entanglement_breaking_cutoff(pI_ind, pX_ind, pY_ind, pZ_ind)
    bounds["EB_dep"] = ub_entanglement_breaking_cutoff(pI_dep, pX_dep, pY_dep, pZ_dep)
    bounds["EB_skw"] = ub_entanglement_breaking_cutoff(pI_skw, pX_skw, pY_skw, pZ_skw)

    bounds["SSC_ind"] = ub_bb84_ssc_sym(p)
    bounds["SSC_dep"] = ub_depolarizing_ssc(p)
    bounds["SSC_skw"] = np.full_like(p, np.nan)

    if compute_thm34_skw:
        if not _try_import_cvxpy():
            raise RuntimeError("compute_thm34_skw=True but cvxpy is not installed.")

        eps_skw = np.array(
            [epsilon_degradable_pauli(*pauli_probs_independent(pi, skw_a * pi), solver=eps_solver) for pi in p]
        )
        bounds["eps_skw"] = eps_skw

        q1_skw = q1_bb84(p, skw_a * p)
        bounds["Q1_skw"] = q1_skw

        bounds["Thm34_skw"] = ub_theorem34_from_q1(q1_skw, eps_skw, E_dim=4)
    else:
        bounds["eps_skw"] = np.full_like(p, np.nan)
        bounds["Q1_skw"] = np.full_like(p, np.nan)
        bounds["Thm34_skw"] = np.full_like(p, np.nan)

    def tightest(*arrs):
        stack = []
        for a in arrs:
            a = np.asarray(a, dtype=float)
            a = np.where(np.isfinite(a), a, np.inf)
            stack.append(a)
        return np.min(np.vstack(stack), axis=0)

    bounds["tight_ind"] = tightest(bounds["EA/2_ind"], bounds["EB_ind"], bounds["SSC_ind"])
    bounds["tight_dep"] = tightest(bounds["EA/2_dep"], bounds["EB_dep"], bounds["SSC_dep"])
    bounds["tight_skw"] = tightest(bounds["EA/2_skw"], bounds["EB_skw"], bounds["Thm34_skw"])

    return p, bounds


# --------------------------
# Plotting + saving
# --------------------------
def _mask_inf_nan(y):
    y = np.asarray(y, dtype=float)
    return np.where(np.isfinite(y), y, np.nan)


def print_ssc_table(channel_key, p, bounds):
    """Print simple copy/paste table for SSC only."""
    key = f"SSC_{channel_key}"
    y = bounds.get(key, None)
    if y is None:
        print(f"\n# SSC table: {channel_key} (no data)\n")
        return
    print(f"\n# SSC table: {channel_key}")
    print("p\tSSC")
    for pi, yi in zip(p, y):
        if np.isfinite(yi):
            print(f"{pi:.10g}\t{yi:.16g}")
        else:
            print(f"{pi:.10g}\tnan")


def print_thm34_table_skewed(p, bounds):
    """Print simple copy/paste table for Theorem-3.4 bound for the skewed channel."""
    y = bounds.get("Thm34_skw", None)
    if y is None:
        print("\n# Thm 3.4 table: skw (no data)\n")
        return
    print("\n# Thm 3.4 table: skw")
    print("p\tThm34")
    for pi, yi in zip(p, y):
        if np.isfinite(yi):
            print(f"{pi:.10g}\t{yi:.16g}")
        else:
            print(f"{pi:.10g}\tnan")


def plot_one_channel(tag, title, p, bounds, xticks=None, xlim=None, out_dir="plots"):
    """
    Plot bounds for one channel tag in {"ind","dep","skw"} on provided p-grid.
    Y-axis: [min(SSC)-0.1, 1.0] (clipped at 0 if needed).
    """
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(8, 5))

    EA = _mask_inf_nan(bounds[f"EA/2_{tag}"])
    EB = _mask_inf_nan(bounds[f"EB_{tag}"])
    SSC = _mask_inf_nan(bounds.get(f"SSC_{tag}", np.full_like(p, np.nan)))
    tight = _mask_inf_nan(bounds[f"tight_{tag}"])

    plt.plot(p, EA, label="EA/2", linewidth=2)
    plt.plot(p, EB, label="EB (0 if EB else inf)", linewidth=2)
    plt.plot(p, SSC, label="SSC (analytic)", linewidth=2)

    if tag == "skw":
        Thm34 = _mask_inf_nan(bounds.get("Thm34_skw", np.full_like(p, np.nan)))
        Q1 = _mask_inf_nan(bounds.get("Q1_skw", np.full_like(p, np.nan)))
        if np.any(np.isfinite(Q1)):
            plt.plot(p, Q1, label="Q^(1) (skewed)", linewidth=2)
        if np.any(np.isfinite(Thm34)):
            plt.plot(p, Thm34, label="Thm 3.4 (SDP eps)", linewidth=2)

    plt.plot(p, tight, label="tightest (among computed)", linewidth=2, linestyle="--")

    plt.title(title)
    plt.xlabel("p")
    plt.ylabel("Upper bound")

    if xlim is not None:
        plt.xlim(xlim)
    if xticks is not None:
        plt.xticks(xticks)

    if np.any(np.isfinite(SSC)):
        y0 = float(np.nanmin(SSC)) - 0.1
        y0 = max(0.0, y0)
    else:
        y0 = 0.0
    plt.ylim((y0, 1.0))

    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    fname = os.path.join(out_dir, f"upper_bounds_{tag}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close(fig)
    return fname


def run_all(channel_cfg, compute_thm34_skw=True, eps_solver="SCS", out_dir="plots"):
    """
    For each channel config entry:
      - evaluate bounds on that channel's p-range
      - print SSC table (p, SSC) [except skewed: print Thm 3.4 table]
      - save plot PNG with requested axis/ticks
    """
    saved = {}

    for tag, cfg in channel_cfg.items():
        do_sdp = bool(compute_thm34_skw) and (tag == "skw")
        skw_a = cfg.get("a", 1.0 / 9.0) if tag == "skw" else (1.0 / 9.0)

        p, bounds = evaluate_bounds(
            p_min=cfg["p_min"],
            p_max=cfg["p_max"],
            p_step=cfg["p_step"],
            compute_thm34_skw=do_sdp,
            eps_solver=eps_solver,
            skw_a=skw_a,
        )

        if tag == "skw":
            print_thm34_table_skewed(p, bounds)
        else:
            print_ssc_table(tag, p, bounds)

        fname = plot_one_channel(
            tag=tag,
            title=cfg["title"],
            p=p,
            bounds=bounds,
            xticks=cfg.get("xticks", None),
            xlim=(cfg["p_min"], cfg["p_max"]),
            out_dir=out_dir,
        )
        saved[tag] = fname

    print("\nSaved plots:")
    for tag, fn in saved.items():
        print(f"  {tag}: {fn}")


if __name__ == "__main__":
    # Per-channel plot ranges + ticks (change these as you like)
    CHANNEL_CFG = {
        "ind": {
            "title": "Independent symmetric (px=pz=p): bounds vs p",
            "p_min": 0.0,
            "p_max": 0.01,
            "p_step": 0.0005,
            "xticks": [0.0, 0.0025, 0.005, 0.0075, 0.01],
        },
        "dep": {
            "title": "Depolarizing: bounds vs p",
            "p_min": 0.0,
            "p_max": 0.04,
            "p_step": 0.002,
            "xticks": [0.0, 0.01, 0.02, 0.03, 0.04],
        },
        "skw": {
            # Set 'a' here:
            #   a = 100.0   => pz = 100 * px   (paper example)
            #   a = 1/9     => pz = (1/9) * px (your example)
            "a": 100,
            "title": "Skewed independent (px=p, pz=a*p): bounds vs p",
            "p_min": 0.0,
            "p_max": 0.001,
            "p_step": 0.00005,
            "xticks": [0.0, 0.00025, 0.0005, 0.00075, 0.001],
        },
    }

    run_all(CHANNEL_CFG, compute_thm34_skw=True, eps_solver="SCS", out_dir="plots")

