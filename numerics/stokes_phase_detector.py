#!/usr/bin/env python3
"""
Stokes-Guided Phase Transition Detector
========================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: Paper Psi v3 — practical diagnostic tool

Validates the Stokes concentration theorem as a phase transition detector
on standard benchmark models:
  1. 2D Ising strip (width L = 4, 6, 8, 10)
  2. q-state Potts strip (q = 2, 3, 4, 5)

Key idea: Fisher zeros of Z_N = Σ λ_k^N concentrate on the Stokes network
S = {Re φ_p = Re φ_q} where φ_k = log λ_k. The Stokes network approaching
the real axis signals a phase transition (spectral gap closing).

Figures saved to hsmi-yang-mills/papers/:
  fig_stokes_ising_L8.png        — 2D Stokes map for Ising L=8
  fig_phase_detection.png        — β_c(L) extrapolation + Potts β_c vs q
  fig_potts_stokes_comparison.png — 2×2 Potts Stokes maps
  fig_stokes_concentration.png   — log-log dist vs N confirming O(1/N)
"""

import numpy as np
import sys
import time

t0 = time.time()

PAPER_DIR = '/home/golbryk/ai/toe/hsmi-yang-mills/papers'


# ===================================================================
# Transfer matrix builders
# ===================================================================

def precompute_ising_strip(L):
    """Precompute integer energy arrays for Ising strip of width L.

    Returns (intra, inter) where intra is (2^L,) and inter is (2^L, 2^L).
    T(β) = diag(exp(β/2 · intra)) · exp(β · inter) · diag(exp(β/2 · intra))
    """
    dim = 2**L
    configs = np.array([[(s >> i) & 1 for i in range(L)] for s in range(dim)])
    spins = 2 * configs - 1

    intra = np.zeros(dim)
    for i in range(L):
        intra += spins[:, i] * spins[:, (i + 1) % L]

    inter = (spins @ spins.T).astype(float)
    return intra, inter


def precompute_potts_strip(L, q):
    """Precompute integer energy arrays for q-state Potts strip of width L.

    Returns (intra, inter).
    """
    dim = q**L
    configs = np.zeros((dim, L), dtype=int)
    for s in range(dim):
        val = s
        for i in range(L):
            configs[s, i] = val % q
            val //= q

    intra = np.zeros(dim)
    for i in range(L):
        intra += (configs[:, i] == configs[:, (i + 1) % L]).astype(float)

    inter = np.zeros((dim, dim))
    for i in range(L):
        inter += (configs[:, i, None] == configs[None, :, i]).astype(float)

    return intra, inter


def build_transfer_matrix(beta, intra, inter):
    """Build T(β) from precomputed energy arrays. Fast."""
    sqrt_V = np.exp(0.5 * beta * intra)
    W = np.exp(beta * inter)
    return sqrt_V[:, None] * W * sqrt_V[None, :]


# ===================================================================
# Eigenvalue computation — vectorized over y for fixed β_r
# ===================================================================

def get_top_eigenvalues(T, k):
    """Get top k eigenvalues sorted by magnitude."""
    evals = np.linalg.eigvals(T)
    idx = np.argsort(-np.abs(evals))
    return evals[idx[:k]]


def match_eigenvalues(prev, curr, k):
    """Match current eigenvalues to previous by complex-plane distance."""
    matched = np.zeros(k, dtype=complex)
    used = set()
    for j in range(k):
        dists = np.abs(curr - prev[j])
        order = np.argsort(dists)
        for idx in order:
            if idx not in used:
                matched[j] = curr[idx]
                used.add(idx)
                break
    return matched


def compute_eigenvalues_on_grid(intra, inter, beta_r_arr, y_arr, k=6):
    """Compute top k eigenvalue magnitudes on (β_r, y) grid.

    Uses precomputed (intra, inter) arrays for speed.
    Eigenvalues are TRACKED from y=y_arr[0] by continuity matching,
    so branch 0 may not be the largest at all points.

    Returns eig_abs array of shape (n_br, n_y, k).
    """
    n_br = len(beta_r_arr)
    n_y = len(y_arr)
    dim = len(intra)
    k = min(k, dim)
    eig_abs = np.zeros((n_br, n_y, k))

    for ib in range(n_br):
        prev = None
        for iy in range(n_y):
            beta = beta_r_arr[ib] + 1j * y_arr[iy]
            T = build_transfer_matrix(beta, intra, inter)
            if not np.all(np.isfinite(T)):
                eig_abs[ib, iy, :] = 0.0
                prev = None
                continue
            try:
                top = get_top_eigenvalues(T, k)
            except np.linalg.LinAlgError:
                eig_abs[ib, iy, :] = 0.0
                prev = None
                continue

            if prev is not None:
                top = match_eigenvalues(prev, top, k)

            eig_abs[ib, iy, :] = np.abs(top)
            prev = top.copy()

    return eig_abs


# ===================================================================
# Stokes network detection
# ===================================================================

def find_stokes_network(eig_abs, beta_r_arr, y_arr):
    """Find Stokes boundary: |λ_0| = |λ_1| via sign-change interpolation.

    Returns list of (β_r, y, 0, 1) boundary points.
    """
    n_br, n_y, _ = eig_abs.shape
    points = []

    for ib in range(n_br):
        diff = eig_abs[ib, :, 0] - eig_abs[ib, :, 1]
        for iy in range(n_y - 1):
            if diff[iy] * diff[iy + 1] < 0:
                frac = abs(diff[iy]) / (abs(diff[iy]) + abs(diff[iy + 1]))
                y_cross = y_arr[iy] + frac * (y_arr[iy + 1] - y_arr[iy])
                points.append((beta_r_arr[ib], y_cross, 0, 1))

    for iy in range(n_y):
        diff = eig_abs[:, iy, 0] - eig_abs[:, iy, 1]
        for ib in range(n_br - 1):
            if diff[ib] * diff[ib + 1] < 0:
                frac = abs(diff[ib]) / (abs(diff[ib]) + abs(diff[ib + 1]))
                br_cross = beta_r_arr[ib] + frac * (beta_r_arr[ib + 1] - beta_r_arr[ib])
                points.append((br_cross, y_arr[iy], 0, 1))

    return points


def find_stokes_nearest_real_axis(stokes_points):
    """Find where the Stokes network is closest to the real axis (y=0).

    The β_r at which min_y{|λ_0|=|λ_1|} is smallest gives the
    Stokes-predicted pseudo-critical coupling.

    Returns (beta_c, min_y) or (None, None) if no points.
    """
    if not stokes_points:
        return None, None

    # Find point with smallest |y|
    best = min(stokes_points, key=lambda p: abs(p[1]))
    return best[0], best[1]


# ===================================================================
# Fisher zero finding — log-space to avoid overflow
# ===================================================================

def compute_ZN_safe(intra, inter, beta, N, k_eig=6):
    """Compute Z_N = Σ_k λ_k^N normalized by λ_0^N to avoid overflow.

    Uses precomputed (intra, inter).
    Returns Z_N / λ_0^N (zeros are the same).
    """
    try:
        T = build_transfer_matrix(beta, intra, inter)
        if not np.all(np.isfinite(T)):
            return np.inf + 0j
        evals = np.linalg.eigvals(T)
    except (np.linalg.LinAlgError, FloatingPointError):
        return np.inf + 0j

    idx = np.argsort(-np.abs(evals))
    k_eig = min(k_eig, len(evals))
    top = evals[idx[:k_eig]]

    lam0 = top[0]
    if abs(lam0) < 1e-300:
        return 0.0 + 0j

    ratios = top / lam0
    return np.sum(ratios**N)


def find_fisher_zeros_grid(intra, inter, beta_r_arr, y_arr, N_values, k_eig=6):
    """Find Fisher zeros via |Z_N| minima + Newton refinement.

    Uses precomputed (intra, inter).
    Returns dict[N → list of complex beta].
    """
    result = {}
    for N in N_values:
        n_br, n_y = len(beta_r_arr), len(y_arr)
        absZ = np.full((n_br, n_y), np.inf)
        for ib in range(n_br):
            for iy in range(n_y):
                beta = beta_r_arr[ib] + 1j * y_arr[iy]
                val = abs(compute_ZN_safe(intra, inter, beta, N, k_eig))
                if np.isfinite(val):
                    absZ[ib, iy] = val

        # Find local minima
        candidates = []
        for ib in range(1, n_br - 1):
            for iy in range(1, n_y - 1):
                val = absZ[ib, iy]
                if (val < absZ[ib - 1, iy] and val < absZ[ib + 1, iy]
                        and val < absZ[ib, iy - 1] and val < absZ[ib, iy + 1]):
                    candidates.append((beta_r_arr[ib], y_arr[iy], val))

        candidates.sort(key=lambda x: x[2])

        zeros = []
        for br0, y0, _ in candidates[:30]:
            beta, converged = newton_fisher_zero(
                intra, inter, br0, y0, N, k_eig)
            if converged:
                is_dup = any(abs(beta - z) < 1e-4 for z in zeros)
                if not is_dup:
                    zeros.append(beta)

        result[N] = sorted(zeros, key=lambda z: z.imag)

    return result


def newton_fisher_zero(intra, inter, br0, y0, N, k_eig=6,
                       tol=1e-10, max_iter=30, eps=1e-7):
    """Newton's method for Z_N(β) = 0 in the complex β plane.

    Returns (beta, converged).
    """
    beta = br0 + 1j * y0

    for _ in range(max_iter):
        Z0 = compute_ZN_safe(intra, inter, beta, N, k_eig)
        if not np.isfinite(abs(Z0)):
            return beta, False
        if abs(Z0) < tol:
            return beta, True

        dZ = (compute_ZN_safe(intra, inter, beta + eps, N, k_eig) - Z0) / eps
        if not np.isfinite(abs(dZ)) or abs(dZ) < 1e-30:
            return beta, False

        step = -Z0 / dZ

        # Damped step with bounds
        alpha = 1.0
        moved = False
        for _ in range(8):
            beta_new = beta + alpha * step
            if (0.01 < beta_new.real < 5.0 and 0.01 < beta_new.imag < 10.0):
                Z_new = compute_ZN_safe(intra, inter, beta_new, N, k_eig)
                if np.isfinite(abs(Z_new)) and abs(Z_new) < abs(Z0):
                    beta = beta_new
                    moved = True
                    break
            alpha *= 0.5
        if not moved:
            return beta, False

    Z_final = compute_ZN_safe(intra, inter, beta, N, k_eig)
    return beta, np.isfinite(abs(Z_final)) and abs(Z_final) < tol * 100


# ===================================================================
# Verification utilities
# ===================================================================

def verify_zeros_on_stokes(zeros, stokes_points):
    """Measure distance from each Fisher zero to nearest Stokes point.

    Returns (distances, fraction_within_quarter_spacing).
    """
    if not zeros or not stokes_points:
        return np.array([]), 0.0

    stokes_arr = np.array([(s[0], s[1]) for s in stokes_points])
    distances = []
    for z in zeros:
        zpt = np.array([z.real, z.imag])
        d = np.min(np.sqrt(np.sum((stokes_arr - zpt)**2, axis=1)))
        distances.append(d)

    distances = np.array(distances)
    if len(zeros) >= 2:
        spacings = np.abs(np.diff(
            [z.imag for z in sorted(zeros, key=lambda z: z.imag)]))
        avg_spacing = np.mean(spacings) if len(spacings) > 0 else 1.0
        frac = np.mean(distances < avg_spacing / 4)
    else:
        frac = 0.0

    return distances, frac


def finite_size_scaling(L_values, beta_c_estimates):
    """Fit β_c(L) = β_c(∞) + a/L.

    Returns (beta_c_inf, a, residual).
    """
    Ls = np.array(L_values, dtype=float)
    betas = np.array(beta_c_estimates)
    x = 1.0 / Ls
    A = np.column_stack([np.ones_like(x), x])
    coeffs, _, _, _ = np.linalg.lstsq(A, betas, rcond=None)
    residual = np.sqrt(np.mean((betas - A @ coeffs)**2))
    return coeffs[0], coeffs[1], residual


def stokes_gap_at_point(intra, inter, beta, k_eig=6):
    """Continuous Stokes gap: |log|λ_0| - log|λ_1|| at β.

    Measures proximity to Stokes curve |λ_0|=|λ_1| EXACTLY at the given point.
    No grid-resolution floor — this is the correct continuous measure.
    Returns 0 on Stokes, > 0 otherwise.
    """
    try:
        T = build_transfer_matrix(beta, intra, inter)
        if not np.all(np.isfinite(T)):
            return np.inf
        evals = np.linalg.eigvals(T)
    except (np.linalg.LinAlgError, FloatingPointError):
        return np.inf
    mags = np.sort(np.abs(evals))[::-1]
    if len(mags) < 2 or mags[0] < 1e-300 or mags[1] < 1e-300:
        return np.inf
    return abs(np.log(mags[0]) - np.log(mags[1]))


def find_zeros_from_stokes(intra, inter, stokes_points, N_values,
                           k_eig=6, max_seeds=100):
    """Find Fisher zeros using Stokes points as Newton seeds at each N.

    Independent Newton search from Stokes seeds for every N value.
    No tracking → no survivor bias. Stokes seeds are optimal starting
    points since zeros concentrate on the Stokes network.

    Returns dict[N → list of (complex_beta, stokes_gap)].
    """
    # Thin Stokes points to ~max_seeds, spread evenly by y
    pts = sorted(stokes_points, key=lambda p: p[1])
    if len(pts) > max_seeds:
        step = max(1, len(pts) // max_seeds)
        pts = pts[::step]

    result = {}
    for N in N_values:
        zeros_gaps = []
        seen = []
        for br, y, _, _ in pts:
            beta, converged = newton_fisher_zero(
                intra, inter, br, y, N, k_eig, tol=1e-12, max_iter=50)
            if converged:
                is_dup = any(abs(beta - z) < 1e-4 for z in seen)
                if not is_dup:
                    gap = stokes_gap_at_point(intra, inter, beta, k_eig)
                    zeros_gaps.append((beta, gap))
                    seen.append(beta)
        result[N] = zeros_gaps

    return result


# ===================================================================
print("=" * 90)
print("  Stokes-Guided Phase Transition Detector")
print("  " + "=" * 86)
print(f"  Models: 2D Ising strip (L=4,6,8), q-state Potts (q=2,3,4,5)")
print(f"  Method: Transfer matrix eigenvalues → Stokes network → β_c detection")
print("=" * 90)


# ===================================================================
# PART 1: 2D Ising — Stokes network and β_c detection
# ===================================================================

print(f"\n  PART 1: 2D Ising Strip — Stokes Network")
print("  " + "-" * 70)

beta_c_exact_ising = 0.5 * np.log(1 + np.sqrt(2))
print(f"  Exact: β_c = (1/2)ln(1+√2) = {beta_c_exact_ising:.6f}")
print(f"  Method: β_c(L) = β_r where Stokes network is closest to y=0")

ising_L_values = [4, 6, 8]
ising_beta_c = {}
ising_eig_data = {}
ising_stokes_data = {}
ising_zero_data = {}
ising_precomp = {}
ising_gap_data = {}

for L in ising_L_values:
    dim = 2**L
    print(f"\n  --- L = {L} (matrix {dim}×{dim}) ---")
    t1 = time.time()

    # Precompute energy arrays once
    intra, inter = precompute_ising_strip(L)
    ising_precomp[L] = (intra, inter)

    # Adaptive grid sizes (256×256 eigvals ~300ms each on this machine)
    if dim <= 64:
        n_grid = 60
    elif dim <= 256:
        n_grid = 20
    else:
        n_grid = 12

    beta_r_arr = np.linspace(0.1, 1.0, n_grid)
    y_arr = np.linspace(0.01, 2.5, n_grid)

    # Step 1: Eigenvalues on 2D grid (with tracking)
    print(f"    Computing eigenvalues on {n_grid}×{n_grid} grid...",
          end=" ", flush=True)
    sys.stdout.flush()
    k_eig = min(6, dim)
    eig_abs = compute_eigenvalues_on_grid(
        intra, inter, beta_r_arr, y_arr, k=k_eig)
    ising_eig_data[L] = (eig_abs, beta_r_arr, y_arr)
    print(f"Done ({time.time()-t1:.1f}s)")

    # Step 2: Stokes network
    stokes_pts = find_stokes_network(eig_abs, beta_r_arr, y_arr)
    ising_stokes_data[L] = stokes_pts
    print(f"    Stokes boundary points: {len(stokes_pts)}")

    # Step 3: β_c from Stokes proximity to real axis
    beta_c_L, min_y = find_stokes_nearest_real_axis(stokes_pts)
    if beta_c_L is not None:
        ising_beta_c[L] = beta_c_L
        print(f"    β_c(L={L}) = {beta_c_L:.6f} (min_y = {min_y:.4f}, "
              f"err = {abs(beta_c_L - beta_c_exact_ising):.6f})")
    else:
        print(f"    No Stokes points found for β_c detection")

    # Step 4: Fisher zeros from Stokes seeds (independent at each N)
    if dim <= 64 and stokes_pts:
        N_values = [10, 15, 20, 30, 50, 75, 100, 150, 200]

        print(f"    Finding zeros from {len(stokes_pts)} Stokes seeds...",
              end=" ", flush=True)
        sys.stdout.flush()
        result = find_zeros_from_stokes(
            intra, inter, stokes_pts, N_values, k_eig=k_eig)

        ising_zero_data[L] = {N: [z for z, _ in zg]
                               for N, zg in result.items()}
        ising_gap_data[L] = result

        for N in N_values:
            zg = result.get(N, [])
            nz = len(zg)
            if nz > 0:
                gaps = [g for _, g in zg if np.isfinite(g)]
                avg_gap = np.mean(gaps) if gaps else float('nan')
                print(f"\n      N={N:4d}: {nz:3d} zeros, "
                      f"⟨gap⟩={avg_gap:.2e}", end="")
        print()
    else:
        ising_zero_data[L] = {}
        ising_gap_data[L] = {}
        if dim > 64:
            print(f"    Skipping Fisher zeros (eigvals too slow for {dim}×{dim})")
        else:
            print(f"    No Stokes points for zero-finding")
    print(f"    [{time.time()-t1:.1f}s total]")


# ===================================================================
# PART 2: q-state Potts — Stokes network
# ===================================================================

print(f"\n\n  PART 2: q-state Potts Strip — Stokes Network")
print("  " + "-" * 70)

potts_configs = [
    (2, 4),
    (3, 4),
    (4, 4),
    (5, 4),
]

potts_beta_c = {}
potts_eig_data = {}
potts_stokes_data = {}
potts_zero_data = {}
potts_precomp = {}
potts_gap_data = {}

for q, L in potts_configs:
    beta_c_exact = np.log(1 + np.sqrt(q))
    dim = q**L
    print(f"\n  --- q={q}, L={L} (matrix {dim}×{dim}), "
          f"β_c(exact) = {beta_c_exact:.6f} ---")

    if dim > 100:
        print(f"    Skipping (dim={dim}, eigvals ~{dim**3/1e6*0.3/256:.0f}ms each)")
        continue

    t1 = time.time()

    # Precompute energy arrays
    intra_p, inter_p = precompute_potts_strip(L, q)
    potts_precomp[(q, L)] = (intra_p, inter_p)

    if dim <= 16:
        n_grid = 50
    else:
        n_grid = 40

    br_lo = max(0.1, beta_c_exact - 0.5)
    br_hi = beta_c_exact + 0.5
    beta_r_arr = np.linspace(br_lo, br_hi, n_grid)
    y_arr = np.linspace(0.01, 2.5, n_grid)

    # 2D eigenvalue grid
    print(f"    Computing eigenvalues on {n_grid}×{n_grid} grid...",
          end=" ", flush=True)
    sys.stdout.flush()
    k_eig = min(6, dim)
    eig_abs = compute_eigenvalues_on_grid(
        intra_p, inter_p, beta_r_arr, y_arr, k=k_eig)
    potts_eig_data[(q, L)] = (eig_abs, beta_r_arr, y_arr)
    print(f"Done ({time.time()-t1:.1f}s)")

    # Stokes network
    stokes_pts = find_stokes_network(eig_abs, beta_r_arr, y_arr)
    potts_stokes_data[(q, L)] = stokes_pts
    print(f"    Stokes boundary points: {len(stokes_pts)}")

    # β_c from Stokes proximity
    beta_c_L, min_y = find_stokes_nearest_real_axis(stokes_pts)
    if beta_c_L is not None:
        potts_beta_c[(q, L)] = beta_c_L
        print(f"    β_c(Stokes) = {beta_c_L:.6f} (exact: {beta_c_exact:.6f}, "
              f"err: {abs(beta_c_L - beta_c_exact):.6f}, min_y: {min_y:.4f})")
    else:
        print(f"    No Stokes points found")

    # Fisher zeros from Stokes seeds
    if dim <= 100 and stokes_pts:
        N_values = [10, 15, 20, 30, 50, 75, 100, 150, 200]

        print(f"    Finding zeros from {len(stokes_pts)} Stokes seeds...",
              end=" ", flush=True)
        sys.stdout.flush()
        result = find_zeros_from_stokes(
            intra_p, inter_p, stokes_pts, N_values, k_eig=k_eig)

        potts_zero_data[(q, L)] = {N: [z for z, _ in zg]
                                    for N, zg in result.items()}
        potts_gap_data[(q, L)] = result

        for N in N_values:
            zg = result.get(N, [])
            nz = len(zg)
            if nz > 0:
                gaps = [g for _, g in zg if np.isfinite(g)]
                avg_gap = np.mean(gaps) if gaps else float('nan')
                print(f"\n      N={N:4d}: {nz:3d} zeros, "
                      f"⟨gap⟩={avg_gap:.2e}", end="")
        print()
    else:
        potts_zero_data[(q, L)] = {}
        potts_gap_data[(q, L)] = {}
        if dim > 100:
            print(f"    Skipping Fisher zeros (dim={dim} too slow)")
        else:
            print(f"    No Stokes points for zero-finding")
    print(f"    [{time.time()-t1:.1f}s total]")


# ===================================================================
# PART 3: Phase Transition Detection — β_c(L) vs exact
# ===================================================================

print(f"\n\n  PART 3: Phase Transition Detection")
print("  " + "-" * 70)

# Ising FSS
print(f"\n  Ising: β_c(L) via Stokes proximity, exact β_c = {beta_c_exact_ising:.6f}")
print(f"  {'L':>4s}  {'β_c(L)':>10s}  {'error':>10s}")
ising_L_list = []
ising_bc_list = []
for L in ising_L_values:
    if L in ising_beta_c:
        err = abs(ising_beta_c[L] - beta_c_exact_ising)
        print(f"  {L:4d}  {ising_beta_c[L]:10.6f}  {err:10.6f}")
        ising_L_list.append(L)
        ising_bc_list.append(ising_beta_c[L])

bc_inf = beta_c_exact_ising  # default
a_fss = 0
res_fss = 0
if len(ising_L_list) >= 2:
    bc_inf, a_fss, res_fss = finite_size_scaling(ising_L_list, ising_bc_list)
    print(f"\n  FSS fit: β_c(L) = {bc_inf:.6f} + {a_fss:.4f}/L")
    print(f"  Extrapolated β_c(∞) = {bc_inf:.6f} (exact: {beta_c_exact_ising:.6f}, "
          f"error: {abs(bc_inf - beta_c_exact_ising):.6f})")

# Potts
print(f"\n  Potts: β_c(L) via Stokes proximity")
print(f"  {'q':>3s}  {'L':>4s}  {'β_c(L)':>10s}  {'exact':>10s}  {'error':>10s}")
for q in [2, 3, 4, 5]:
    bc_exact = np.log(1 + np.sqrt(q))
    for L in [4, 6]:
        if (q, L) in potts_beta_c:
            err = abs(potts_beta_c[(q, L)] - bc_exact)
            print(f"  {q:3d}  {L:4d}  {potts_beta_c[(q,L)]:10.6f}  "
                  f"{bc_exact:10.6f}  {err:10.6f}")


# ===================================================================
# PART 4: Paper Psi Verification — dist(zero, Stokes) ~ O(1/N)
# ===================================================================

print(f"\n\n  PART 4: Stokes Concentration — continuous gap measure")
print("  " + "-" * 70)
print(f"  Measure: gap = |log|λ_0/λ_1|| at each Fisher zero (continuous, no grid floor)")
print(f"  Theory: gap → 0 as N → ∞; for 2-eigenvalue sums, gap ~ exp(-cN)")

concentration_data = {}

# Collect from tracked gap data (Ising)
for L in ising_L_values:
    if L not in ising_gap_data:
        continue
    label = f"Ising L={L}"
    concentration_data[label] = []
    for N, zeros_gaps in sorted(ising_gap_data[L].items()):
        gaps = [g for _, g in zeros_gaps if np.isfinite(g) and g > 0]
        if gaps:
            concentration_data[label].append(
                (N, np.mean(gaps), np.std(gaps), len(gaps)))

# Collect from tracked gap data (Potts)
for key in sorted(potts_gap_data.keys()):
    q, L = key
    label = f"Potts q={q} L={L}"
    concentration_data[label] = []
    for N, zeros_gaps in sorted(potts_gap_data[key].items()):
        gaps = [g for _, g in zeros_gaps if np.isfinite(g) and g > 0]
        if gaps:
            concentration_data[label].append(
                (N, np.mean(gaps), np.std(gaps), len(gaps)))

print(f"\n  {'Model':>20s}  {'N':>5s}  {'⟨gap⟩':>12s}  {'σ(gap)':>12s}  {'#zeros':>6s}")
for label, data in concentration_data.items():
    for N, avg, std, nz in sorted(data):
        print(f"  {label:>20s}  {N:5d}  {avg:12.4e}  {std:12.4e}  {nz:6d}")

# Power-law fits with R²
concentration_fits = {}
print(f"\n  Concentration exponent fits (gap ~ N^α):")
for label, data in concentration_data.items():
    if len(data) < 3:
        continue
    Ns = np.array([d[0] for d in data])
    gs = np.array([d[1] for d in data])
    valid = gs > 0
    if np.sum(valid) < 3:
        continue
    log_N = np.log(Ns[valid])
    log_g = np.log(gs[valid])
    A = np.column_stack([np.ones_like(log_N), log_N])
    coeffs = np.linalg.lstsq(A, log_g, rcond=None)[0]
    alpha = coeffs[1]
    # Also try exponential fit: log(gap) = a + b*N
    A_exp = np.column_stack([np.ones_like(Ns[valid].astype(float)), Ns[valid].astype(float)])
    coeffs_exp = np.linalg.lstsq(A_exp, log_g, rcond=None)[0]
    rate = coeffs_exp[1]
    # R² for power-law fit
    predicted = A @ coeffs
    ss_res = np.sum((log_g - predicted)**2)
    ss_tot = np.sum((log_g - np.mean(log_g))**2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    concentration_fits[label] = (alpha, R2, rate)
    print(f"  {label:>20s}: α = {alpha:.2f}, R² = {R2:.3f}, "
          f"exp rate = {rate:.4f}/N")


# ===================================================================
# PART 5: Density Verification
# ===================================================================

print(f"\n\n  PART 5: Zero Density vs Theorem Prediction")
print("  " + "-" * 70)
print(f"  Theorem: #zeros ~ O(N)")

for L in ising_L_values:
    if L not in ising_zero_data:
        continue
    print(f"\n  Ising L={L}:")
    print(f"  {'N':>5s}  {'#zeros':>7s}  {'#/N':>8s}")
    for N in sorted(ising_zero_data[L].keys()):
        nz = len(ising_zero_data[L][N])
        print(f"  {N:5d}  {nz:7d}  {nz/N:8.3f}")


# ===================================================================
# PART 6: Figure Generation
# ===================================================================

print(f"\n\n  PART 6: Figure Generation")
print("  " + "-" * 70)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("  [matplotlib not available — skipping figures]")

if HAS_MPL:
    # ---------------------------------------------------------------
    # Figure 1: 2D Stokes map for Ising L=8
    # ---------------------------------------------------------------
    L_fig = 8
    if L_fig in ising_eig_data and L_fig in ising_stokes_data:
        print(f"  Generating fig_stokes_ising_L{L_fig}.png...", end=" ", flush=True)
        eig_abs, beta_r_arr, y_arr = ising_eig_data[L_fig]

        fig, ax = plt.subplots(figsize=(10, 8))

        p_dom = np.argmax(eig_abs, axis=2)
        im = ax.pcolormesh(beta_r_arr, y_arr, p_dom.T,
                           cmap='tab10', shading='auto', alpha=0.3)
        plt.colorbar(im, ax=ax, label='Dominant eigenvalue index')

        stokes_pts = ising_stokes_data[L_fig]
        if stokes_pts:
            sk = [s[0] for s in stokes_pts]
            sy = [s[1] for s in stokes_pts]
            ax.scatter(sk, sy, c='gray', s=2, alpha=0.4, label='Stokes boundary')

        markers = {10: 'o', 20: 's', 50: '^', 100: 'D'}
        for N in sorted(ising_zero_data.get(L_fig, {}).keys()):
            zeros = ising_zero_data[L_fig][N]
            if zeros:
                zr = [z.real for z in zeros]
                zi = [z.imag for z in zeros]
                m = markers.get(N, 'o')
                ax.scatter(zr, zi, s=30, marker=m, linewidths=1,
                           zorder=10, label=f'N={N} ({len(zeros)} zeros)')

        ax.axvline(beta_c_exact_ising, color='red', linestyle='--',
                   lw=2, alpha=0.7, label=f'$\\beta_c$ = {beta_c_exact_ising:.4f}')

        ax.set_xlabel(r'$\beta_r = \mathrm{Re}\,\beta$', fontsize=12)
        ax.set_ylabel(r'$y = \mathrm{Im}\,\beta$', fontsize=12)
        ax.set_title(f'2D Ising Strip L={L_fig}: Stokes Map + Fisher Zeros',
                     fontsize=13)
        ax.legend(fontsize=9, loc='upper right')
        plt.tight_layout()
        path = f'{PAPER_DIR}/fig_stokes_ising_L{L_fig}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved")

    # ---------------------------------------------------------------
    # Figure 2: Phase detection — β_c(L) extrapolation + Potts
    # ---------------------------------------------------------------
    print(f"  Generating fig_phase_detection.png...", end=" ", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    if ising_L_list:
        Ls = np.array(ising_L_list)
        bcs = np.array(ising_bc_list)
        ax.plot(1.0 / Ls, bcs, 'bo-', markersize=8, label='Gap minimum')
        ax.axhline(beta_c_exact_ising, color='red', linestyle='--',
                   label=f'Exact $\\beta_c$ = {beta_c_exact_ising:.4f}')
        if len(ising_L_list) >= 2:
            x_fit = np.linspace(0, 0.3, 50)
            ax.plot(x_fit, bc_inf + a_fss * x_fit, 'g--', alpha=0.5,
                    label=f'FSS: $\\beta_c(\\infty)$ = {bc_inf:.4f}')
    ax.set_xlabel('1/L', fontsize=12)
    ax.set_ylabel(r'$\beta_c(L)$', fontsize=12)
    ax.set_title('2D Ising: Finite-Size Scaling', fontsize=13)
    ax.legend(fontsize=9)

    ax = axes[1]
    q_vals = [2, 3, 4, 5]
    q_exact = [np.log(1 + np.sqrt(q)) for q in q_vals]
    ax.plot(q_vals, q_exact, 'r--', lw=2, label='Exact $\\beta_c$')
    for L in [4, 6]:
        q_det, bc_det = [], []
        for q in q_vals:
            if (q, L) in potts_beta_c:
                q_det.append(q)
                bc_det.append(potts_beta_c[(q, L)])
        if q_det:
            ax.plot(q_det, bc_det, 'o-', markersize=8, label=f'Gap min L={L}')
    ax.set_xlabel('q', fontsize=12)
    ax.set_ylabel(r'$\beta_c$', fontsize=12)
    ax.set_title('Potts Model: Phase Transition Detection', fontsize=13)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = f'{PAPER_DIR}/fig_phase_detection.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved")

    # ---------------------------------------------------------------
    # Figure 3: 2×2 Potts Stokes maps
    # ---------------------------------------------------------------
    print(f"  Generating fig_potts_stokes_comparison.png...", end=" ", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    plot_configs = [(2, 4), (3, 4), (4, 4), (5, 4)]
    for idx, (q, L) in enumerate(plot_configs):
        ax = axes[idx // 2, idx % 2]
        bc_exact = np.log(1 + np.sqrt(q))

        if (q, L) in potts_eig_data:
            eig_abs_p, beta_r_arr_p, y_arr_p = potts_eig_data[(q, L)]
            p_dom = np.argmax(eig_abs_p, axis=2)
            ax.pcolormesh(beta_r_arr_p, y_arr_p, p_dom.T,
                         cmap='tab10', shading='auto', alpha=0.3)

        if (q, L) in potts_stokes_data:
            pts = potts_stokes_data[(q, L)]
            if pts:
                ax.scatter([s[0] for s in pts], [s[1] for s in pts],
                          c='gray', s=2, alpha=0.4)

        if (q, L) in potts_zero_data:
            for N in sorted(potts_zero_data[(q, L)].keys()):
                zs = potts_zero_data[(q, L)][N]
                if zs:
                    ax.scatter([z.real for z in zs], [z.imag for z in zs],
                              s=20, marker='x', linewidths=1,
                              zorder=10, label=f'N={N}')

        ax.axvline(bc_exact, color='red', linestyle='--', lw=1.5, alpha=0.7)
        trans_type = 'continuous' if q <= 4 else 'first-order'
        ax.set_title(f'q={q} Potts, L={L} ($\\beta_c$={bc_exact:.3f}, {trans_type})',
                     fontsize=11)
        ax.set_xlabel(r'$\beta_r$', fontsize=10)
        ax.set_ylabel(r'$y$', fontsize=10)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle('Potts Stokes Maps: Fisher Zeros on Stokes Boundaries',
                 fontsize=14)
    plt.tight_layout()
    path = f'{PAPER_DIR}/fig_potts_stokes_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved")

    # ---------------------------------------------------------------
    # Figure 4: Stokes concentration — continuous gap vs N
    # ---------------------------------------------------------------
    print(f"  Generating fig_stokes_concentration.png...", end=" ", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    has_data = False

    for ic, (label, data) in enumerate(concentration_data.items()):
        if len(data) < 2:
            continue
        Ns = np.array([d[0] for d in sorted(data)])
        gs = np.array([d[1] for d in sorted(data)])
        errs = np.array([d[2] for d in sorted(data)])
        valid = gs > 0
        if not np.any(valid):
            continue

        # Left: log-log (power-law check)
        axes[0].loglog(Ns[valid], gs[valid], 'o-', color=colors[ic],
                       label=label, markersize=5, linewidth=1.5)
        # Right: semilog-y (exponential check)
        axes[1].semilogy(Ns[valid], gs[valid], 'o-', color=colors[ic],
                         label=label, markersize=5, linewidth=1.5)
        has_data = True

    if has_data:
        N_ref = np.array([10, 200])
        axes[0].loglog(N_ref, 0.3 / N_ref, 'k--', alpha=0.5, lw=2,
                       label='O(1/N)')
        axes[0].loglog(N_ref, 0.3 / N_ref**2, 'k:', alpha=0.3, lw=1.5,
                       label=r'O(1/N$^2$)')
        axes[0].set_xlabel('N', fontsize=12)
        axes[0].set_ylabel(
            r'$\langle|\log|\lambda_0/\lambda_1||\rangle$ at zeros',
            fontsize=11)
        axes[0].set_title('Log-log: Stokes gap vs N', fontsize=13)
        axes[0].legend(fontsize=8, loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # Add fit annotations
        fit_text = '\n'.join(
            f'{lab}: α={a:.2f}, R²={r2:.3f}'
            for lab, (a, r2, _) in concentration_fits.items())
        if fit_text:
            axes[0].text(0.03, 0.03, fit_text, transform=axes[0].transAxes,
                         fontsize=7, verticalalignment='bottom',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='white',
                                   alpha=0.8))

        axes[1].set_xlabel('N', fontsize=12)
        axes[1].set_ylabel(
            r'$\langle|\log|\lambda_0/\lambda_1||\rangle$ at zeros',
            fontsize=11)
        axes[1].set_title('Semilog: gap vs N (linear = exponential decay)',
                         fontsize=13)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'Insufficient data',
                     transform=axes[0].transAxes, ha='center', fontsize=12)

    plt.suptitle('Fisher Zero Concentration on Stokes Network', fontsize=14,
                 y=1.02)
    plt.tight_layout()
    path = f'{PAPER_DIR}/fig_stokes_concentration.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved")


# ===================================================================
# PART 7: Summary Table
# ===================================================================

print(f"\n\n  PART 7: Summary Table")
print("  " + "=" * 86)

print(f"\n  {'Model':<20s}  {'L':>3s}  {'β_c(Stokes)':>12s}  "
      f"{'β_c(exact)':>10s}  {'error':>8s}  {'#zeros N=50':>12s}")
print(f"  {'-'*20}  {'-'*3}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*12}")

for L in ising_L_values:
    bc_s = ising_beta_c.get(L, float('nan'))
    err = abs(bc_s - beta_c_exact_ising) if L in ising_beta_c else float('nan')
    nz = len(ising_zero_data.get(L, {}).get(50, []))
    print(f"  {'Ising':<20s}  {L:3d}  {bc_s:12.6f}  "
          f"{beta_c_exact_ising:10.6f}  {err:8.5f}  {nz:12d}")

for q in [2, 3, 4, 5]:
    bc_exact = np.log(1 + np.sqrt(q))
    for L in [4, 6]:
        if (q, L) in potts_beta_c:
            bc_s = potts_beta_c[(q, L)]
            err = abs(bc_s - bc_exact)
            nz = len(potts_zero_data.get((q, L), {}).get(50, []))
            print(f"  {'Potts q='+str(q):<20s}  {L:3d}  {bc_s:12.6f}  "
                  f"{bc_exact:10.6f}  {err:8.5f}  {nz:12d}")

if len(ising_L_list) >= 2:
    print(f"\n  Finite-size scaling: β_c(∞) = {bc_inf:.6f} "
          f"(exact: {beta_c_exact_ising:.6f}, "
          f"error: {abs(bc_inf - beta_c_exact_ising):.6f})")

print(f"\n  Stokes concentration (gap ~ N^α):")
print(f"  {'Model':>20s}  {'α':>6s}  {'R²':>6s}")
print(f"  {'-'*20}  {'-'*6}  {'-'*6}")
for label, (alpha, R2, rate) in concentration_fits.items():
    print(f"  {label:>20s}  {alpha:6.2f}  {R2:6.3f}")

elapsed = time.time() - t0
print(f"\n\n  [Total runtime: {elapsed:.1f}s]")
print("=" * 90)
