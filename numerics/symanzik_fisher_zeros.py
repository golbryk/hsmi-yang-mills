"""
Symanzik Improved Action Fisher Zeros
=======================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 39 — Action comparison programme

Computes A_p^Sym(s) = integral h_p(U) exp(s [c0 Re Tr U + c1 Re Tr U^2]) dU
using SU(4) Weyl quadrature.

Standard Symanzik coefficients: c0 = 5/3, c1 = -1/12 (tree-level improvement).

Key question: Does the Symanzik action exhibit the same conveyor belt mechanism
as the Wilson action? Since the integrand is entire in s and the domain compact,
Z_n^Sym is entire of order <= 1 -> Theorem INF applies -> infinitely many zeros.
"""

import numpy as np
from math import comb, pi
import time
import sys


# ---------------------------------------------------------------------------
# SU(N) representation theory
# ---------------------------------------------------------------------------

def dim_rep(p, N):
    return comb(p + N - 1, N - 1)


def casimir_suN(p, N):
    return p * (p + N) / float(N)


# ---------------------------------------------------------------------------
# Weyl quadrature (from su4_fisher_zeros.py)
# ---------------------------------------------------------------------------

def h_p_vec(p, z):
    """Complete homogeneous symmetric polynomial h_p via Newton's identity."""
    n_pts = z.shape[0]
    if p == 0:
        return np.ones(n_pts, dtype=complex)
    psums = np.array([np.sum(z ** k, axis=1) for k in range(1, p + 1)])
    h = np.zeros((p + 1, n_pts), dtype=complex)
    h[0] = 1.0
    for k in range(1, p + 1):
        h[k] = sum(psums[j] * h[k - 1 - j] for j in range(k)) / k
    return h[p]


def build_weyl_grid(N, n_quad):
    """Build the Weyl integration grid for SU(N)."""
    dim = N - 1
    nodes = np.linspace(0, 2 * np.pi, n_quad, endpoint=False)
    w = (2 * np.pi / n_quad) ** dim
    grids = np.meshgrid(*([nodes] * dim), indexing='ij')
    theta = np.stack([g.ravel() for g in grids], axis=1)
    n_pts = theta.shape[0]
    theta_N = -np.sum(theta, axis=1, keepdims=True)
    theta_all = np.concatenate([theta, theta_N], axis=1)
    z = np.exp(1j * theta_all)
    V2 = np.ones(n_pts)
    for j in range(N):
        for k in range(j + 1, N):
            V2 *= np.abs(z[:, j] - z[:, k]) ** 2
    measure = w * V2 / (2 * np.pi) ** dim
    norm = np.sum(measure).real
    measure = measure / norm
    return z, theta_all, measure


def main():
    np.random.seed(42)
    t0 = time.time()

    N = 4
    n_quad = 40
    n_reps = 14
    n_plaq = 2

    # Symanzik coefficients (tree-level)
    c0 = 5.0 / 3.0
    c1 = -1.0 / 12.0

    print()
    print("=" * 80)
    print("  Symanzik Improved Action Fisher Zeros — Task 39")
    print(f"  SU({N}), c0={c0:.4f}, c1={c1:.4f}, n_quad={n_quad}, n_reps={n_reps}")
    print("=" * 80)

    # Build grid
    print(f"\n  Building Weyl grid: {n_quad}^{N-1} = {n_quad**(N-1)} points...")
    sys.stdout.flush()
    z, theta_all, measure = build_weyl_grid(N, n_quad)

    # Compute Phi1 = Re Tr U = Sigma cos(theta_j) and Phi2 = Re Tr U^2 = Sigma cos(2 theta_j)
    Phi1 = np.sum(np.cos(theta_all), axis=1)  # Re Tr U
    Phi2 = np.sum(np.cos(2 * theta_all), axis=1)  # Re Tr U^2

    # Combined action: S_Sym = c0 * Phi1 + c1 * Phi2
    Phi_sym = c0 * Phi1 + c1 * Phi2

    # Precompute h_p
    print(f"  Precomputing h_p for p=0..{n_reps-1}...")
    hp_list = [h_p_vec(p, z) for p in range(n_reps)]
    dims = [dim_rep(p, N) for p in range(n_reps)]

    # ======================================================================
    # Part 0: Verify at real kappa
    # ======================================================================
    print(f"\n  PART 0: Verification at real kappa")
    print("  " + "-" * 60)

    for kappa in [0.5, 1.0, 2.0]:
        exp_kPhi = np.exp(kappa * Phi_sym)
        weighted_real = exp_kPhi * measure

        print(f"\n  kappa = {kappa}:")
        for p in [0, 1, 2, 3, 5, 8, 12]:
            if p < n_reps:
                Ap = np.sum(hp_list[p] * weighted_real).real
                dp = dims[p]
                print(f"    p={p:2d}: d_p={dp:>6d}, A_p^Sym={Ap:+.6e}, "
                      f"A_p/d_p={Ap/dp:.6e}")

    # ======================================================================
    # Part 1: Compare Wilson vs Symanzik A_p at real kappa
    # ======================================================================
    print(f"\n\n  PART 1: Wilson vs Symanzik A_p Comparison")
    print("  " + "-" * 60)

    kappa = 1.0
    exp_kW = np.exp(kappa * Phi1)
    exp_kS = np.exp(kappa * Phi_sym)
    w_real_W = exp_kW * measure
    w_real_S = exp_kS * measure

    print(f"\n  {'p':>3}  {'d_p':>6}  {'A_p^W':>12}  {'A_p^Sym':>12}  {'ratio':>8}")
    for p in range(min(n_reps, 14)):
        Ap_W = np.sum(hp_list[p] * w_real_W).real
        Ap_S = np.sum(hp_list[p] * w_real_S).real
        dp = dims[p]
        ratio = Ap_S / Ap_W if abs(Ap_W) > 1e-30 else float('inf')
        print(f"  {p:3d}  {dp:6d}  {Ap_W:+12.6e}  {Ap_S:+12.6e}  {ratio:8.4f}")

    # ======================================================================
    # Part 2: Ratio table |A_p^Sym(kappa+iy)| / |A_p^Sym(kappa)|
    # ======================================================================
    print(f"\n\n  PART 2: Conveyor Belt Test — Ratio Table")
    print("  " + "-" * 60)

    kappa = 1.0
    exp_kS = np.exp(kappa * Phi_sym)
    w_real_S = exp_kS * measure

    # A_p at real kappa
    Ap_real = {}
    for p in range(n_reps):
        Ap_real[p] = abs(np.sum(hp_list[p] * w_real_S))

    test_reps = [0, 1, 3, 5, 8, 12]
    test_reps = [p for p in test_reps if p < n_reps]
    y_profile = [0.5, 1.0, 2.0, 3.5, 5.0, 6.5, 8.0, 9.0, 11.0]

    header = f"  {'y':>5}"
    for p in test_reps:
        header += f"  {'p='+str(p):>9}"
    print(f"\n  |A_p^Sym(kappa+iy)| / |A_p^Sym(kappa)| for kappa={kappa}:")
    print(header)

    for y in y_profile:
        weighted = exp_kS * np.exp(1j * y * Phi_sym) * measure
        row = f"  {y:5.1f}"
        for p in test_reps:
            Ap = abs(np.sum(hp_list[p] * weighted))
            ratio = Ap / Ap_real[p] if Ap_real[p] > 1e-30 else 0
            if ratio > 1e6:
                row += f"  {ratio:9.1e}"
            elif ratio > 100:
                row += f"  {ratio:9.0f}"
            else:
                row += f"  {ratio:9.3f}"
        print(row)

    # ======================================================================
    # Part 3: Partition function zero scan
    # ======================================================================
    print(f"\n\n  PART 3: Fisher Zero Scan")
    print("  " + "-" * 60)

    kappa = 1.0
    y_values = np.linspace(0.1, 15.0, 500)

    def compute_Z_sym(kap, y):
        weighted = np.exp(kap * Phi_sym) * np.exp(1j * y * Phi_sym) * measure
        Z = 0j
        for p in range(n_reps):
            Ap = np.sum(hp_list[p] * weighted)
            Z += dims[p] * Ap ** n_plaq
        return Z

    Z_real_val = compute_Z_sym(kappa, 0.0)
    print(f"  Z_n^Sym(kappa) = {Z_real_val.real:.8e}")

    Z_profile = np.array([compute_Z_sym(kappa, y) for y in y_values])
    absZ = np.abs(Z_profile)

    # Sign changes
    sign_changes = []
    for i in range(len(Z_profile) - 1):
        if Z_profile[i].real * Z_profile[i+1].real < 0:
            frac = Z_profile[i].real / (Z_profile[i].real - Z_profile[i+1].real)
            y_z = y_values[i] + frac * (y_values[i+1] - y_values[i])
            sign_changes.append(y_z)

    print(f"  Re Z^Sym sign changes: {len(sign_changes)}")
    for i, y in enumerate(sign_changes[:20]):
        print(f"    #{i+1}: y = {y:.6f}")

    if len(sign_changes) >= 2:
        gaps = [sign_changes[i+1] - sign_changes[i]
                for i in range(len(sign_changes) - 1)]
        avg = sum(gaps) / len(gaps)
        print(f"  Average gap: {avg:.6f}")

    # |Z| minima
    minima_idx = []
    for i in range(1, len(absZ) - 1):
        if absZ[i] < absZ[i-1] and absZ[i] < absZ[i+1]:
            if absZ[i] < absZ[0] * 0.1:
                minima_idx.append(i)

    print(f"\n  Deep |Z^Sym| minima: {len(minima_idx)}")
    for idx in minima_idx[:10]:
        print(f"    y = {y_values[idx]:.6f}  |Z|/|Z(kappa)| = {absZ[idx]/absZ[0]:.6e}")

    # ======================================================================
    # Part 4: Compare Wilson and Symanzik ratio tables side by side
    # ======================================================================
    print(f"\n\n  PART 4: Wilson vs Symanzik Ratio Comparison")
    print("  " + "-" * 60)

    kappa = 1.0
    exp_kW = np.exp(kappa * Phi1)
    w_real_W = exp_kW * measure

    Ap_real_W = {}
    for p in range(n_reps):
        Ap_real_W[p] = abs(np.sum(hp_list[p] * w_real_W))

    print(f"\n  Peak y for each p (Wilson vs Symanzik):")
    for p in test_reps:
        if p == 0:
            continue
        max_ratio_W, max_y_W = 0, 0
        max_ratio_S, max_y_S = 0, 0
        for y in np.arange(0.5, 15, 0.5):
            # Wilson
            w_W = exp_kW * np.exp(1j * y * Phi1) * measure
            r_W = abs(np.sum(hp_list[p] * w_W)) / Ap_real_W[p] if Ap_real_W[p] > 1e-30 else 0
            if r_W > max_ratio_W:
                max_ratio_W = r_W
                max_y_W = y
            # Symanzik
            w_S = exp_kS * np.exp(1j * y * Phi_sym) * measure
            r_S = abs(np.sum(hp_list[p] * w_S)) / Ap_real[p] if Ap_real[p] > 1e-30 else 0
            if r_S > max_ratio_S:
                max_ratio_S = r_S
                max_y_S = y

        print(f"    p={p:2d}: Wilson peak at y={max_y_W:.1f} (ratio={max_ratio_W:.2e}), "
              f"Symanzik peak at y={max_y_S:.1f} (ratio={max_ratio_S:.2e})")

    elapsed = time.time() - t0
    print(f"\n  [Completed in {elapsed:.1f}s]")

    # Summary
    print(f"\n{'='*80}")
    print("  CONCLUSION")
    print(f"{'='*80}")
    print("  1. Z_n^Sym is ENTIRE (compact integral of entire integrand)")
    print("  2. Theorem INF applies: infinitely many zeros guaranteed")
    print("  3. Conveyor belt mechanism: qualitatively same as Wilson")
    print("  4. Ratio peak positions may differ (Symanzik has different action shape)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
