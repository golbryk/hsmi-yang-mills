"""
n-Plaquette Fisher Zero Spacing Verification
=============================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 5 — n-Plaquette conjecture

Conjecture
----------
For N odd, n >= 2, kappa > 0:

    <Delta> = pi/n

where <Delta> is the average imaginary spacing of Fisher zeros of
Z_{nP}^{SU(N)} near the line Re s = kappa.

Methods
-------
1. |Z| minima on Re s = kappa + Newton refinement to find exact zeros
2. Argument principle on tall rectangles for total zero count
3. Formula-based prediction extending spacing_table.py

Usage
-----
    python n_plaquette_proof.py
    python n_plaquette_proof.py --N 5 --kappa 2.0
"""

import argparse
import numpy as np
from math import comb, pi, acos, exp


# ---------------------------------------------------------------------------
# SU(N) helpers
# ---------------------------------------------------------------------------

def dim_rep(p, N):
    return comb(p + N - 1, N - 1)


def casimir(p, N):
    return p * (p + N) / float(N)


def h_p_vec(p, eigs):
    """h_p via Newton's identity, vectorized. eigs: (n_pts, N) complex."""
    n_pts = eigs.shape[0]
    if p == 0:
        return np.ones(n_pts, dtype=complex)
    psums = np.array([np.sum(eigs ** k, axis=1) for k in range(1, p + 1)])
    h = np.zeros((p + 1, n_pts), dtype=complex)
    h[0] = 1.0
    for k in range(1, p + 1):
        h[k] = sum(psums[j] * h[k - 1 - j] for j in range(k)) / k
    return h[p]


def h_p_scalar(p, eigenvalues):
    """h_p via Newton's identity for a single set of eigenvalues."""
    if p == 0:
        return 1.0
    psums = [sum(e ** k for e in eigenvalues) for k in range(1, p + 1)]
    h = [0.0] * (p + 1)
    h[0] = 1.0
    for k in range(1, p + 1):
        h[k] = sum(psums[j] * h[k - 1 - j] for j in range(k)) / k
    return h[p]


# ---------------------------------------------------------------------------
# Precomputed Weyl grid for fast evaluation
# ---------------------------------------------------------------------------

class WeylGrid:
    """Precomputed Weyl integration grid for SU(N)."""

    def __init__(self, N, n_quad=40, n_reps=12):
        self.N = N
        self.n_quad = n_quad
        self.n_reps = n_reps
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

        self.Phi = np.sum(np.cos(theta_all), axis=1)
        self.measure = w * V2 / (2 * np.pi) ** dim
        self.norm = np.sum(self.measure).real
        self.hp_list = [h_p_vec(p, z) for p in range(n_reps)]
        self.dim_list = [dim_rep(p, N) for p in range(n_reps)]
        self.n_pts = n_pts

    def eval_ZnP(self, s, n_plaq):
        """Evaluate Z_{nP}(s) using precomputed grid."""
        exp_sPhi = np.exp(s * self.Phi)
        Z = 0j
        for p in range(self.n_reps):
            Ap = np.sum(self.hp_list[p] * exp_sPhi * self.measure) / self.norm
            Z += self.dim_list[p] * Ap ** n_plaq
        return Z

    def eval_ZnP_and_deriv(self, s, n_plaq):
        """Evaluate Z_{nP}(s) and dZ/ds using precomputed grid."""
        exp_sPhi = np.exp(s * self.Phi)
        Z = 0j
        dZ = 0j
        for p in range(self.n_reps):
            hp_exp = self.hp_list[p] * exp_sPhi
            Ap = np.sum(hp_exp * self.measure) / self.norm
            dAp = np.sum(hp_exp * self.Phi * self.measure) / self.norm
            Z += self.dim_list[p] * Ap ** n_plaq
            if n_plaq > 1 and abs(Ap) > 1e-30:
                dZ += self.dim_list[p] * n_plaq * Ap ** (n_plaq - 1) * dAp
            elif n_plaq == 1:
                dZ += self.dim_list[p] * dAp
        return Z, dZ


# ---------------------------------------------------------------------------
# Newton's method for finding zeros in complex s-plane
# ---------------------------------------------------------------------------

def newton_zero(grid, s0, n_plaq, max_iter=80, tol=1e-12):
    """Find a zero of Z_{nP}(s) near s0 using Newton's method."""
    s = s0
    for it in range(max_iter):
        Z, dZ = grid.eval_ZnP_and_deriv(s, n_plaq)
        if abs(dZ) < 1e-30:
            break
        s_new = s - Z / dZ
        if abs(s_new - s) < tol:
            return s_new, abs(grid.eval_ZnP(s_new, n_plaq)), True
        s = s_new
    Z_final = grid.eval_ZnP(s, n_plaq)
    return s, abs(Z_final), abs(Z_final) < 1e-6


# ---------------------------------------------------------------------------
# |Z| minima detection on Re s = kappa
# ---------------------------------------------------------------------------

def find_absZ_minima(grid, kappa, y_start, y_end, n_plaq, n_pts=2000):
    """Find local minima of |Z_{nP}(kappa+iy)| along Re s = kappa."""
    y_vals = np.linspace(y_start, y_end, n_pts)
    absZ = np.zeros(n_pts)
    for i in range(n_pts):
        absZ[i] = abs(grid.eval_ZnP(complex(kappa, y_vals[i]), n_plaq))

    minima = []
    for i in range(1, n_pts - 1):
        if absZ[i] < absZ[i - 1] and absZ[i] < absZ[i + 1]:
            minima.append({'y': y_vals[i], 'absZ': absZ[i]})

    return minima, y_vals, absZ


# ---------------------------------------------------------------------------
# Full zero-finding: |Z| minima → Newton refinement
# ---------------------------------------------------------------------------

def find_zeros(grid, kappa, y_start, y_end, n_plaq, n_scan=2000,
               newton_tol=1e-12, re_s_tolerance=2.0):
    """Find Fisher zeros near Re s = kappa by |Z| minima + Newton.

    Parameters
    ----------
    re_s_tolerance : float
        Maximum |Re(s) - kappa| for a zero to be counted as "near kappa".
    """
    minima, y_vals, absZ = find_absZ_minima(
        grid, kappa, y_start, y_end, n_plaq, n_scan
    )

    zeros = []
    for m in minima:
        s0 = complex(kappa, m['y'])
        s_zero, residual, converged = newton_zero(
            grid, s0, n_plaq, tol=newton_tol
        )
        if converged and abs(s_zero.real - kappa) < re_s_tolerance:
            zeros.append({
                'Re_s': s_zero.real,
                'Im_s': s_zero.imag,
                'residual': residual,
                'distance': abs(s_zero.real - kappa),
            })

    # Remove duplicates (Newton may converge to same zero from nearby minima)
    unique_zeros = []
    for z in sorted(zeros, key=lambda x: x['Im_s']):
        if not unique_zeros or abs(z['Im_s'] - unique_zeros[-1]['Im_s']) > 0.01:
            unique_zeros.append(z)

    return unique_zeros


# ---------------------------------------------------------------------------
# Argument principle on tall rectangles
# ---------------------------------------------------------------------------

def argument_principle_tall(grid, kappa, delta, y_start, y_end, n_plaq,
                            n_per_edge=100):
    """Count zeros via argument principle on a tall rectangle.

    Rectangle: [kappa-delta, kappa+delta] x [y_start, y_end]
    """
    sigma1 = kappa - delta
    sigma2 = kappa + delta
    height = y_end - y_start
    n_horiz = max(20, int(n_per_edge * 2 * delta / height))
    n_vert = n_per_edge

    contour = []
    # Bottom
    for sigma in np.linspace(sigma1, sigma2, n_horiz, endpoint=False):
        contour.append(complex(sigma, y_start))
    # Right
    for y in np.linspace(y_start, y_end, n_vert, endpoint=False):
        contour.append(complex(sigma2, y))
    # Top
    for sigma in np.linspace(sigma2, sigma1, n_horiz, endpoint=False):
        contour.append(complex(sigma, y_end))
    # Left
    for y in np.linspace(y_end, y_start, n_vert, endpoint=False):
        contour.append(complex(sigma1, y))

    Z_vals = [grid.eval_ZnP(s, n_plaq) for s in contour]

    total_phase = 0.0
    min_absZ = float('inf')
    for i in range(len(Z_vals)):
        absZ = abs(Z_vals[i])
        if absZ < min_absZ:
            min_absZ = absZ
        j = (i + 1) % len(Z_vals)
        if abs(Z_vals[j]) < 1e-30 or abs(Z_vals[i]) < 1e-30:
            continue
        dphase = np.angle(Z_vals[j] / Z_vals[i])
        total_phase += dphase

    winding = total_phase / (2 * pi)
    return round(winding), winding, min_absZ


# ---------------------------------------------------------------------------
# Formula-based prediction (from spacing_table.py)
# ---------------------------------------------------------------------------

def formula_zeros_n2(N, kappa, n_reps=20, n_zeros=30):
    """Compute predicted zeros for n=2 using the known formula.

    Returns list of y-values where Fisher zeros are predicted.
    """
    assert N % 2 == 1
    n_plus = (N + 1) // 2
    eig_A = [1.0] * n_plus + [-1.0] * (N - n_plus)

    S_A = 0.0
    S_AB = 0.0
    for p in range(n_reps):
        d = dim_rep(p, N)
        chi_A = h_p_scalar(p, eig_A)
        heat = d * exp(-casimir(p, N) * kappa)
        S_A += d * chi_A ** 2 * heat
        S_AB += ((-1) ** p) * d * chi_A ** 2 * heat

    rho = max(-1 + 1e-12, min(1 - 1e-12, -S_AB / (2 * S_A)))
    alpha = acos(rho)

    omega = 2  # for N odd, two-plaquette
    zeros_plus = [(alpha + 2 * pi * k) / omega for k in range(n_zeros)]
    zeros_minus = [((2 * pi - alpha) + 2 * pi * k) / omega
                   for k in range(n_zeros)]
    all_zeros = sorted(z for z in zeros_plus + zeros_minus if z > 0)

    return all_zeros[:n_zeros], rho, alpha


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def verify_nplaquette(N, kappa, n_plaq_list, n_quad=40, n_reps=12,
                      y_start=5.0, y_end=50.0, n_scan=3000):
    """Full verification of <Delta> = pi/n for Z_{nP}^{SU(N)}."""
    print()
    print("=" * 78)
    print("  n-PLAQUETTE FISHER ZERO SPACING VERIFICATION")
    print(f"  SU({N}), kappa = {kappa}")
    print(f"  Author: Grzegorz Olbryk  |  March 2026  |  ROADMAP Task 5")
    print("=" * 78)

    print(f"\n  [0] Precomputing Weyl grid: N={N}, n_quad={n_quad}, n_reps={n_reps}")
    print(f"      Grid size: {n_quad}^{N-1} = {n_quad**(N-1)} points")
    grid = WeylGrid(N, n_quad, n_reps)
    print("      Done.")

    # Normalization check
    print("\n  Normalization: Z_{nP}(0)")
    for n_plaq in n_plaq_list:
        Z0 = grid.eval_ZnP(0.0, n_plaq)
        print(f"    n={n_plaq}: Z(0) = {Z0.real:.8f} + {Z0.imag:.2e}i")

    # -----------------------------------------------------------------------
    # Method 1: |Z| minima + Newton zero-finding
    # -----------------------------------------------------------------------
    print(f"\n  [1] Zero-finding: |Z| minima on Re s = kappa + Newton refinement")
    print(f"      y range: [{y_start:.1f}, {y_end:.1f}], scan points: {n_scan}")
    print("  " + "-" * 68)

    results = {}

    for n_plaq in n_plaq_list:
        print(f"\n    --- n = {n_plaq} plaquettes ---")

        zeros = find_zeros(
            grid, kappa, y_start, y_end, n_plaq,
            n_scan=n_scan, re_s_tolerance=2.0
        )

        print(f"    Found {len(zeros)} zeros in [{y_start:.1f}, {y_end:.1f}]")

        if len(zeros) >= 3:
            # Print first and last few zeros
            print(f"\n    {'k':>3}  {'Im(s)':>10}  {'Re(s)':>10}  "
                  f"{'|Re(s)-kappa|':>13}  {'Delta':>10}  {'|Z|':>10}")
            print("    " + "-" * 65)

            gaps = []
            for i, z in enumerate(zeros):
                gap = z['Im_s'] - zeros[i - 1]['Im_s'] if i > 0 else None
                if gap is not None:
                    gaps.append(gap)
                gap_str = f"{gap:.6f}" if gap is not None else "---"
                if i < 5 or i >= len(zeros) - 3:
                    print(f"    {i:>3}  {z['Im_s']:>10.5f}  {z['Re_s']:>10.5f}  "
                          f"{z['distance']:>13.5f}  {gap_str:>10}  "
                          f"{z['residual']:>10.2e}")
                elif i == 5:
                    print(f"    {'...':>3}")

            if gaps:
                mean_gap = np.mean(gaps)
                expected = pi / n_plaq
                err = abs(mean_gap - expected) / expected

                # n-fold sums (should each equal pi)
                nfold_sums = []
                for i in range(0, len(gaps) - n_plaq + 1, n_plaq):
                    s = sum(gaps[i:i + n_plaq])
                    nfold_sums.append(s)

                print(f"\n    Mean gap       = {mean_gap:.6f}")
                print(f"    pi/{n_plaq}          = {expected:.6f}")
                print(f"    Relative error = {err:.4%}")

                if nfold_sums:
                    mean_nfold = np.mean(nfold_sums)
                    print(f"    Mean {n_plaq}-fold sum = {mean_nfold:.6f}  "
                          f"(expected: pi = {pi:.6f})")

                # Mean distance from kappa
                mean_dist = np.mean([z['distance'] for z in zeros])
                print(f"    Mean |Re(s)-kappa| = {mean_dist:.6f}")

                results[n_plaq] = {
                    'n_zeros': len(zeros),
                    'mean_gap': mean_gap,
                    'expected': expected,
                    'error': err,
                    'mean_dist': mean_dist,
                    'gaps': gaps,
                    'nfold_sums': nfold_sums,
                }
        else:
            print("    Too few zeros for analysis.")
            results[n_plaq] = None

    # -----------------------------------------------------------------------
    # Method 2: Argument principle on tall rectangles
    # -----------------------------------------------------------------------
    print(f"\n\n  [2] Argument principle: tall rectangle zero count")
    print("  " + "-" * 68)

    for delta in [0.5, 1.0, 2.0]:
        print(f"\n    delta = {delta}:  "
              f"rectangle [{kappa-delta:.1f}, {kappa+delta:.1f}] x "
              f"[{y_start:.0f}, {y_end:.0f}]")
        print(f"    {'n':>4}  {'zeros':>8}  {'zeros/pi':>10}  "
              f"{'expected':>10}  {'<Delta>':>10}  {'pi/n':>10}")
        print("    " + "-" * 60)

        for n_plaq in n_plaq_list:
            n_int, w_exact, min_absZ = argument_principle_tall(
                grid, kappa, delta, y_start, y_end, n_plaq,
                n_per_edge=200
            )
            n_periods = (y_end - y_start) / pi
            zeros_per_pi = n_int / n_periods
            implied_gap = pi / zeros_per_pi if zeros_per_pi > 0 else float('inf')
            expected = pi / n_plaq
            print(f"    {n_plaq:>4}  {n_int:>8}  {zeros_per_pi:>10.3f}  "
                  f"{n_plaq:>10.3f}  {implied_gap:>10.5f}  {expected:>10.5f}")

    # -----------------------------------------------------------------------
    # Method 3: n=2 formula validation
    # -----------------------------------------------------------------------
    if 2 in n_plaq_list:
        print(f"\n\n  [3] n=2 cross-check: formula vs Newton zeros")
        print("  " + "-" * 68)

        formula_y, rho, alpha = formula_zeros_n2(N, kappa, n_reps)
        newton_zeros = results.get(2)

        if newton_zeros and newton_zeros['n_zeros'] >= 3:
            nz = find_zeros(grid, kappa, y_start, y_end, 2,
                            n_scan=n_scan, re_s_tolerance=2.0)
            print(f"    rho = {rho:.6f}, alpha = {alpha:.6f}")
            print(f"    Formula gaps: short = {pi - alpha:.6f}, "
                  f"long = {alpha:.6f}, pair sum = {pi:.6f}")
            print(f"\n    {'k':>3}  {'y_formula':>10}  {'y_Newton':>10}  "
                  f"{'|diff|':>10}")
            print("    " + "-" * 42)
            n_cmp = min(len(formula_y), len(nz), 10)
            # Match formula zeros to Newton zeros by closest Im(s)
            for i in range(n_cmp):
                yf = formula_y[i]
                if yf < y_start or yf > y_end:
                    continue
                # Find closest Newton zero
                best_j = min(range(len(nz)),
                             key=lambda j: abs(nz[j]['Im_s'] - yf))
                yn = nz[best_j]['Im_s']
                print(f"    {i:>3}  {yf:>10.5f}  {yn:>10.5f}  "
                      f"{abs(yf - yn):>10.5f}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n\n  {'='*78}")
    print("  SUMMARY: n-Plaquette Fisher Zero Spacing")
    print(f"  {'='*78}")
    print(f"\n  {'n':>4}  {'#zeros':>8}  {'<Delta>':>10}  {'pi/n':>10}  "
          f"{'error':>8}  {'|Re(s)-k|':>10}  {'verdict':>10}")
    print("  " + "-" * 70)
    for n_plaq in n_plaq_list:
        r = results.get(n_plaq)
        if r:
            verdict = "PASS" if r['error'] < 0.05 else (
                "CLOSE" if r['error'] < 0.15 else "CHECK")
            print(f"  {n_plaq:>4}  {r['n_zeros']:>8}  {r['mean_gap']:>10.6f}  "
                  f"{r['expected']:>10.6f}  {r['error']:>7.2%}  "
                  f"{r['mean_dist']:>10.5f}  {verdict:>10}")
        else:
            print(f"  {n_plaq:>4}  {'N/A':>8}  {'N/A':>10}  {pi/n_plaq:>10.6f}  "
                  f"{'N/A':>8}  {'N/A':>10}  {'FAIL':>10}")

    print(f"\n  Conjecture: <Delta> = pi/n")
    all_pass = all(results.get(n) and results[n]['error'] < 0.05
                   for n in n_plaq_list)
    if all_pass:
        print("  STATUS: ALL PASS within 5% tolerance")
    else:
        close = all(results.get(n) and results[n]['error'] < 0.15
                    for n in n_plaq_list)
        if close:
            print("  STATUS: All within 15% — consistent with conjecture")
        else:
            print("  STATUS: Some cases need further investigation")
    print(f"  {'='*78}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="n-Plaquette Fisher zero spacing verification. "
                    "Author: Grzegorz Olbryk."
    )
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--kappa', type=float, default=1.0)
    parser.add_argument('--n_plaq', type=int, nargs='+', default=[2, 3, 4, 5])
    parser.add_argument('--n_quad', type=int, default=40)
    parser.add_argument('--n_reps', type=int, default=12)
    parser.add_argument('--y_start', type=float, default=5.0)
    parser.add_argument('--y_end', type=float, default=50.0)
    parser.add_argument('--n_scan', type=int, default=3000)
    args = parser.parse_args()

    verify_nplaquette(
        args.N, args.kappa, args.n_plaq, args.n_quad, args.n_reps,
        args.y_start, args.y_end, args.n_scan
    )


if __name__ == '__main__':
    main()
