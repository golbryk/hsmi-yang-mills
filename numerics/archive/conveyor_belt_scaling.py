"""
Conveyor Belt Scaling Law: p*(y) ~ a*y
========================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026

Uses the CROSSING criterion: p*(y) = first p >= 1 where |T_p(κ+iy)| > |T_0(κ+iy)|.
This is cleaner than the anti-phase criterion used earlier.

SU(4) exact Weyl quadrature with high p_max.
"""

import numpy as np
from math import comb, pi
import time


def dim_rep(p, N):
    return comb(p + N - 1, N - 1)


def h_p_vec(p, z):
    n_pts = z.shape[0]
    if p == 0:
        return np.ones(n_pts, dtype=complex)
    psums = np.array([np.sum(z ** k, axis=1) for k in range(1, p + 1)])
    h = np.zeros((p + 1, n_pts), dtype=complex)
    h[0] = 1.0
    for k in range(1, p + 1):
        h[k] = sum(psums[j] * h[k - 1 - j] for j in range(k)) / k
    return h[p]


def build_weyl_grid_su4(n_quad=50):
    N = 4
    dim = 3
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
    Phi = np.sum(np.cos(theta_all), axis=1)
    measure = w * V2 / (2 * np.pi) ** dim
    norm = np.sum(measure).real
    measure /= norm
    return z, Phi, measure


def main():
    np.random.seed(42)
    kappa = 1.0
    n_plaq = 2
    N = 4
    p_max = 50
    n_quad = 50

    print("=" * 80)
    print("  Conveyor Belt Scaling: p*(y) via Crossing Criterion")
    print("  SU(4), kappa=1.0, n=2, p_max=50, n_quad=50")
    print("=" * 80)

    t0 = time.time()
    print(f"\n  Building grid: {n_quad}^3 = {n_quad**3} points...")
    z, Phi, measure = build_weyl_grid_su4(n_quad)

    print(f"  Precomputing h_p for p=0..{p_max}...")
    hp_list = [h_p_vec(p, z) for p in range(p_max + 1)]
    exp_kPhi = np.exp(kappa * Phi)

    # Verify at real kappa
    weighted_real = exp_kPhi * measure
    A0_real = np.sum(hp_list[0] * weighted_real).real
    print(f"\n  A_0(kappa) = {A0_real:.6e}")
    for p in [1, 5, 10, 20, 30, 40, 50]:
        if p <= p_max:
            Ap = np.sum(hp_list[p] * weighted_real).real
            dp = dim_rep(p, N)
            print(f"  A_{p}(kappa) = {Ap:.6e}, d_p = {dp}, "
                  f"A_p/d_p = {Ap/dp:.6e}")

    # ==============================================================
    # Crossing criterion: first p where |T_p| > |T_0|
    # ==============================================================
    y_values = np.arange(0.5, 18, 0.25)
    crossing_p = []

    print(f"\n  Computing crossing p*(y) for {len(y_values)} y values...")

    for y in y_values:
        weighted = exp_kPhi * np.exp(1j * y * Phi) * measure

        # Compute |T_p| = d_p |A_p|^n
        T_abs = np.zeros(p_max + 1)
        for p in range(p_max + 1):
            Ap = np.sum(hp_list[p] * weighted)
            dp = dim_rep(p, N)
            T_abs[p] = dp * abs(Ap) ** n_plaq

        T0 = T_abs[0]

        # Find first p >= 1 where |T_p| > |T_0|
        first_p = -1
        for p in range(1, p_max + 1):
            if T_abs[p] > T0:
                first_p = p
                break
        crossing_p.append(first_p)

    # Print
    print(f"\n  {'y':>6}  {'p*_cross':>8}  note")
    for i, (y, pp) in enumerate(zip(y_values, crossing_p)):
        if i % 4 == 0 or (i > 0 and crossing_p[i] != crossing_p[i-1]):
            note = ""
            if pp == -1:
                note = "  (no crossing)"
            elif pp >= p_max - 5:
                note = "  (near ceiling)"
            print(f"  {y:6.2f}  {pp:8d}  {note}")

    # ==============================================================
    # Alternative: dominant p (largest |T_p| for p >= 1)
    # ==============================================================
    dominant_p = []
    for y in y_values:
        weighted = exp_kPhi * np.exp(1j * y * Phi) * measure

        T_abs = np.zeros(p_max + 1)
        for p in range(p_max + 1):
            Ap = np.sum(hp_list[p] * weighted)
            dp = dim_rep(p, N)
            T_abs[p] = dp * abs(Ap) ** n_plaq

        # Dominant p >= 1
        dom_p = np.argmax(T_abs[1:]) + 1
        dominant_p.append(dom_p)

    print(f"\n  {'y':>6}  {'dominant_p':>10}  note")
    for i, (y, pp) in enumerate(zip(y_values, dominant_p)):
        if i % 4 == 0 or (i > 0 and dominant_p[i] != dominant_p[i-1]):
            note = ""
            if pp >= p_max - 5:
                note = "  (near ceiling)"
            print(f"  {y:6.2f}  {pp:10d}  {note}")

    # ==============================================================
    # Scaling analysis: restricted to reliable range
    # ==============================================================
    print(f"\n{'='*80}")
    print(f"  Scaling Analysis")
    print(f"{'='*80}")

    # Use crossing_p, restrict to p* in [1, p_max-10] and y >= 2
    reliable = [(y, pp) for y, pp in zip(y_values, crossing_p)
                if 1 <= pp <= p_max - 10 and y >= 2]

    if len(reliable) > 5:
        yr = np.array([r[0] for r in reliable])
        pr = np.array([r[1] for r in reliable], dtype=float)

        # Linear fit
        c_lin = np.polyfit(yr, pr, 1)
        rmse_lin = np.sqrt(np.mean((pr - np.polyval(c_lin, yr))**2))

        # Power law fit (p* = A * y^alpha)
        mask = pr > 0
        if np.sum(mask) > 3:
            c_pow = np.polyfit(np.log(yr[mask]), np.log(pr[mask]), 1)
            alpha = c_pow[0]
            A = np.exp(c_pow[1])
            rmse_pow = np.sqrt(np.mean((pr[mask] - A * yr[mask]**alpha)**2))

        print(f"\n  Crossing criterion (reliable range: y >= 2, p* <= {p_max-10}):")
        print(f"  N data points: {len(reliable)}")
        print(f"\n  Linear: p* = {c_lin[0]:.3f} y + ({c_lin[1]:.1f})")
        print(f"          RMSE = {rmse_lin:.2f}")
        if np.sum(mask) > 3:
            print(f"  Power:  p* = {A:.3f} y^{alpha:.3f}")
            print(f"          RMSE = {rmse_pow:.2f}")

    # Also fit the dominant_p
    reliable_d = [(y, pp) for y, pp in zip(y_values, dominant_p)
                  if 1 <= pp <= p_max - 10 and y >= 2]
    if len(reliable_d) > 5:
        yr_d = np.array([r[0] for r in reliable_d])
        pr_d = np.array([r[1] for r in reliable_d], dtype=float)
        c_lin_d = np.polyfit(yr_d, pr_d, 1)
        rmse_lin_d = np.sqrt(np.mean((pr_d - np.polyval(c_lin_d, yr_d))**2))
        print(f"\n  Dominant criterion (reliable range: y >= 2, p* <= {p_max-10}):")
        print(f"  N data points: {len(reliable_d)}")
        print(f"  Linear: p_dom = {c_lin_d[0]:.3f} y + ({c_lin_d[1]:.1f})")
        print(f"          RMSE = {rmse_lin_d:.2f}")

    elapsed = time.time() - t0
    print(f"\n  [Completed in {elapsed:.1f}s]")

    # ==============================================================
    # |A_p| ratio profile for the paper
    # ==============================================================
    print(f"\n{'='*80}")
    print(f"  |A_p(kappa+iy)| / |A_p(kappa)| — Complete Profile")
    print(f"{'='*80}")

    test_reps = [0, 1, 2, 3, 5, 8, 12, 18, 25]
    test_reps = [p for p in test_reps if p <= p_max]

    # A_p at real kappa
    Ap_real_dict = {}
    for p in test_reps:
        Ap_real_dict[p] = abs(np.sum(hp_list[p] * weighted_real))

    y_profile = np.arange(0.5, 16, 0.5)

    print(f"\n  {'y':>5}", end="")
    for p in test_reps:
        print(f"  {'p='+str(p):>9}", end="")
    print()

    for y in y_profile:
        weighted = exp_kPhi * np.exp(1j * y * Phi) * measure
        print(f"  {y:5.1f}", end="")
        for p in test_reps:
            Ap = abs(np.sum(hp_list[p] * weighted))
            ratio = Ap / Ap_real_dict[p] if Ap_real_dict[p] > 1e-30 else 0
            if ratio > 1e6:
                print(f"  {ratio:9.1e}", end="")
            else:
                print(f"  {ratio:9.3f}", end="")
        print()


if __name__ == '__main__':
    main()
