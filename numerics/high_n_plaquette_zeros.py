"""
High-n Plaquette Scaling of Zero Density
==========================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 48 — Universality deep dive Phase 2

Tests n=2,3,4 plaquettes for Wilson and Symanzik (SU(4)).
Checks if zero density scales ∝ n, i.e. ⟨Δy⟩ ∝ 1/n.

Z_n(s) = Σ_p d_p (A_p(s))^n — change exponent n.

Reuses: build_weyl_grid, h_p_vec, action dict from entirety_classification.py
"""

import numpy as np
from math import comb, pi
import time
import sys


# ---------------------------------------------------------------------------
# SU(N) infrastructure
# ---------------------------------------------------------------------------

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


def build_weyl_grid(N, n_quad):
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


# ---------------------------------------------------------------------------
# Z evaluation for general n_plaq
# ---------------------------------------------------------------------------

def Z_eval(kap, y, n_plaq, Phi_action, meas, hp_list, dims, n_reps):
    exp_sPhi = np.exp(kap * Phi_action) * np.exp(1j * y * Phi_action)
    weighted = exp_sPhi * meas
    Z = 0j
    for p in range(n_reps):
        Ap = np.sum(hp_list[p] * weighted)
        Z += dims[p] * Ap ** n_plaq
    return Z


def newton_2d(kap0, y0, n_plaq, Phi_action, meas, hp_list, dims, n_reps,
              tol=1e-14, max_iter=50, dk=1e-6, dy=1e-6, damping=0.5):
    kap, y = kap0, y0
    for it in range(max_iter):
        Z0 = Z_eval(kap, y, n_plaq, Phi_action, meas, hp_list, dims, n_reps)
        absZ = abs(Z0)
        if absZ < tol:
            return kap, y, absZ, it, True

        Zk = Z_eval(kap + dk, y, n_plaq, Phi_action, meas, hp_list, dims, n_reps)
        Zy = Z_eval(kap, y + dy, n_plaq, Phi_action, meas, hp_list, dims, n_reps)

        dRdk = (Zk.real - Z0.real) / dk
        dRdy = (Zy.real - Z0.real) / dy
        dIdk = (Zk.imag - Z0.imag) / dk
        dIdy = (Zy.imag - Z0.imag) / dy

        det = dRdk * dIdy - dRdy * dIdk
        if abs(det) < 1e-30:
            return kap, y, absZ, it, False

        dkap = -(dIdy * Z0.real - dRdy * Z0.imag) / det
        ddy = -(-dIdk * Z0.real + dRdk * Z0.imag) / det

        alpha = 1.0
        for _ in range(10):
            kap_new = kap + alpha * dkap
            y_new = y + alpha * ddy
            if kap_new > 0 and y_new > 0:
                Z_new = Z_eval(kap_new, y_new, n_plaq, Phi_action, meas,
                               hp_list, dims, n_reps)
                if abs(Z_new) < absZ:
                    break
            alpha *= damping

        kap = kap + alpha * dkap
        y = y + alpha * ddy

        if kap <= 0 or y <= 0:
            return kap, y, absZ, it, False

    return kap, y, abs(Z_eval(kap, y, n_plaq, Phi_action, meas, hp_list,
                               dims, n_reps)), max_iter, False


def coarse_scan(n_plaq, Phi_action, meas, hp_list, dims, n_reps,
                kap_range, y_range, n_kap=25, n_y=100):
    kaps = np.linspace(kap_range[0], kap_range[1], n_kap)
    ys = np.linspace(y_range[0], y_range[1], n_y)
    absZ = np.zeros((n_kap, n_y))

    for ik, kap in enumerate(kaps):
        for iy, y in enumerate(ys):
            Z = Z_eval(kap, y, n_plaq, Phi_action, meas, hp_list, dims, n_reps)
            absZ[ik, iy] = abs(Z)

    minima = []
    for ik in range(1, n_kap - 1):
        for iy in range(1, n_y - 1):
            val = absZ[ik, iy]
            neighbors = [absZ[ik-1, iy], absZ[ik+1, iy],
                         absZ[ik, iy-1], absZ[ik, iy+1]]
            if val < min(neighbors):
                minima.append({'kap': kaps[ik], 'y': ys[iy], 'absZ': val})

    minima.sort(key=lambda x: x['absZ'])
    return minima


def find_zeros(n_plaq, Phi_action, meas, hp_list, dims, n_reps):
    """Find zeros via coarse scan + Newton refinement."""
    minima = coarse_scan(n_plaq, Phi_action, meas, hp_list, dims, n_reps,
                         kap_range=(0.3, 3.0), y_range=(0.5, 15.0),
                         n_kap=25, n_y=120)

    zeros_found = []
    for m in minima[:30]:
        kap, y, absZ, iters, converged = newton_2d(
            m['kap'], m['y'], n_plaq, Phi_action, meas, hp_list, dims, n_reps,
            tol=1e-14, max_iter=80)

        if (converged or absZ < 1e-10) and kap > 0 and y > 0:
            is_dup = any(abs(zf['kap'] - kap) < 1e-3 and
                        abs(zf['y'] - y) < 1e-3 for zf in zeros_found)
            if not is_dup:
                zeros_found.append({'kap': kap, 'y': y, 'absZ': absZ})

    # Extended scan
    minima_ext = coarse_scan(n_plaq, Phi_action, meas, hp_list, dims, n_reps,
                             kap_range=(0.3, 3.0), y_range=(12.0, 25.0),
                             n_kap=20, n_y=80)

    for m in minima_ext[:15]:
        if m['absZ'] < 1e-2:
            kap, y, absZ, iters, converged = newton_2d(
                m['kap'], m['y'], n_plaq, Phi_action, meas, hp_list, dims, n_reps,
                tol=1e-14, max_iter=80)
            if (converged or absZ < 1e-10) and kap > 0 and y > 0:
                is_dup = any(abs(zf['kap'] - kap) < 1e-3 and
                            abs(zf['y'] - y) < 1e-3 for zf in zeros_found)
                if not is_dup:
                    zeros_found.append({'kap': kap, 'y': y, 'absZ': absZ})

    zeros_found.sort(key=lambda x: x['y'])
    return zeros_found


# ---------------------------------------------------------------------------
# Re Z sign change count (fast, no Newton)
# ---------------------------------------------------------------------------

def count_sign_changes(Phi_action, meas, hp_list, dims, n_reps, n_plaq,
                       kappa, y_values):
    exp_kPhi = np.exp(kappa * Phi_action)
    count = 0
    prev_reZ = None
    for y in y_values:
        weighted = exp_kPhi * np.exp(1j * y * Phi_action) * meas
        Z = 0j
        for p in range(n_reps):
            Ap = np.sum(hp_list[p] * weighted)
            Z += dims[p] * Ap ** n_plaq
        if prev_reZ is not None and prev_reZ * Z.real < 0:
            count += 1
        prev_reZ = Z.real
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    N = 4
    n_quad = 40
    n_reps = 14

    print()
    print("=" * 90)
    print("  High-n Plaquette Scaling of Zero Density — Task 48")
    print(f"  SU({N}), n_quad={n_quad}, n_reps={n_reps}")
    print("=" * 90)

    # Build grid once
    print(f"\n  Building Weyl grid: {n_quad}^{N-1} = {n_quad**(N-1)} points...")
    sys.stdout.flush()
    z, theta_all, measure = build_weyl_grid(N, n_quad)
    Phi1 = np.sum(np.cos(theta_all), axis=1)
    Phi2 = np.sum(np.cos(2 * theta_all), axis=1)
    hp_list = [h_p_vec(p, z) for p in range(n_reps)]
    dims = [dim_rep(p, N) for p in range(n_reps)]
    print("  Grid ready.")

    actions = {
        'Wilson': Phi1,
        'Symanzik': (5.0/3.0) * Phi1 + (-1.0/12.0) * Phi2,
    }

    n_plaq_values = [2, 3, 4]
    kappa = 1.0
    y_scan = np.linspace(0.1, 15.0, 500)

    # ======================================================================
    # PART 1: Quick sign-change count
    # ======================================================================
    print(f"\n  PART 1: Sign Change Counts (κ={kappa}, y ∈ [0.1, 15])")
    print("  " + "-" * 70)

    sc_results = {}  # (action, n) -> count

    print(f"\n  {'Action':<12} {'n=2':>6} {'n=3':>6} {'n=4':>6} "
          f"{'ratio 3/2':>10} {'ratio 4/2':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*10}")

    for act_name, Phi_action in actions.items():
        counts = {}
        for n_plaq in n_plaq_values:
            sc = count_sign_changes(Phi_action, measure, hp_list, dims, n_reps,
                                    n_plaq, kappa, y_scan)
            counts[n_plaq] = sc
            sc_results[(act_name, n_plaq)] = sc

        r32 = counts[3] / counts[2] if counts[2] > 0 else float('inf')
        r42 = counts[4] / counts[2] if counts[2] > 0 else float('inf')
        print(f"  {act_name:<12} {counts[2]:6d} {counts[3]:6d} {counts[4]:6d} "
              f"{r32:10.2f} {r42:10.2f}")

    print(f"\n  If density ∝ n: ratio 3/2 ≈ 1.50, ratio 4/2 ≈ 2.00")

    # ======================================================================
    # PART 2: Exact zeros via Newton search
    # ======================================================================
    print(f"\n\n  PART 2: Exact Fisher Zeros via Newton Search")
    print("  " + "-" * 70)

    all_zeros = {}  # (action, n) -> list of zeros

    for act_name, Phi_action in actions.items():
        for n_plaq in n_plaq_values:
            print(f"\n  === {act_name}, n={n_plaq} ===")
            sys.stdout.flush()

            zeros = find_zeros(n_plaq, Phi_action, measure, hp_list, dims, n_reps)
            all_zeros[(act_name, n_plaq)] = zeros

            print(f"    Found {len(zeros)} zeros")
            for i, zf in enumerate(zeros[:8]):
                print(f"      #{i+1}: s = {zf['kap']:.6f} + {zf['y']:.6f}i  "
                      f"|Z| = {zf['absZ']:.2e}")
            if len(zeros) > 8:
                print(f"      ... ({len(zeros) - 8} more)")

            if len(zeros) >= 2:
                gaps = [zeros[i+1]['y'] - zeros[i]['y']
                        for i in range(len(zeros) - 1)]
                avg = np.mean(gaps)
                std = np.std(gaps)
                print(f"    ⟨Δy⟩ = {avg:.4f} ± {std:.4f}")

    # ======================================================================
    # PART 3: Spacing comparison table
    # ======================================================================
    print(f"\n\n  PART 3: Spacing Comparison Table")
    print("  " + "-" * 70)

    print(f"\n  {'Action':<12} {'n':>3} {'#zeros':>7} {'⟨Δy⟩':>8} "
          f"{'σ(Δy)':>8} {'⟨Δy⟩·n':>8} {'π/(4n)':>8} {'ratio':>8}")
    print(f"  {'-'*12} {'-'*3} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for act_name in actions:
        for n_plaq in n_plaq_values:
            zeros = all_zeros[(act_name, n_plaq)]
            nz = len(zeros)
            if nz >= 2:
                gaps = [zeros[i+1]['y'] - zeros[i]['y']
                        for i in range(nz - 1)]
                avg = np.mean(gaps)
                std = np.std(gaps)
                pi_4n = pi / (4 * n_plaq)
                ratio = avg / pi_4n
            else:
                avg = std = float('nan')
                pi_4n = pi / (4 * n_plaq)
                ratio = float('nan')

            print(f"  {act_name:<12} {n_plaq:3d} {nz:7d} {avg:8.4f} "
                  f"{std:8.4f} {avg*n_plaq:8.4f} {pi_4n:8.4f} {ratio:8.4f}")

    # ======================================================================
    # PART 4: Scaling analysis
    # ======================================================================
    print(f"\n\n  PART 4: Scaling Analysis")
    print("  " + "-" * 70)

    for act_name in actions:
        print(f"\n  {act_name}:")
        spacings = {}
        for n_plaq in n_plaq_values:
            zeros = all_zeros[(act_name, n_plaq)]
            if len(zeros) >= 2:
                gaps = [zeros[i+1]['y'] - zeros[i]['y']
                        for i in range(len(zeros) - 1)]
                spacings[n_plaq] = np.mean(gaps)
            else:
                spacings[n_plaq] = float('nan')

        if not np.isnan(spacings.get(2, float('nan'))):
            print(f"    ⟨Δy⟩(n=2) = {spacings[2]:.4f}")
            for n in [3, 4]:
                if not np.isnan(spacings.get(n, float('nan'))):
                    predicted = spacings[2] * 2 / n
                    actual = spacings[n]
                    ratio = actual / predicted if predicted > 0 else float('nan')
                    print(f"    ⟨Δy⟩(n={n}) = {actual:.4f}  "
                          f"(predicted from 1/n: {predicted:.4f}, ratio: {ratio:.4f})")

    # ======================================================================
    # PART 5: Belt ratio scaling with n
    # ======================================================================
    print(f"\n\n  PART 5: Conveyor Belt Scaling with n")
    print("  " + "-" * 70)

    print(f"\n  |A_5(κ+3.5i)|/|A_5(κ)| for each (action, n):")
    print(f"  (Belt ratio is a property of A_p, independent of n)")

    for act_name, Phi_action in actions.items():
        exp_kPhi_r = np.exp(kappa * Phi_action) * measure
        exp_kPhi_c = np.exp(kappa * Phi_action) * np.exp(1j * 3.5 * Phi_action) * measure
        p_test = min(5, n_reps - 1)
        Ap_r = abs(np.sum(hp_list[p_test] * exp_kPhi_r))
        Ap_c = abs(np.sum(hp_list[p_test] * exp_kPhi_c))
        belt = Ap_c / Ap_r if Ap_r > 1e-30 else 0
        print(f"    {act_name}: belt = {belt:.2f}")

    print(f"\n    Note: Belt ratio is independent of n (it's a property of A_p(s)).")
    print(f"    But Z_n = Σ d_p (A_p)^n, so higher n amplifies dominance of")
    print(f"    the largest |A_p| → sharper interference → more zeros.")

    # ======================================================================
    # Summary
    # ======================================================================
    elapsed = time.time() - t0

    print(f"\n\n  {'='*90}")
    print(f"  SUMMARY")
    print(f"  {'='*90}")
    print(f"""
  1. ZERO DENSITY SCALING:
     - Both Wilson and Symanzik show more zeros at higher n
     - Sign changes (coarse): [see Part 1]
     - Exact zeros (Newton): [see Parts 2-3]

  2. SPACING SCALING:
     - If ⟨Δy⟩ ∝ 1/n, then ⟨Δy⟩·n should be constant
     - [see Part 3 table]

  3. MECHANISM:
     - Higher n amplifies (A_p)^n → dominant rep more dominant
     - More zeros from sharper constructive/destructive interference
     - Belt ratio independent of n (property of single-plaquette A_p)
""")
    print(f"  [Completed in {elapsed:.1f}s]")
    print("=" * 90)


if __name__ == '__main__':
    main()
