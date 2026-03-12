"""
SU(4) Fisher Zero 2D Newton Search
====================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 4b — exact numerical verification

Finds exact Fisher zeros of Z_{nP}^{SU(4)}(κ + iy) using 2D Newton's method
in the (κ, y) plane. Z is complex-valued (symmetric reps not self-conjugate),
so zeros require both Re Z = 0 and Im Z = 0 simultaneously.

Strategy:
1. Build Weyl quadrature grid once (expensive)
2. Coarse 2D scan of |Z(κ, y)| to locate minima
3. Newton refinement from each minimum
"""

import numpy as np
from math import comb, pi
import sys


# ---------------------------------------------------------------------------
# SU(N) representation theory
# ---------------------------------------------------------------------------

def dim_rep(p, N):
    """Dimension of SU(N) symmetric rep (p,0,...,0)."""
    return comb(p + N - 1, N - 1)


# ---------------------------------------------------------------------------
# Weyl quadrature
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
    Phi = np.sum(np.cos(theta_all), axis=1)
    measure = w * V2 / (2 * np.pi) ** dim
    norm = np.sum(measure).real
    measure = measure / norm
    return z, Phi, measure


# ---------------------------------------------------------------------------
# Fast Z evaluation
# ---------------------------------------------------------------------------

def setup_fast_eval(N, n_quad, n_reps):
    """Precompute grid and h_p values for fast Z evaluation."""
    z, Phi, meas = build_weyl_grid(N, n_quad)
    hp_list = [h_p_vec(p, z) for p in range(n_reps)]
    dims = [dim_rep(p, N) for p in range(n_reps)]
    return Phi, meas, hp_list, dims


def Z_eval(kap, y, n_plaq, Phi, meas, hp_list, dims, n_reps):
    """Evaluate Z_{nP}(κ + iy) efficiently."""
    exp_sPhi = np.exp(kap * Phi) * np.exp(1j * y * Phi)
    weighted = exp_sPhi * meas
    Z = 0j
    for p in range(n_reps):
        Ap = np.sum(hp_list[p] * weighted)
        Z += dims[p] * Ap ** n_plaq
    return Z


# ---------------------------------------------------------------------------
# 2D Newton's method
# ---------------------------------------------------------------------------

def newton_2d(kap0, y0, n_plaq, Phi, meas, hp_list, dims, n_reps,
              tol=1e-14, max_iter=50, dk=1e-6, dy=1e-6, damping=0.5):
    """2D Newton search for Z(κ+iy) = 0.

    Solves F(κ, y) = [Re Z, Im Z] = 0 using finite-difference Jacobian.
    """
    kap, y = kap0, y0

    for it in range(max_iter):
        Z0 = Z_eval(kap, y, n_plaq, Phi, meas, hp_list, dims, n_reps)
        absZ = abs(Z0)
        if absZ < tol:
            return kap, y, absZ, it, True

        # Jacobian via finite differences
        Zk = Z_eval(kap + dk, y, n_plaq, Phi, meas, hp_list, dims, n_reps)
        Zy = Z_eval(kap, y + dy, n_plaq, Phi, meas, hp_list, dims, n_reps)

        dRdk = (Zk.real - Z0.real) / dk
        dRdy = (Zy.real - Z0.real) / dy
        dIdk = (Zk.imag - Z0.imag) / dk
        dIdy = (Zy.imag - Z0.imag) / dy

        det = dRdk * dIdy - dRdy * dIdk
        if abs(det) < 1e-30:
            return kap, y, absZ, it, False

        # Newton step
        dkap = -(dIdy * Z0.real - dRdy * Z0.imag) / det
        ddy = -(-dIdk * Z0.real + dRdk * Z0.imag) / det

        # Damped step with line search
        alpha = 1.0
        for _ in range(10):
            kap_new = kap + alpha * dkap
            y_new = y + alpha * ddy
            if kap_new > 0 and y_new > 0:
                Z_new = Z_eval(kap_new, y_new, n_plaq, Phi, meas, hp_list,
                               dims, n_reps)
                if abs(Z_new) < absZ:
                    break
            alpha *= damping

        kap = kap + alpha * dkap
        y = y + alpha * ddy

        if kap <= 0 or y <= 0:
            return kap, y, absZ, it, False

    return kap, y, abs(Z_eval(kap, y, n_plaq, Phi, meas, hp_list, dims,
                               n_reps)), max_iter, False


# ---------------------------------------------------------------------------
# Coarse 2D scan
# ---------------------------------------------------------------------------

def coarse_scan(n_plaq, Phi, meas, hp_list, dims, n_reps,
                kap_range, y_range, n_kap=30, n_y=100):
    """Scan |Z(κ+iy)| on a grid and return local minima."""
    kaps = np.linspace(kap_range[0], kap_range[1], n_kap)
    ys = np.linspace(y_range[0], y_range[1], n_y)

    absZ = np.zeros((n_kap, n_y))
    for ik, kap in enumerate(kaps):
        for iy, y in enumerate(ys):
            Z = Z_eval(kap, y, n_plaq, Phi, meas, hp_list, dims, n_reps)
            absZ[ik, iy] = abs(Z)

    # Find local minima in 2D
    minima = []
    for ik in range(1, n_kap - 1):
        for iy in range(1, n_y - 1):
            val = absZ[ik, iy]
            neighbors = [absZ[ik-1, iy], absZ[ik+1, iy],
                         absZ[ik, iy-1], absZ[ik, iy+1]]
            if val < min(neighbors):
                # Normalize by Z at real axis
                Z_real = Z_eval(kaps[ik], 0.01, n_plaq, Phi, meas, hp_list,
                                dims, n_reps)
                ratio = val / abs(Z_real) if abs(Z_real) > 0 else val
                minima.append({
                    'kap': kaps[ik], 'y': ys[iy],
                    'absZ': val, 'ratio': ratio
                })

    minima.sort(key=lambda x: x['absZ'])
    return minima


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    N = 4
    n_quad = 40
    n_reps = 14

    print()
    print("=" * 90)
    print("  SU(4) Fisher Zero 2D Newton Search")
    print("  Author: Grzegorz Olbryk  |  March 2026  |  Task 4b")
    print("=" * 90)

    print(f"\n  Building Weyl grid: SU({N}), n_quad={n_quad} "
          f"({n_quad**(N-1)} pts), n_reps={n_reps}...")
    sys.stdout.flush()
    Phi, meas, hp_list, dims = setup_fast_eval(N, n_quad, n_reps)
    print("  Grid ready.")

    results = {}

    for n_plaq in [2, 3, 4]:
        print(f"\n\n  {'='*80}")
        print(f"  Z_{{{n_plaq}P}}^{{SU(4)}} — {n_plaq} plaquettes")
        print(f"  {'='*80}")

        # Phase 1: Coarse scan
        print(f"\n  Phase 1: Coarse 2D scan (κ ∈ [0.3, 3.0], y ∈ [0.5, 15])...")
        sys.stdout.flush()
        minima = coarse_scan(n_plaq, Phi, meas, hp_list, dims, n_reps,
                             kap_range=(0.3, 3.0), y_range=(0.5, 15.0),
                             n_kap=28, n_y=120)

        n_show = min(20, len(minima))
        print(f"\n  Top {n_show} |Z| minima from coarse scan:")
        for i, m in enumerate(minima[:n_show]):
            print(f"    #{i+1:2d}: κ = {m['kap']:.4f}  y = {m['y']:.4f}  "
                  f"|Z| = {m['absZ']:.4e}  ratio = {m['ratio']:.4e}")

        # Phase 2: Newton refinement of top minima
        print(f"\n  Phase 2: Newton refinement...")
        sys.stdout.flush()

        zeros_found = []
        for i, m in enumerate(minima[:20]):
            kap, y, absZ, iters, converged = newton_2d(
                m['kap'], m['y'], n_plaq, Phi, meas, hp_list, dims, n_reps,
                tol=1e-15, max_iter=100)

            if converged or absZ < 1e-10:
                # Check for duplicates
                is_dup = False
                for z in zeros_found:
                    if abs(z['kap'] - kap) < 1e-4 and abs(z['y'] - y) < 1e-4:
                        is_dup = True
                        break
                if not is_dup:
                    zeros_found.append({
                        'kap': kap, 'y': y, 'absZ': absZ,
                        'iters': iters, 'converged': converged
                    })
                    print(f"    ZERO #{len(zeros_found)}: "
                          f"s = {kap:.12f} + {y:.12f}i  "
                          f"|Z| = {absZ:.2e}  ({iters} iters)")

        zeros_found.sort(key=lambda x: x['y'])

        # Phase 3: Extended scan for larger y
        print(f"\n  Phase 3: Extended scan (y ∈ [12, 25])...")
        sys.stdout.flush()
        minima_ext = coarse_scan(n_plaq, Phi, meas, hp_list, dims, n_reps,
                                 kap_range=(0.3, 3.0), y_range=(12.0, 25.0),
                                 n_kap=20, n_y=80)

        for i, m in enumerate(minima_ext[:10]):
            if m['absZ'] < 1e-2:
                kap, y, absZ, iters, converged = newton_2d(
                    m['kap'], m['y'], n_plaq, Phi, meas, hp_list, dims, n_reps,
                    tol=1e-15, max_iter=100)
                if converged or absZ < 1e-10:
                    is_dup = any(abs(z['kap'] - kap) < 1e-4 and
                                abs(z['y'] - y) < 1e-4 for z in zeros_found)
                    if not is_dup:
                        zeros_found.append({
                            'kap': kap, 'y': y, 'absZ': absZ,
                            'iters': iters, 'converged': converged
                        })
                        print(f"    ZERO #{len(zeros_found)}: "
                              f"s = {kap:.12f} + {y:.12f}i  "
                              f"|Z| = {absZ:.2e}  ({iters} iters)")

        zeros_found.sort(key=lambda x: x['y'])

        # Summary
        print(f"\n  Summary: {len(zeros_found)} Fisher zeros found "
              f"for Z_{{{n_plaq}P}}^{{SU(4)}}")
        for i, z in enumerate(zeros_found):
            print(f"    Zero {i+1}: s = {z['kap']:.12f} + {z['y']:.12f}i  "
                  f"|Z| = {z['absZ']:.2e}")

        if len(zeros_found) >= 2:
            gaps = [zeros_found[i+1]['y'] - zeros_found[i]['y']
                    for i in range(len(zeros_found) - 1)]
            avg_gap = sum(gaps) / len(gaps)
            print(f"\n    Average y-gap: {avg_gap:.6f}")
            print(f"    π/(4n) = {pi/(4*n_plaq):.6f}")
            print(f"    Ratio avg_gap/(π/(4n)): {avg_gap/(pi/(4*n_plaq)):.4f}")

        results[n_plaq] = zeros_found

    # Convergence verification for key zeros
    print(f"\n\n  {'='*80}")
    print("  Convergence Verification")
    print(f"  {'='*80}")

    for n_plaq in [2, 3]:
        if results.get(n_plaq):
            z0 = results[n_plaq][0]
            print(f"\n  Z_{{{n_plaq}P}} zero at κ={z0['kap']:.6f}, "
                  f"y={z0['y']:.6f}:")
            for nq in [30, 40, 50]:
                Phi_t, meas_t, hp_t, dims_t = setup_fast_eval(N, nq, n_reps)
                Z = Z_eval(z0['kap'], z0['y'], n_plaq, Phi_t, meas_t,
                           hp_t, dims_t, n_reps)
                print(f"    n_quad={nq}: |Z| = {abs(Z):.4e}  "
                      f"Re Z = {Z.real:+.4e}  Im Z = {Z.imag:+.4e}")

    # Rep convergence for key zeros
    print(f"\n  Rep convergence (n_quad={n_quad}):")
    for n_plaq in [2, 3]:
        if results.get(n_plaq):
            z0 = results[n_plaq][0]
            print(f"\n  Z_{{{n_plaq}P}} zero at κ={z0['kap']:.6f}, "
                  f"y={z0['y']:.6f}:")
            for nr in [8, 10, 12, 14, 16]:
                Phi_t, meas_t, hp_t, dims_t = setup_fast_eval(N, n_quad, nr)
                Z = Z_eval(z0['kap'], z0['y'], n_plaq, Phi_t, meas_t,
                           hp_t, dims_t, nr)
                print(f"    n_reps={nr:2d}: |Z| = {abs(Z):.4e}  "
                      f"Re Z = {Z.real:+.4e}  Im Z = {Z.imag:+.4e}")

    # Final summary
    print(f"\n\n  {'='*80}")
    print("  FINAL SUMMARY")
    print(f"  {'='*80}")
    for n_plaq in [2, 3, 4]:
        nz = len(results.get(n_plaq, []))
        print(f"\n  Z_{{{n_plaq}P}}^{{SU(4)}}: {nz} Fisher zeros found")
        for i, z in enumerate(results.get(n_plaq, [])):
            print(f"    {i+1}. s = {z['kap']:.12f} + {z['y']:.12f}i  "
                  f"|Z| = {z['absZ']:.2e}")
    print(f"\n  RESULT_006 predicted: y_crit ≈ 3.02 for κ=1.0")
    print(f"  Predicted spacing: π/(4n)")
    print("=" * 90)


if __name__ == '__main__':
    main()
