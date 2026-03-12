"""
Large-N Scaling of Fisher Zeros
================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: Task 20 — large-N analysis of conveyor belt mechanism

Studies:
1. SU(8) Fisher zeros (N ≡ 0 mod 4): confirm conveyor belt
2. N-dependence of zero spacing for N ≡ 0 mod 4
3. α_N(κ) convergence for N odd
"""

import numpy as np
from math import comb, pi
import sys


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
    Phi = np.sum(np.cos(theta_all), axis=1)
    measure = w * V2 / (2 * np.pi) ** dim
    norm = np.sum(measure).real
    measure = measure / norm
    return z, Phi, measure


def setup_fast_eval(N, n_quad, n_reps):
    z, Phi, meas = build_weyl_grid(N, n_quad)
    hp_list = [h_p_vec(p, z) for p in range(n_reps)]
    dims = [dim_rep(p, N) for p in range(n_reps)]
    return Phi, meas, hp_list, dims


def Z_eval(kap, y, n_plaq, Phi, meas, hp_list, dims, n_reps):
    exp_sPhi = np.exp(kap * Phi) * np.exp(1j * y * Phi)
    weighted = exp_sPhi * meas
    Z = 0j
    for p in range(n_reps):
        Ap = np.sum(hp_list[p] * weighted)
        Z += dims[p] * Ap ** n_plaq
    return Z


def newton_2d(kap0, y0, n_plaq, Phi, meas, hp_list, dims, n_reps,
              tol=1e-12, max_iter=50, dk=1e-6, dy=1e-6, damping=0.5):
    kap, y = kap0, y0
    for it in range(max_iter):
        Z0 = Z_eval(kap, y, n_plaq, Phi, meas, hp_list, dims, n_reps)
        absZ = abs(Z0)
        if absZ < tol:
            return kap, y, absZ, it, True
        Zk = Z_eval(kap + dk, y, n_plaq, Phi, meas, hp_list, dims, n_reps)
        Zy = Z_eval(kap, y + dy, n_plaq, Phi, meas, hp_list, dims, n_reps)
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


def coarse_1d_scan(kap, n_plaq, y_vals, Phi, meas, hp_list, dims, n_reps):
    """Scan |Z(κ+iy)| along a fixed κ line."""
    absZ = np.zeros(len(y_vals))
    for iy, y in enumerate(y_vals):
        Z = Z_eval(kap, y, n_plaq, Phi, meas, hp_list, dims, n_reps)
        absZ[iy] = abs(Z)
    # Find local minima
    minima = []
    for i in range(1, len(absZ) - 1):
        if absZ[i] < absZ[i-1] and absZ[i] < absZ[i+1]:
            minima.append({'y': y_vals[i], 'absZ': absZ[i]})
    minima.sort(key=lambda x: x['absZ'])
    return minima, absZ


def main():
    print()
    print("=" * 90)
    print("  Large-N Scaling of Fisher Zeros")
    print("  Author: Grzegorz Olbryk  |  March 2026")
    print("=" * 90)

    # ==================================================================
    # Part 1: SU(8) Fisher zeros (N ≡ 0 mod 4)
    # ==================================================================
    N = 8
    n_quad = 8  # 8^7 = 2,097,152 points
    n_reps = 10
    n_plaq = 2

    print(f"\n  PART 1: SU({N}) Fisher Zeros (N ≡ 0 mod 4)")
    print("  " + "-" * 70)
    print(f"  Grid: n_quad={n_quad} ({n_quad**(N-1):,d} pts), n_reps={n_reps}")
    print(f"  Building Weyl grid...", flush=True)
    Phi, meas, hp_list, dims = setup_fast_eval(N, n_quad, n_reps)
    print(f"  Grid ready.")

    # 1D scan at κ = 1.0 to find minima
    kap = 1.0
    y_vals = np.linspace(0.5, 10.0, 200)
    print(f"\n  Scanning |Z_{{{n_plaq}P}}(1.0 + iy)| for y ∈ [0.5, 10]...",
          flush=True)
    minima, absZ = coarse_1d_scan(kap, n_plaq, y_vals, Phi, meas,
                                   hp_list, dims, n_reps)

    print(f"\n  Top |Z| minima at κ={kap}:")
    for i, m in enumerate(minima[:15]):
        print(f"    #{i+1}: y = {m['y']:.4f}  |Z| = {m['absZ']:.4e}")

    # Newton refinement for deepest minima (with κ as free parameter)
    print(f"\n  Newton refinement (2D in κ, y)...")
    zeros = []
    for m in minima[:10]:
        kap_r, y_r, absZ_r, iters, conv = newton_2d(
            kap, m['y'], n_plaq, Phi, meas, hp_list, dims, n_reps,
            tol=1e-12, max_iter=60)
        if conv or absZ_r < 1e-8:
            is_dup = any(abs(z['y'] - y_r) < 0.1 for z in zeros)
            if not is_dup and kap_r > 0 and y_r > 0:
                zeros.append({'kap': kap_r, 'y': y_r, 'absZ': absZ_r})
                print(f"    ZERO: s = {kap_r:.8f} + {y_r:.8f}i  "
                      f"|Z| = {absZ_r:.2e}  ({iters} iters)")

    # Also scan κ = 0.5 and κ = 2.0
    for kap_scan in [0.5, 2.0]:
        print(f"\n  Scanning κ={kap_scan}...", flush=True)
        minima_k, _ = coarse_1d_scan(kap_scan, n_plaq, y_vals, Phi, meas,
                                      hp_list, dims, n_reps)
        for m in minima_k[:5]:
            kap_r, y_r, absZ_r, iters, conv = newton_2d(
                kap_scan, m['y'], n_plaq, Phi, meas, hp_list, dims, n_reps,
                tol=1e-12, max_iter=60)
            if (conv or absZ_r < 1e-8) and kap_r > 0 and y_r > 0:
                is_dup = any(abs(z['y'] - y_r) < 0.1 and
                             abs(z['kap'] - kap_r) < 0.1 for z in zeros)
                if not is_dup:
                    zeros.append({'kap': kap_r, 'y': y_r, 'absZ': absZ_r})
                    print(f"    ZERO: s = {kap_r:.8f} + {y_r:.8f}i  "
                          f"|Z| = {absZ_r:.2e}  ({iters} iters)")

    zeros.sort(key=lambda x: x['y'])
    print(f"\n  SU({N}), n={n_plaq}: {len(zeros)} Fisher zeros found")
    for i, z in enumerate(zeros):
        print(f"    {i+1}. s = {z['kap']:.8f} + {z['y']:.8f}i  "
              f"|Z| = {z['absZ']:.2e}")

    if len(zeros) >= 2:
        gaps = [zeros[i+1]['y'] - zeros[i]['y']
                for i in range(len(zeros) - 1)]
        print(f"\n    Gaps: {[f'{g:.3f}' for g in gaps]}")
        print(f"    Mean gap: {np.mean(gaps):.4f}")

    # n=3 as well
    n_plaq = 3
    print(f"\n\n  SU({N}), n={n_plaq}:")
    kap = 1.0
    y_vals = np.linspace(0.5, 10.0, 200)
    print(f"  Scanning...", flush=True)
    minima3, _ = coarse_1d_scan(kap, n_plaq, y_vals, Phi, meas,
                                 hp_list, dims, n_reps)

    print(f"  Top minima:")
    for i, m in enumerate(minima3[:10]):
        print(f"    #{i+1}: y = {m['y']:.4f}  |Z| = {m['absZ']:.4e}")

    zeros3 = []
    for m in minima3[:10]:
        kap_r, y_r, absZ_r, iters, conv = newton_2d(
            kap, m['y'], n_plaq, Phi, meas, hp_list, dims, n_reps,
            tol=1e-12, max_iter=60)
        if (conv or absZ_r < 1e-8) and kap_r > 0 and y_r > 0:
            is_dup = any(abs(z['y'] - y_r) < 0.1 for z in zeros3)
            if not is_dup:
                zeros3.append({'kap': kap_r, 'y': y_r, 'absZ': absZ_r})
                print(f"    ZERO: s = {kap_r:.8f} + {y_r:.8f}i  "
                      f"|Z| = {absZ_r:.2e}  ({iters} iters)")

    zeros3.sort(key=lambda x: x['y'])
    if len(zeros3) >= 2:
        gaps3 = [zeros3[i+1]['y'] - zeros3[i]['y']
                 for i in range(len(zeros3) - 1)]
        print(f"\n    Gaps: {[f'{g:.3f}' for g in gaps3]}")
        print(f"    Mean gap: {np.mean(gaps3):.4f}")

    # ==================================================================
    # Part 2: N-dependence comparison
    # ==================================================================
    print(f"\n\n  PART 2: N-Dependence of Zero Spacing")
    print("  " + "-" * 70)

    # SU(4) comparison data from RESULT_015
    su4_n2_gap = 1.25
    su4_n3_gap = 1.22  # main branch
    su4_n4_gap = 0.61

    print(f"\n  SU(4) (from RESULT_015):")
    print(f"    n=2: mean gap ≈ {su4_n2_gap:.2f}")
    print(f"    n=3 main: mean gap ≈ {su4_n3_gap:.2f}")
    print(f"    n=4: mean gap ≈ {su4_n4_gap:.2f}")

    if len(zeros) >= 2:
        su8_n2_gap = np.mean([zeros[i+1]['y'] - zeros[i]['y']
                              for i in range(len(zeros) - 1)])
        print(f"\n  SU(8):")
        print(f"    n=2: mean gap ≈ {su8_n2_gap:.2f}")
    if len(zeros3) >= 2:
        su8_n3_gap = np.mean([zeros3[i+1]['y'] - zeros3[i]['y']
                              for i in range(len(zeros3) - 1)])
        print(f"    n=3: mean gap ≈ {su8_n3_gap:.2f}")

    # ==================================================================
    # Part 3: Convergence check for SU(8)
    # ==================================================================
    if zeros:
        print(f"\n\n  PART 3: Convergence Check (SU(8))")
        print("  " + "-" * 70)
        z0 = zeros[0]
        for nq in [6, 8, 10]:
            print(f"  n_quad={nq} ({nq**(N-1):,d} pts)...", flush=True)
            P, m, hp, d = setup_fast_eval(N, nq, n_reps)
            Z = Z_eval(z0['kap'], z0['y'], 2, P, m, hp, d, n_reps)
            print(f"    |Z| = {abs(Z):.4e}  Re Z = {Z.real:+.4e}  "
                  f"Im Z = {Z.imag:+.4e}")

    # Summary
    print(f"\n\n  {'='*80}")
    print("  SUMMARY")
    print(f"  {'='*80}")
    print(f"  SU(4): zeros confirmed with machine precision (RESULT_015)")
    print(f"  SU(8): zeros {'confirmed' if len(zeros) > 0 else 'NOT found'}")
    if len(zeros) >= 2 and len(zeros3) >= 2:
        print(f"\n  N-scaling of zero spacing (n=2):")
        print(f"    SU(4): {su4_n2_gap:.2f}")
        print(f"    SU(8): {su8_n2_gap:.2f}")
        print(f"    Ratio SU(8)/SU(4): {su8_n2_gap/su4_n2_gap:.3f}")
    print("=" * 90)


if __name__ == '__main__':
    main()
