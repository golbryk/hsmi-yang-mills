"""
Large-n Asymptotic Law: Do Zeros Converge to Stokes Lines?
============================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: Paper Psi — key test for asymptotic theorem

THE critical question: as n → ∞, does distance(zero, |A_p|=|A_q|) → 0?

Part 1: SU(4) Wilson/Symanzik at n = 2,3,4,5,6,8,10,15,20,30,50
Part 2: 1D Ising model (universality test)
Part 3: Overlay data for killer figure

If distance/⟨Δy⟩ → 0 as n → ∞ → ASYMPTOTIC THEOREM.
If Ising works → UNIVERSAL MECHANISM.
"""

import numpy as np
from math import comb, pi, factorial
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
# 1D Ising transfer matrix
# ---------------------------------------------------------------------------

def ising_transfer_eigenvalues(beta_complex):
    """Transfer matrix eigenvalues for 1D Ising model.

    T = [[exp(β), exp(-β)], [exp(-β), exp(β)]]
    eigenvalues: λ_+ = 2 cosh(β), λ_- = 2 sinh(β)
    """
    lam_plus = 2 * np.cosh(beta_complex)
    lam_minus = 2 * np.sinh(beta_complex)
    return lam_plus, lam_minus


def ising_Zn(beta_complex, n):
    """Z_n = λ_+^n + λ_-^n for 1D Ising."""
    lp, lm = ising_transfer_eigenvalues(beta_complex)
    return lp ** n + lm ** n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print()
    print("=" * 90)
    print("  Large-n Asymptotic Law: Do Zeros Converge to Stokes Lines?")
    print("=" * 90)

    # ======================================================================
    # PART 1: SU(4) Wilson at n = 2..50
    # ======================================================================
    print(f"\n  PART 1: SU(4) Wilson — Distance to Stokes Lines vs n")
    print("  " + "-" * 70)

    N = 4
    n_quad = 40
    n_reps = 14
    kappa = 1.0

    print(f"  Building SU({N}) Weyl grid...", end=" ", flush=True)
    z, theta_all, measure = build_weyl_grid(N, n_quad)
    Phi1 = np.sum(np.cos(theta_all), axis=1)
    hp_list = [h_p_vec(p, z) for p in range(n_reps)]
    dims = [dim_rep(p, N) for p in range(n_reps)]
    print("Done.")

    # Compute A_p along dense y scan
    y_dense = np.linspace(0.01, 25.0, 10000)
    exp_kPhi = np.exp(kappa * Phi1)
    print("  Computing A_p(κ+iy) on dense grid...", end=" ", flush=True)
    sys.stdout.flush()
    Ap = np.zeros((n_reps, len(y_dense)), dtype=complex)
    for iy, y in enumerate(y_dense):
        weighted = exp_kPhi * np.exp(1j * y * Phi1) * measure
        for p in range(n_reps):
            Ap[p, iy] = np.sum(hp_list[p] * weighted)
    print("Done.")

    abs_Ap = np.abs(Ap)

    # Find all |A_p|=|A_q| crossings
    all_crossings_y = []
    crossing_pairs = []
    for p in range(min(10, n_reps)):
        for q in range(p + 1, min(10, n_reps)):
            diff = abs_Ap[p] - abs_Ap[q]
            for i in range(len(diff) - 1):
                if diff[i] * diff[i + 1] < 0:
                    frac = abs(diff[i]) / (abs(diff[i]) + abs(diff[i + 1]))
                    yc = y_dense[i] + frac * (y_dense[i + 1] - y_dense[i])
                    all_crossings_y.append(yc)
                    crossing_pairs.append((p, q))
    all_crossings_y = np.array(all_crossings_y)
    print(f"  |A_p|=|A_q| crossings: {len(all_crossings_y)}")

    # Test at various n
    n_values = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]

    print(f"\n  {'n':>4} {'#zeros':>7} {'⟨Δy⟩':>8} {'⟨d⟩':>8} "
          f"{'⟨d⟩/⟨Δy⟩':>10} {'med_d':>8} {'med/⟨Δy⟩':>10} "
          f"{'f<Δy/4':>7} {'2-term err':>10}")
    print(f"  {'-'*4} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*8} "
          f"{'-'*10} {'-'*7} {'-'*10}")

    results = []
    for n_plaq in n_values:
        # Compute Z_n and terms
        Z = np.zeros(len(y_dense), dtype=complex)
        terms = np.zeros((n_reps, len(y_dense)), dtype=complex)
        for p in range(n_reps):
            terms[p] = dims[p] * Ap[p] ** n_plaq
            Z += terms[p]

        # Find zeros
        zeros = []
        zeros_idx = []
        for i in range(len(Z) - 1):
            if Z[i].real * Z[i + 1].real < 0:
                frac = abs(Z[i].real) / (abs(Z[i].real) + abs(Z[i + 1].real))
                zeros.append(y_dense[i] + frac *
                             (y_dense[i + 1] - y_dense[i]))
                zeros_idx.append(i)

        if len(zeros) < 2:
            print(f"  {n_plaq:4d} {len(zeros):7d}      —")
            continue

        gaps = np.diff(zeros)
        avg_gap = np.mean(gaps)

        # Distance to nearest crossing
        distances = []
        for yz in zeros:
            d = np.min(np.abs(all_crossings_y - yz))
            distances.append(d)
        darr = np.array(distances)
        avg_d = np.mean(darr)
        med_d = np.median(darr)
        frac_close = np.sum(darr < avg_gap / 4) / len(darr)

        # 2-term error
        errs = []
        for iz in zeros_idx:
            T_mag = np.abs(terms[:, iz])
            sorted_p = np.argsort(-T_mag)
            p1, p2 = sorted_p[0], sorted_p[1]
            Z_2 = terms[p1, iz] + terms[p2, iz]
            scale = T_mag[p1]
            if scale > 0:
                errs.append(abs(Z[iz] - Z_2) / scale)
        med_err = np.median(errs) if errs else float('nan')

        results.append({
            'n': n_plaq, 'nz': len(zeros), 'avg_gap': avg_gap,
            'avg_d': avg_d, 'med_d': med_d,
            'ratio': avg_d / avg_gap, 'med_ratio': med_d / avg_gap,
            'frac': frac_close, 'err': med_err
        })
        print(f"  {n_plaq:4d} {len(zeros):7d} {avg_gap:8.4f} {avg_d:8.4f} "
              f"{avg_d/avg_gap:10.4f} {med_d:8.4f} {med_d/avg_gap:10.4f} "
              f"{frac_close:7.3f} {med_err:10.4f}")

    # Fit: does med_d/⟨Δy⟩ → 0?
    if len(results) >= 4:
        ns = np.array([r['n'] for r in results])
        med_ratios = np.array([r['med_ratio'] for r in results])
        errs = np.array([r['err'] for r in results])

        # Fit med_d/⟨Δy⟩ = a/n^α
        log_n = np.log(ns)
        log_r = np.log(med_ratios + 1e-10)
        valid = med_ratios > 0
        if np.sum(valid) >= 3:
            A = np.column_stack([np.ones(np.sum(valid)), log_n[valid]])
            coeffs = np.linalg.lstsq(A, log_r[valid], rcond=None)[0]
            a_fit = np.exp(coeffs[0])
            alpha = coeffs[1]
            print(f"\n  Power-law fit: med_d/⟨Δy⟩ ≈ {a_fit:.4f} · n^{alpha:.4f}")
            if alpha < -0.1:
                print(f"  → med_d/⟨Δy⟩ → 0 as n → ∞ (α = {alpha:.3f})")
            else:
                print(f"  → med_d/⟨Δy⟩ does NOT vanish (α = {alpha:.3f})")

        # Also fit 2-term error
        log_e = np.log(errs + 1e-10)
        valid_e = errs > 0
        if np.sum(valid_e) >= 3:
            A = np.column_stack([np.ones(np.sum(valid_e)),
                                 log_n[valid_e]])
            coeffs = np.linalg.lstsq(A, log_e[valid_e], rcond=None)[0]
            a_e = np.exp(coeffs[0])
            alpha_e = coeffs[1]
            print(f"  2-term error fit: error ≈ {a_e:.4f} · n^{alpha_e:.4f}")
            if alpha_e < -0.5:
                print(f"  → Error → 0 as n → ∞ (α = {alpha_e:.3f}) "
                      f"→ DOMINANT-PAIR APPROXIMATION VALID")

    # ======================================================================
    # PART 2: 1D Ising Model — Universality Test
    # ======================================================================
    print(f"\n\n  {'='*90}")
    print(f"  PART 2: 1D Ising Model — Universality Test")
    print("  " + "-" * 70)
    print(f"  Z_n(β) = λ_+(β)^n + λ_-(β)^n")
    print(f"  λ_+ = 2cosh(β), λ_- = 2sinh(β)")
    print(f"  Stokes line: |λ_+| = |λ_-|, i.e., |cosh(β)| = |sinh(β)|")

    # For β = β_r + iy, find Fisher zeros and Stokes lines
    beta_r = 0.5  # real part of coupling

    y_ising = np.linspace(0.01, 20.0, 20000)
    beta_vals = beta_r + 1j * y_ising

    # Stokes line: |cosh(β)| = |sinh(β)|
    cosh_abs = np.abs(np.cosh(beta_vals))
    sinh_abs = np.abs(np.sinh(beta_vals))
    stokes_diff = cosh_abs - sinh_abs
    stokes_y = []
    for i in range(len(stokes_diff) - 1):
        if stokes_diff[i] * stokes_diff[i + 1] < 0:
            frac = abs(stokes_diff[i]) / (abs(stokes_diff[i]) +
                                           abs(stokes_diff[i + 1]))
            stokes_y.append(y_ising[i] + frac *
                            (y_ising[i + 1] - y_ising[i]))
    stokes_y = np.array(stokes_y)
    print(f"\n  Stokes crossings (|λ_+|=|λ_-|): {len(stokes_y)}")
    if len(stokes_y) > 0:
        print(f"  First few: {', '.join(f'{y:.4f}' for y in stokes_y[:5])}")

    # Analytic: |cosh(β_r + iy)| = |sinh(β_r + iy)|
    # |cosh|² = cosh²(β_r)cos²(y) + sinh²(β_r)sin²(y)
    # |sinh|² = sinh²(β_r)cos²(y) + cosh²(β_r)sin²(y)
    # Equal when: (cosh²-sinh²)(cos²y - sin²y) = 0
    # → cos(2y) = 0 → y = π/4 + kπ/2
    analytic_stokes = [pi / 4 + k * pi / 2 for k in range(15)
                       if pi / 4 + k * pi / 2 < 20]
    print(f"  Analytic: y = π/4 + kπ/2 = "
          f"{', '.join(f'{y:.4f}' for y in analytic_stokes[:5])}, ...")

    n_ising_values = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100]

    print(f"\n  {'n':>4} {'#zeros':>7} {'⟨Δy⟩':>8} {'⟨d⟩':>8} "
          f"{'⟨d⟩/⟨Δy⟩':>10} {'med_d':>8} {'med/⟨Δy⟩':>10} "
          f"{'f<Δy/4':>7}")
    print(f"  {'-'*4} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*8} "
          f"{'-'*10} {'-'*7}")

    ising_results = []
    for n_plaq in n_ising_values:
        Z = ising_Zn(beta_vals, n_plaq)

        zeros = []
        for i in range(len(Z) - 1):
            if Z[i].real * Z[i + 1].real < 0:
                frac = abs(Z[i].real) / (abs(Z[i].real) + abs(Z[i + 1].real))
                zeros.append(y_ising[i] + frac *
                             (y_ising[i + 1] - y_ising[i]))

        if len(zeros) < 2:
            print(f"  {n_plaq:4d} {len(zeros):7d}      —")
            continue

        gaps = np.diff(zeros)
        avg_gap = np.mean(gaps)

        distances = []
        for yz in zeros:
            d = np.min(np.abs(stokes_y - yz))
            distances.append(d)
        darr = np.array(distances)
        avg_d = np.mean(darr)
        med_d = np.median(darr)
        frac_close = np.sum(darr < avg_gap / 4) / len(darr)

        ising_results.append({
            'n': n_plaq, 'nz': len(zeros), 'avg_gap': avg_gap,
            'avg_d': avg_d, 'med_d': med_d,
            'ratio': avg_d / avg_gap, 'med_ratio': med_d / avg_gap,
            'frac': frac_close
        })
        print(f"  {n_plaq:4d} {len(zeros):7d} {avg_gap:8.4f} {avg_d:8.4f} "
              f"{avg_d/avg_gap:10.4f} {med_d:8.4f} {med_d/avg_gap:10.4f} "
              f"{frac_close:7.3f}")

    # 1/n scaling check for Ising
    if len(ising_results) >= 3:
        print(f"\n  1/n scaling check (Ising):")
        for r in ising_results:
            Cf = r['avg_gap'] * r['n']
            print(f"    n={r['n']:3d}: ⟨Δy⟩·n = {Cf:.4f}")

    # Fit power law
    if len(ising_results) >= 4:
        ns = np.array([r['n'] for r in ising_results])
        med_ratios = np.array([r['med_ratio'] for r in ising_results])
        log_n = np.log(ns)
        log_r = np.log(med_ratios + 1e-15)
        valid = med_ratios > 1e-10
        if np.sum(valid) >= 3:
            A = np.column_stack([np.ones(np.sum(valid)), log_n[valid]])
            coeffs = np.linalg.lstsq(A, log_r[valid], rcond=None)[0]
            a_fit = np.exp(coeffs[0])
            alpha = coeffs[1]
            print(f"\n  Power-law fit: med_d/⟨Δy⟩ ≈ {a_fit:.4f} · n^{alpha:.4f}")
            if alpha < -0.1:
                print(f"  → ISING: zeros converge to Stokes lines "
                      f"(α = {alpha:.3f})")

    # ======================================================================
    # PART 3: Overlay Data (for killer figure)
    # ======================================================================
    print(f"\n\n  {'='*90}")
    print(f"  PART 3: Overlay Data for Figure (Wilson, SU(4), n=2)")
    print("  " + "-" * 70)

    # Find Fisher zeros
    n_plaq = 2
    Z = np.zeros(len(y_dense), dtype=complex)
    for p in range(n_reps):
        Z += dims[p] * Ap[p] ** n_plaq
    fisher_zeros = []
    for i in range(len(Z) - 1):
        if Z[i].real * Z[i + 1].real < 0:
            frac = abs(Z[i].real) / (abs(Z[i].real) + abs(Z[i + 1].real))
            fisher_zeros.append(y_dense[i] + frac *
                                (y_dense[i + 1] - y_dense[i]))

    print(f"\n  Fisher zeros (Wilson, n=2, κ=1):")
    for i, yz in enumerate(fisher_zeros):
        # Find nearest crossing
        idx = np.argmin(np.abs(all_crossings_y - yz))
        yc = all_crossings_y[idx]
        pair = crossing_pairs[idx]
        d = abs(yz - yc)
        print(f"    zero {i+1:2d}: y = {yz:8.4f}  "
              f"nearest crossing: y = {yc:8.4f} ({pair})  "
              f"d = {d:.4f}")

    # Print crossing locations for overlay
    print(f"\n  |A_p|=|A_q| crossings in y ∈ [0, 25]:")
    sorted_idx = np.argsort(all_crossings_y)
    for i in sorted_idx[:40]:
        print(f"    y = {all_crossings_y[i]:8.4f}  pair = {crossing_pairs[i]}")

    # Ising overlay
    print(f"\n  Ising overlay (β_r=0.5, n=10):")
    Z_ising = ising_Zn(beta_r + 1j * y_ising, 10)
    ising_zeros_10 = []
    for i in range(len(Z_ising) - 1):
        if Z_ising[i].real * Z_ising[i + 1].real < 0:
            frac = (abs(Z_ising[i].real) /
                    (abs(Z_ising[i].real) + abs(Z_ising[i + 1].real)))
            ising_zeros_10.append(y_ising[i] + frac *
                                  (y_ising[i + 1] - y_ising[i]))

    print(f"  Ising n=10: {len(ising_zeros_10)} zeros")
    print(f"  Stokes lines at y = π/4 + kπ/2:")
    for k in range(8):
        ys = pi / 4 + k * pi / 2
        # Nearest zero
        dists = [abs(yz - ys) for yz in ising_zeros_10]
        if dists:
            min_d = min(dists)
            print(f"    y_stokes = {ys:8.4f}  "
                  f"nearest zero dist = {min_d:.6f}")

    # ======================================================================
    # SUMMARY
    # ======================================================================
    print(f"\n\n  {'='*90}")
    print(f"  SUMMARY")
    print(f"  {'='*90}")
    print(f"""
  PART 1 (SU(4) Wilson):
    Does med_d/⟨Δy⟩ → 0 as n → ∞?  [See power-law fit above]

  PART 2 (1D Ising):
    Does the mechanism work for Ising?
    Analytic Stokes lines: y = π/4 + kπ/2 (exact for β_r = any)
    [See distance table above]

  PART 3 (Overlay data):
    [Ready for figure generation]

  [Completed in {time.time()-t0:.1f}s]""")
    print("=" * 90)


if __name__ == '__main__':
    main()
