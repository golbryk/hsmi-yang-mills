"""
Asymptotic Validation: Zeros → Stokes Lines for Large n
=========================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: Paper Psi — numerical support for Proposition 1

Tests:
  (1) As n grows, do zeros get closer to |A_p|=|A_q| crossings?
  (2) Does the 2-term approximation predict zero locations?
  (3) Does the distance scale as 1/n?
"""

import numpy as np
from math import comb, pi
import time
import sys


# ---------------------------------------------------------------------------
# SU(N) infrastructure (reused from level_crossing_statistics.py)
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


def main():
    t0 = time.time()

    print()
    print("=" * 90)
    print("  Asymptotic Validation: Zeros Concentrate on Stokes Lines")
    print("=" * 90)

    N = 4
    n_quad = 40
    n_reps = 14
    kappa = 1.0

    print(f"  Building SU({N}) Weyl grid...", end=" ", flush=True)
    z, theta_all, measure = build_weyl_grid(N, n_quad)
    Phi1 = np.sum(np.cos(theta_all), axis=1)
    Phi2 = np.sum(np.cos(2 * theta_all), axis=1)
    hp_list = [h_p_vec(p, z) for p in range(n_reps)]
    dims = [dim_rep(p, N) for p in range(n_reps)]
    print("Done.")

    actions = {
        'Wilson': Phi1,
        'Symanzik': (5.0 / 3.0) * Phi1 + (-1.0 / 12.0) * Phi2,
    }

    y_dense = np.linspace(0.01, 25.0, 5000)

    # ======================================================================
    # TEST 1: Distance to Stokes line vs n
    # ======================================================================
    print(f"\n  TEST 1: Distance to Stokes Lines vs n")
    print("  " + "-" * 70)
    print(f"  Does distance decrease with n?")

    n_values = [2, 3, 4, 5, 6, 8]

    for act_name, Phi_action in actions.items():
        print(f"\n  {act_name}:")

        # Precompute A_p along y
        exp_kPhi = np.exp(kappa * Phi_action)
        Ap = np.zeros((n_reps, len(y_dense)), dtype=complex)
        for iy, y in enumerate(y_dense):
            weighted = exp_kPhi * np.exp(1j * y * Phi_action) * measure
            for p in range(n_reps):
                Ap[p, iy] = np.sum(hp_list[p] * weighted)

        abs_Ap = np.abs(Ap)

        # Find all |A_p|=|A_q| crossings (p,q < 8)
        all_crossings_y = []
        for p in range(min(8, n_reps)):
            for q in range(p + 1, min(8, n_reps)):
                diff = abs_Ap[p] - abs_Ap[q]
                for i in range(len(diff) - 1):
                    if diff[i] * diff[i + 1] < 0:
                        frac = abs(diff[i]) / (abs(diff[i]) + abs(diff[i+1]))
                        yc = y_dense[i] + frac * (y_dense[i+1] - y_dense[i])
                        all_crossings_y.append(yc)
        all_crossings_y = np.sort(all_crossings_y)

        print(f"    |A_p|=|A_q| crossings found: {len(all_crossings_y)}")
        print(f"\n    {'n':>4} {'#zeros':>7} {'⟨Δy⟩':>8} {'⟨d_cross⟩':>10} "
              f"{'⟨d⟩/⟨Δy⟩':>10} {'med_d/⟨Δy⟩':>11} {'frac<Δy/4':>10}")

        results = []
        for n_plaq in n_values:
            # Compute Z_n
            Z = np.zeros(len(y_dense), dtype=complex)
            for p in range(n_reps):
                Z += dims[p] * Ap[p] ** n_plaq

            # Find zeros
            zeros = []
            for i in range(len(Z) - 1):
                if Z[i].real * Z[i + 1].real < 0:
                    frac = abs(Z[i].real) / (abs(Z[i].real) +
                                              abs(Z[i + 1].real))
                    zeros.append(y_dense[i] + frac *
                                 (y_dense[i + 1] - y_dense[i]))

            if len(zeros) < 2:
                print(f"    {n_plaq:4d} {len(zeros):7d}      —        —"
                      f"          —           —          —")
                continue

            gaps = np.diff(zeros)
            avg_gap = np.mean(gaps)

            # Distance to nearest crossing
            distances = []
            for yz in zeros:
                if len(all_crossings_y) > 0:
                    d = np.min(np.abs(all_crossings_y - yz))
                    distances.append(d)

            if distances:
                darr = np.array(distances)
                avg_d = np.mean(darr)
                med_d = np.median(darr)
                frac_close = np.sum(darr < avg_gap / 4) / len(darr)
                results.append((n_plaq, len(zeros), avg_gap, avg_d,
                                avg_d / avg_gap, med_d / avg_gap,
                                frac_close))
                print(f"    {n_plaq:4d} {len(zeros):7d} {avg_gap:8.4f} "
                      f"{avg_d:10.4f} {avg_d/avg_gap:10.4f} "
                      f"{med_d/avg_gap:11.4f} {frac_close:10.3f}")

        # Check if distance/spacing decreases with n
        if len(results) >= 3:
            ns = np.array([r[0] for r in results])
            ratios = np.array([r[4] for r in results])
            print(f"\n    Trend: d/⟨Δy⟩ vs n:")
            for n_val, ratio in zip(ns, ratios):
                bar = "#" * int(ratio * 50)
                print(f"      n={n_val}: {ratio:.4f} {bar}")

    # ======================================================================
    # TEST 2: Two-term approximation accuracy
    # ======================================================================
    print(f"\n\n  {'='*90}")
    print(f"  TEST 2: Two-Term Approximation")
    print("  " + "-" * 70)
    print(f"  At each zero: |Z_exact - Z_{{2-term}}| / |Z_{{max-term}}|")

    for act_name in ['Wilson', 'Symanzik']:
        Phi_action = actions[act_name]

        exp_kPhi = np.exp(kappa * Phi_action)
        Ap = np.zeros((n_reps, len(y_dense)), dtype=complex)
        for iy, y in enumerate(y_dense):
            weighted = exp_kPhi * np.exp(1j * y * Phi_action) * measure
            for p in range(n_reps):
                Ap[p, iy] = np.sum(hp_list[p] * weighted)

        for n_plaq in [2, 4, 6, 8]:
            print(f"\n  {act_name}, n={n_plaq}:")

            Z_exact = np.zeros(len(y_dense), dtype=complex)
            terms = np.zeros((n_reps, len(y_dense)), dtype=complex)
            for p in range(n_reps):
                terms[p] = dims[p] * Ap[p] ** n_plaq
                Z_exact += terms[p]

            # Find zeros
            zeros_idx = []
            for i in range(len(Z_exact) - 1):
                if Z_exact[i].real * Z_exact[i + 1].real < 0:
                    zeros_idx.append(i)

            if len(zeros_idx) < 2:
                print(f"    Only {len(zeros_idx)} zeros.")
                continue

            # At each zero: compute 2-term and 3-term approximations
            err_2term = []
            err_3term = []
            for iz in zeros_idx:
                T_mag = np.abs(terms[:, iz])
                sorted_p = np.argsort(-T_mag)
                p1, p2, p3 = sorted_p[0], sorted_p[1], sorted_p[2]

                Z_2 = terms[p1, iz] + terms[p2, iz]
                Z_3 = Z_2 + terms[p3, iz]
                Z_ex = Z_exact[iz]

                scale = T_mag[p1]
                if scale > 0:
                    err_2term.append(abs(Z_ex - Z_2) / scale)
                    err_3term.append(abs(Z_ex - Z_3) / scale)

            if err_2term:
                e2 = np.array(err_2term)
                e3 = np.array(err_3term)
                print(f"    {len(zeros_idx)} zeros")
                print(f"    2-term error: mean={np.mean(e2):.4f}, "
                      f"median={np.median(e2):.4f}")
                print(f"    3-term error: mean={np.mean(e3):.4f}, "
                      f"median={np.median(e3):.4f}")
                print(f"    Improvement 3/2: {np.mean(e2)/np.mean(e3):.2f}×")

    # ======================================================================
    # TEST 3: Predicted vs actual zero locations (2-term formula)
    # ======================================================================
    print(f"\n\n  {'='*90}")
    print(f"  TEST 3: Predicted vs Actual Zero Locations")
    print("  " + "-" * 70)
    print(f"  Using nΔθ = (2k+1)π with local dominant pair")

    Phi_W = actions['Wilson']
    exp_kPhi = np.exp(kappa * Phi_W)
    Ap_W = np.zeros((n_reps, len(y_dense)), dtype=complex)
    for iy, y in enumerate(y_dense):
        weighted = exp_kPhi * np.exp(1j * y * Phi_W) * measure
        for p in range(n_reps):
            Ap_W[p, iy] = np.sum(hp_list[p] * weighted)

    theta_W = np.zeros((n_reps, len(y_dense)))
    for p in range(n_reps):
        theta_W[p] = np.unwrap(np.angle(Ap_W[p]))

    for n_plaq in [2, 3]:
        print(f"\n  Wilson, n={n_plaq}:")

        Z = np.zeros(len(y_dense), dtype=complex)
        T_all = np.zeros((n_reps, len(y_dense)), dtype=complex)
        for p in range(n_reps):
            T_all[p] = dims[p] * Ap_W[p] ** n_plaq
            Z += T_all[p]

        # Find actual zeros
        actual_zeros = []
        for i in range(len(Z) - 1):
            if Z[i].real * Z[i + 1].real < 0:
                frac = abs(Z[i].real) / (abs(Z[i].real) + abs(Z[i+1].real))
                actual_zeros.append(y_dense[i] + frac *
                                    (y_dense[i+1] - y_dense[i]))

        # At each y, find dominant pair and predict zeros from
        # n·(θ_p - θ_q) = (2k+1)π
        # Look for where n·Δθ crosses odd multiples of π
        abs_T = np.abs(T_all)
        for iy in range(len(y_dense)):
            sorted_p = np.argsort(-abs_T[:, iy])
            # We want the dominant pair for each y

        # Simpler approach: for each consecutive pair (p, p+1),
        # find where n·(θ_p - θ_{p+1}) = (2k+1)π
        predicted = []
        for p in range(min(8, n_reps - 1)):
            phase_diff = n_plaq * (theta_W[p] - theta_W[p + 1])
            # Find where phase_diff crosses odd multiples of π
            for i in range(len(phase_diff) - 1):
                # Nearest odd multiple of π
                k_before = round((phase_diff[i] - pi) / (2 * pi))
                target = (2 * k_before + 1) * pi
                if ((phase_diff[i] - target) * (phase_diff[i+1] - target)
                        < 0):
                    frac = (abs(phase_diff[i] - target) /
                            (abs(phase_diff[i] - target) +
                             abs(phase_diff[i+1] - target)))
                    yp = y_dense[i] + frac * (y_dense[i+1] - y_dense[i])
                    # Check that these two terms are among top 3 at this y
                    iy_mid = (i + i + 1) // 2
                    mag_p = abs_T[p, iy_mid]
                    mag_q = abs_T[p + 1, iy_mid]
                    total = np.sum(abs_T[:, iy_mid])
                    if (mag_p + mag_q) / total > 0.3:
                        predicted.append(yp)

        predicted = np.sort(predicted)
        print(f"    Actual zeros: {len(actual_zeros)}")
        print(f"    Predicted zeros (from dominant pairs): {len(predicted)}")

        # Match predicted to actual
        if actual_zeros and len(predicted) > 0:
            matched = 0
            avg_gap = (np.mean(np.diff(actual_zeros))
                       if len(actual_zeros) >= 2 else 1.0)
            for ya in actual_zeros:
                best_d = np.min(np.abs(predicted - ya))
                if best_d < avg_gap / 3:
                    matched += 1
            print(f"    Matched (within ⟨Δy⟩/3): {matched}/{len(actual_zeros)}"
                  f" = {matched/len(actual_zeros):.1%}")

    # ======================================================================
    # SUMMARY
    # ======================================================================
    print(f"\n\n  {'='*90}")
    print(f"  SUMMARY")
    print(f"  {'='*90}")
    print(f"""
  TEST 1: Zeros concentrate on Stokes lines
    [See tables above — d/⟨Δy⟩ should decrease or stay small for all n]

  TEST 2: Two-term approximation
    [Error at zeros — how well does Z ≈ T_p + T_q work?]

  TEST 3: Predicted zeros from phase condition
    [Match rate between nΔθ = (2k+1)π predictions and actual zeros]

  [Completed in {time.time()-t0:.1f}s]""")
    print("=" * 90)


if __name__ == '__main__':
    main()
