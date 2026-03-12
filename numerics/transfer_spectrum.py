"""
Transfer Operator Eigenvalue Spectrum and Fisher Zeros
=======================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 52

Key insight: Z_n(s) = Σ_p d_p A_p(s)^n = Tr(T^n)
where T has eigenvalues A_p(s) with degeneracy d_p.

For s = κ+iy: A_p = |A_p| exp(iθ_p), so Z_n = Σ d_p |A_p|^n exp(inθ_p).
Fisher zeros occur where this sum vanishes.

Tests:
  (1) Eigenphase spectrum θ_p(y) — map the full structure
  (2) At each Fisher zero: does nΔθ_{pq} = π for the dominant pair?
  (3) r-statistic vs effective mode count N_eff
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


def casimir_suN(p, N):
    return p * (p + N) / float(N)


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
# Compute A_p(κ+iy) along dense y scan
# ---------------------------------------------------------------------------

def compute_Ap_scan(Phi_action, measure, hp_list, n_reps, kappa, y_values):
    """Compute A_p(κ+iy) for all p and all y."""
    n_y = len(y_values)
    exp_kPhi = np.exp(kappa * Phi_action)
    Ap = np.zeros((n_reps, n_y), dtype=complex)
    for iy, y in enumerate(y_values):
        weighted = exp_kPhi * np.exp(1j * y * Phi_action) * measure
        for p in range(n_reps):
            Ap[p, iy] = np.sum(hp_list[p] * weighted)
    return Ap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print()
    print("=" * 90)
    print("  Transfer Operator Eigenvalue Spectrum and Fisher Zeros")
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
        'Iwasaki': 3.648 * Phi1 + (-0.331) * Phi2,
        'DBW2': 12.2688 * Phi1 + (-1.4086) * Phi2,
    }

    # ======================================================================
    # PART 1: Eigenphase spectrum θ_p(y)
    # ======================================================================
    print(f"\n  PART 1: Eigenphase Spectrum θ_p(y) = arg(A_p(κ+iy))")
    print("  " + "-" * 70)

    y_dense = np.linspace(0.01, 25.0, 2500)
    n_plaq = 2

    for act_name in ['Wilson', 'Symanzik', 'Iwasaki']:
        Phi_action = actions[act_name]
        print(f"\n  === {act_name} (SU({N}), κ={kappa}) ===")
        sys.stdout.flush()

        Ap = compute_Ap_scan(Phi_action, measure, hp_list, n_reps, kappa,
                             y_dense)

        # Eigenphases (unwrapped)
        theta = np.zeros((n_reps, len(y_dense)))
        for p in range(n_reps):
            theta[p] = np.unwrap(np.angle(Ap[p]))

        # Eigenphase velocities (average over y range)
        dtheta_dy = np.zeros(n_reps)
        for p in range(n_reps):
            # Linear fit θ_p(y) ≈ θ_p(0) + (dθ/dy)·y
            coeffs = np.polyfit(y_dense, theta[p], 1)
            dtheta_dy[p] = coeffs[0]

        print(f"\n    Eigenphase velocities (linear fit dθ_p/dy):")
        print(f"      {'p':>3} {'d_p':>6} {'dθ/dy':>10} {'C_2(p)':>8} "
              f"{'|A_p(κ)|':>12} {'log|A_p|':>10}")
        for p in range(min(10, n_reps)):
            Ap_real = abs(Ap[p, 0])
            c2 = casimir_suN(p, N)
            logAp = np.log(Ap_real) if Ap_real > 1e-30 else -99
            print(f"      {p:3d} {dims[p]:6d} {dtheta_dy[p]:10.4f} "
                  f"{c2:8.4f} {Ap_real:12.6e} {logAp:10.4f}")

        # Phase differences n·(θ_p - θ_q) at Fisher zero locations
        Z = np.zeros(len(y_dense), dtype=complex)
        for p in range(n_reps):
            Z += dims[p] * Ap[p] ** n_plaq
        # Find zeros
        zeros_idx = []
        for i in range(len(Z) - 1):
            if Z[i].real * Z[i + 1].real < 0:
                zeros_idx.append(i)

        print(f"\n    Fisher zeros: {len(zeros_idx)} found")

        if len(zeros_idx) >= 3:
            print(f"\n    At each zero: n·Δθ_{'{p,q}'} mod 2π for dominant pair")
            print(f"      {'y':>8} {'dom_p':>5} {'dom_q':>5} {'|T_p|':>10} "
                  f"{'|T_q|':>10} {'nΔθ mod 2π':>12} {'|nΔθ-π|':>10}")

            ndtheta_at_zeros = []
            for count, iz in enumerate(zeros_idx):
                y0 = y_dense[iz]
                # Term magnitudes and phases
                T_mag = np.array([dims[p] * abs(Ap[p, iz]) ** n_plaq
                                  for p in range(n_reps)])
                T_phase = np.array([n_plaq * theta[p, iz]
                                    for p in range(n_reps)])

                # Find two largest terms
                sorted_p = np.argsort(-T_mag)
                p1, p2 = sorted_p[0], sorted_p[1]

                # Phase difference
                delta_phase = (T_phase[p1] - T_phase[p2]) % (2 * pi)
                dist_to_pi = min(abs(delta_phase - pi),
                                 abs(delta_phase - 3 * pi),
                                 abs(delta_phase + pi))
                ndtheta_at_zeros.append(dist_to_pi)

                if count < 25 or count == len(zeros_idx) - 1:
                    print(f"      {y0:8.3f} {p1:5d} {p2:5d} "
                          f"{T_mag[p1]:10.4e} {T_mag[p2]:10.4e} "
                          f"{delta_phase:12.4f} {dist_to_pi:10.4f}")

            arr = np.array(ndtheta_at_zeros)
            print(f"\n    Summary: |nΔθ - π| at zeros")
            print(f"      Mean = {np.mean(arr):.4f}")
            print(f"      Median = {np.median(arr):.4f}")
            print(f"      Max = {np.max(arr):.4f}")
            print(f"      Fraction with |nΔθ - π| < 0.5: "
                  f"{np.sum(arr < 0.5)/len(arr):.3f}")
            print(f"      Fraction with |nΔθ - π| < 1.0: "
                  f"{np.sum(arr < 1.0)/len(arr):.3f}")

        # Eigenphase spacing at fixed y
        print(f"\n    Eigenphase distribution at y=5:")
        iy5 = np.argmin(np.abs(y_dense - 5.0))
        phases_y5 = np.array([theta[p, iy5] % (2 * pi) for p in range(n_reps)])
        sorted_phases = np.sort(phases_y5)
        phase_gaps = np.diff(sorted_phases)
        if len(phase_gaps) > 0:
            print(f"      Sorted phases mod 2π: "
                  f"{', '.join(f'{p:.3f}' for p in sorted_phases[:10])}")
            print(f"      Phase gaps: "
                  f"{', '.join(f'{g:.3f}' for g in phase_gaps[:10])}")
            print(f"      ⟨Δθ⟩ = {np.mean(phase_gaps):.4f}, "
                  f"σ = {np.std(phase_gaps):.4f}, "
                  f"CV = {np.std(phase_gaps)/np.mean(phase_gaps):.3f}")

    # ======================================================================
    # PART 2: n·Δθ = π test for multiple n values
    # ======================================================================
    print(f"\n\n  {'='*90}")
    print(f"  PART 2: n·Δθ = π Test at Different n")
    print("  " + "-" * 70)
    print(f"  If zeros arise from 2-mode cancellation: nΔθ_pq = π at zero")

    Phi_W = actions['Wilson']
    Ap_W = compute_Ap_scan(Phi_W, measure, hp_list, n_reps, kappa, y_dense)
    theta_W = np.zeros((n_reps, len(y_dense)))
    for p in range(n_reps):
        theta_W[p] = np.unwrap(np.angle(Ap_W[p]))

    for n_plaq in [2, 3, 4]:
        print(f"\n  Wilson, n={n_plaq}:")
        Z = np.zeros(len(y_dense), dtype=complex)
        for p in range(n_reps):
            Z += dims[p] * Ap_W[p] ** n_plaq

        zeros_idx = []
        for i in range(len(Z) - 1):
            if Z[i].real * Z[i + 1].real < 0:
                zeros_idx.append(i)

        if len(zeros_idx) < 2:
            print(f"    Only {len(zeros_idx)} zeros found.")
            continue

        # For each zero, check if dominant pair has nΔθ ≈ π
        deviations = []
        pair_deviations = []  # including 3rd term
        for iz in zeros_idx:
            T_mag = np.array([dims[p] * abs(Ap_W[p, iz]) ** n_plaq
                              for p in range(n_reps)])
            T_phase = np.array([n_plaq * theta_W[p, iz]
                                for p in range(n_reps)])
            sorted_p = np.argsort(-T_mag)
            p1, p2, p3 = sorted_p[0], sorted_p[1], sorted_p[2]

            delta = (T_phase[p1] - T_phase[p2]) % (2 * pi)
            dist = min(abs(delta - pi), abs(delta - 3*pi), abs(delta + pi))
            deviations.append(dist)

            # Check all pairs among top 3
            best_dist = dist
            for pa, pb in [(p1, p2), (p1, p3), (p2, p3)]:
                d = (T_phase[pa] - T_phase[pb]) % (2 * pi)
                dd = min(abs(d - pi), abs(d - 3*pi), abs(d + pi))
                best_dist = min(best_dist, dd)
            pair_deviations.append(best_dist)

        dev = np.array(deviations)
        pdev = np.array(pair_deviations)
        print(f"    {len(zeros_idx)} zeros")
        print(f"    Dominant pair |nΔθ - π|:")
        print(f"      Mean = {np.mean(dev):.4f}, "
              f"Median = {np.median(dev):.4f}")
        print(f"      Frac < 0.3: {np.sum(dev < 0.3)/len(dev):.3f}, "
              f"Frac < 0.5: {np.sum(dev < 0.5)/len(dev):.3f}")
        print(f"    Best among top-3 pairs |nΔθ - π|:")
        print(f"      Mean = {np.mean(pdev):.4f}, "
              f"Median = {np.median(pdev):.4f}")
        print(f"      Frac < 0.3: {np.sum(pdev < 0.3)/len(pdev):.3f}, "
              f"Frac < 0.5: {np.sum(pdev < 0.5)/len(pdev):.3f}")

    # ======================================================================
    # PART 3: r-statistic vs N_eff (effective mode count)
    # ======================================================================
    print(f"\n\n  {'='*90}")
    print(f"  PART 3: r-Statistic vs Effective Mode Count")
    print("  " + "-" * 70)

    n_plaq = 2
    y_scan = np.linspace(0.1, 25.0, 2000)

    # For each action: compute N_eff and r
    all_actions = {
        'Wilson': Phi1,
        'Symanzik': (5.0/3.0) * Phi1 + (-1.0/12.0) * Phi2,
        'Iwasaki': 3.648 * Phi1 + (-0.331) * Phi2,
        'DBW2': 12.2688 * Phi1 + (-1.4086) * Phi2,
    }

    print(f"\n  {'Action':<12} {'N_eff_1':>7} {'N_eff_2':>7} "
          f"{'#zeros':>7} {'⟨Δy⟩':>8} {'r':>8}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")

    neff_r_data = []

    for act_name, Phi_action in all_actions.items():
        sys.stdout.flush()

        # Compute A_p along scan
        Ap = compute_Ap_scan(Phi_action, measure, hp_list, n_reps, kappa,
                             y_scan)

        # N_eff method 1: count terms with |T_p(y=0)|/|T_max| > 0.01
        T0_mag = np.array([dims[p] * abs(Ap[p, 0]) ** n_plaq
                           for p in range(n_reps)])
        T0_max = np.max(T0_mag)
        neff1 = np.sum(T0_mag / T0_max > 0.01)

        # N_eff method 2: participation ratio PR = (Σ|T_p|)² / (Σ|T_p|²)
        # averaged over y
        pr_vals = []
        for iy in range(0, len(y_scan), 10):
            Tmag = np.array([dims[p] * abs(Ap[p, iy]) ** n_plaq
                             for p in range(n_reps)])
            if np.sum(Tmag ** 2) > 0:
                pr = np.sum(Tmag) ** 2 / np.sum(Tmag ** 2)
                pr_vals.append(pr)
        neff2 = np.mean(pr_vals) if pr_vals else 0

        # Find zeros and compute r
        Z = np.zeros(len(y_scan), dtype=complex)
        for p in range(n_reps):
            Z += dims[p] * Ap[p] ** n_plaq

        zeros = []
        for i in range(len(Z) - 1):
            if Z[i].real * Z[i + 1].real < 0:
                frac = abs(Z[i].real) / (abs(Z[i].real) + abs(Z[i + 1].real))
                zeros.append(y_scan[i] + frac * (y_scan[i + 1] - y_scan[i]))

        if len(zeros) >= 3:
            gaps = np.diff(zeros)
            avg_gap = np.mean(gaps)
            # r-statistic
            ratios = []
            for i in range(len(gaps) - 1):
                mn = min(gaps[i], gaps[i + 1])
                mx = max(gaps[i], gaps[i + 1])
                if mx > 1e-10:
                    ratios.append(mn / mx)
            r = np.mean(ratios) if ratios else float('nan')

            neff_r_data.append((act_name, neff1, neff2, len(zeros),
                                avg_gap, r))
            print(f"  {act_name:<12} {neff1:7d} {neff2:7.2f} "
                  f"{len(zeros):7d} {avg_gap:8.4f} {r:8.4f}")
        else:
            print(f"  {act_name:<12} {neff1:7d} {neff2:7.2f} "
                  f"{len(zeros):7d}      N/A      N/A")

    # Add HK data
    print(f"\n  Heat-Kernel (computed separately):")
    for N_hk in [3, 4]:
        p_max = 30
        y_hk = np.linspace(0.01, 40.0, 4000)

        # Compute Z_n^HK
        n_plaq_hk = 2
        Z_hk = np.zeros(len(y_hk), dtype=complex)
        T_hk_mag_0 = []
        for p in range(p_max + 1):
            dp = dim_rep(p, N_hk)
            c2 = casimir_suN(p, N_hk)
            s = kappa + 1j * y_hk
            term = dp ** (n_plaq_hk + 1) * np.exp(-n_plaq_hk * c2 * s)
            Z_hk += term
            T_hk_mag_0.append(abs(dp ** (n_plaq_hk + 1) *
                                  np.exp(-n_plaq_hk * c2 * kappa)))

        T_arr = np.array(T_hk_mag_0)
        T_max = np.max(T_arr)
        neff1_hk = np.sum(T_arr / T_max > 0.01)
        neff2_hk = np.sum(T_arr) ** 2 / np.sum(T_arr ** 2)

        zeros_hk = []
        for i in range(len(Z_hk) - 1):
            if Z_hk[i].real * Z_hk[i + 1].real < 0:
                frac = (abs(Z_hk[i].real) /
                        (abs(Z_hk[i].real) + abs(Z_hk[i + 1].real)))
                zeros_hk.append(y_hk[i] + frac *
                                (y_hk[i + 1] - y_hk[i]))

        if len(zeros_hk) >= 3:
            gaps_hk = np.diff(zeros_hk)
            ratios_hk = []
            for i in range(len(gaps_hk) - 1):
                mn = min(gaps_hk[i], gaps_hk[i + 1])
                mx = max(gaps_hk[i], gaps_hk[i + 1])
                if mx > 1e-10:
                    ratios_hk.append(mn / mx)
            r_hk = np.mean(ratios_hk)
            neff_r_data.append((f'HK SU({N_hk})', neff1_hk, neff2_hk,
                                len(zeros_hk), np.mean(gaps_hk), r_hk))
            print(f"  {'HK SU('+str(N_hk)+')':<12} {neff1_hk:7d} "
                  f"{neff2_hk:7.2f} {len(zeros_hk):7d} "
                  f"{np.mean(gaps_hk):8.4f} {r_hk:8.4f}")

    # Correlation analysis
    print(f"\n  Correlation: r vs N_eff")
    print(f"  {'System':<12} {'N_eff(PR)':>9} {'r':>8}")
    print(f"  {'-'*12} {'-'*9} {'-'*8}")
    for name, n1, n2, nz, avg, r in sorted(neff_r_data, key=lambda x: x[2]):
        print(f"  {name:<12} {n2:9.2f} {r:8.4f}")

    # Fit r = f(N_eff)
    neffs = np.array([x[2] for x in neff_r_data])
    rs = np.array([x[5] for x in neff_r_data])
    valid = ~np.isnan(rs)
    if np.sum(valid) >= 3:
        neffs_v = neffs[valid]
        rs_v = rs[valid]
        # Try r = a + b·log(N_eff)
        log_neff = np.log(neffs_v)
        A = np.column_stack([np.ones(len(log_neff)), log_neff])
        result = np.linalg.lstsq(A, rs_v, rcond=None)
        a_fit, b_fit = result[0]
        residuals = rs_v - (a_fit + b_fit * log_neff)
        rmse = np.sqrt(np.mean(residuals ** 2))

        print(f"\n  Fit: r ≈ {a_fit:.4f} + {b_fit:.4f}·ln(N_eff)")
        print(f"  RMSE = {rmse:.4f}")
        print(f"\n  Predictions:")
        for name, n1, n2, nz, avg, r in neff_r_data:
            r_pred = a_fit + b_fit * np.log(n2)
            print(f"    {name:<12}: N_eff={n2:.2f}, "
                  f"r_actual={r:.4f}, r_pred={r_pred:.4f}, "
                  f"Δ={abs(r-r_pred):.4f}")

    # ======================================================================
    # PART 4: Eigenphase level crossings and Fisher zeros
    # ======================================================================
    print(f"\n\n  {'='*90}")
    print(f"  PART 4: Eigenphase Level Crossings → Fisher Zeros")
    print("  " + "-" * 70)
    print(f"  Do |A_p| = |A_q| crossings coincide with Fisher zeros?")

    for act_name in ['Wilson', 'Symanzik']:
        Phi_action = all_actions[act_name]
        n_plaq = 2
        print(f"\n  {act_name} (n={n_plaq}):")

        Ap = compute_Ap_scan(Phi_action, measure, hp_list, n_reps, kappa,
                             y_dense)

        # Find Fisher zeros
        Z = np.zeros(len(y_dense), dtype=complex)
        for p in range(n_reps):
            Z += dims[p] * Ap[p] ** n_plaq

        f_zeros = []
        for i in range(len(Z) - 1):
            if Z[i].real * Z[i + 1].real < 0:
                frac = abs(Z[i].real) / (abs(Z[i].real) + abs(Z[i + 1].real))
                f_zeros.append(y_dense[i] + frac *
                               (y_dense[i + 1] - y_dense[i]))

        # Find |A_p| = |A_q| crossings for all pairs
        abs_Ap = np.abs(Ap)
        crossings = {}
        for p in range(min(8, n_reps)):
            for q in range(p + 1, min(8, n_reps)):
                diff = abs_Ap[p] - abs_Ap[q]
                for i in range(len(diff) - 1):
                    if diff[i] * diff[i + 1] < 0:
                        frac = abs(diff[i]) / (abs(diff[i]) + abs(diff[i+1]))
                        yc = y_dense[i] + frac * (y_dense[i+1] - y_dense[i])
                        if (p, q) not in crossings:
                            crossings[(p, q)] = []
                        crossings[(p, q)].append(yc)

        # For each Fisher zero, find nearest |A_p|=|A_q| crossing
        all_crossings = []
        for pq, ycs in crossings.items():
            for yc in ycs:
                all_crossings.append((yc, pq))
        all_crossings.sort()

        if all_crossings and f_zeros:
            print(f"    Fisher zeros: {len(f_zeros)}")
            print(f"    |A_p|=|A_q| crossings (p,q < 8): {len(all_crossings)}")
            print(f"\n    Nearest crossing to each Fisher zero:")
            print(f"      {'y_zero':>8} {'y_cross':>8} {'|Δy|':>8} "
                  f"{'pair':>8}")

            distances = []
            for yz in f_zeros:
                best_dist = float('inf')
                best_cross = None
                best_pair = None
                for yc, pq in all_crossings:
                    d = abs(yz - yc)
                    if d < best_dist:
                        best_dist = d
                        best_cross = yc
                        best_pair = pq
                distances.append(best_dist)
                if len(distances) <= 20:
                    print(f"      {yz:8.3f} {best_cross:8.3f} "
                          f"{best_dist:8.4f} {str(best_pair):>8}")

            darr = np.array(distances)
            avg_gap = np.mean(np.diff(f_zeros)) if len(f_zeros) >= 2 else 1.0
            print(f"\n    Distance statistics (⟨Δy⟩ = {avg_gap:.4f}):")
            print(f"      ⟨distance⟩ = {np.mean(darr):.4f} "
                  f"({np.mean(darr)/avg_gap:.2f}× ⟨Δy⟩)")
            print(f"      Median = {np.median(darr):.4f} "
                  f"({np.median(darr)/avg_gap:.2f}× ⟨Δy⟩)")
            print(f"      Frac < ⟨Δy⟩/4: "
                  f"{np.sum(darr < avg_gap/4)/len(darr):.3f}")
            print(f"      Frac < ⟨Δy⟩/2: "
                  f"{np.sum(darr < avg_gap/2)/len(darr):.3f}")

    # ======================================================================
    # SUMMARY
    # ======================================================================
    print(f"\n\n  {'='*90}")
    print(f"  SUMMARY")
    print(f"  {'='*90}")
    print(f"""
  PART 1: Eigenphase spectrum θ_p(y)
    Eigenvalues A_p(κ+iy) are the transfer operator eigenvalues.
    Phase velocities dθ_p/dy are NOT Casimir values (for Wilson).
    The eigenphase structure is action-dependent.

  PART 2: nΔθ = π test
    Fraction of zeros where dominant pair has |nΔθ - π| < 0.5?
    [See Part 2 results above]

  PART 3: r vs N_eff
    [See fit above]

  PART 4: Eigenphase crossings ↔ Fisher zeros
    [See Part 4 results above]

  [Completed in {time.time()-t0:.1f}s]""")
    print("=" * 90)


if __name__ == '__main__':
    main()
