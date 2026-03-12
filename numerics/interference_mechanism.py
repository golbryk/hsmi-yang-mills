"""
Interference Mechanism of Fisher Zeros
=======================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 51 — Three critical tests

Three tests:
  (1) Fourier spectrum |Ẑ(ω)| — how many modes dominate?
  (2) Spacing vs representation gap: Δy ≈ 2π/ω_{pq}?
  (3) Large N (N=24,32): C_f → π/2?

Key idea (Grzegorz): zeros arise from CHARACTER INTERFERENCE
  Z_n = Σ_p d_p A_p(κ+iy)^n
  phase θ_p(y) = n·arg(A_p(κ+iy))
  zero when n(θ_p - θ_q) = π → Δy ~ 1/n
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
# Compute A_p(κ+iy) along y scan
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


def compute_Z_from_Ap(Ap, dims, n_plaq):
    """Z_n = Σ d_p A_p^n."""
    n_reps, n_y = Ap.shape
    Z = np.zeros(n_y, dtype=complex)
    for p in range(n_reps):
        Z += dims[p] * Ap[p] ** n_plaq
    return Z


def find_zeros_from_Z(Z_vals, y_values):
    """Find approximate zero locations from Re Z sign changes."""
    zeros = []
    for i in range(len(Z_vals) - 1):
        if Z_vals[i].real * Z_vals[i + 1].real < 0:
            frac = abs(Z_vals[i].real) / (abs(Z_vals[i].real) +
                                           abs(Z_vals[i + 1].real))
            zeros.append(y_values[i] + frac * (y_values[i + 1] - y_values[i]))
    return np.array(zeros)


# ---------------------------------------------------------------------------
# Haar MC for large N
# ---------------------------------------------------------------------------

def h_p_vec_flat(p, z):
    n_pts = z.shape[0]
    if p == 0:
        return np.ones(n_pts, dtype=complex)
    psums = np.array([np.sum(z ** k, axis=1) for k in range(1, p + 1)])
    h = np.zeros((p + 1, n_pts), dtype=complex)
    h[0] = 1.0
    for k in range(1, p + 1):
        h[k] = sum(psums[j] * h[k - 1 - j] for j in range(k)) / k
    return h[p]


def haar_eigenvalues(N, n_samples):
    thetas = np.zeros((n_samples, N))
    for i in range(n_samples):
        G = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2)
        Q, R = np.linalg.qr(G)
        d = np.diagonal(R)
        ph = d / np.abs(d)
        Q = Q * ph[np.newaxis, :]
        eigs = np.linalg.eigvals(Q)
        angles = np.angle(eigs)
        angles = angles - np.mean(angles)
        thetas[i] = angles
    return thetas


def mc_Ap_scan(N, kappa, y_values, n_samples, n_reps):
    """MC computation of A_p(κ+iy) for Wilson action."""
    thetas = haar_eigenvalues(N, n_samples)
    z = np.exp(1j * thetas)
    Phi = np.sum(np.cos(thetas), axis=1)  # Re Tr U (Wilson)

    hp = np.zeros((n_reps, n_samples), dtype=complex)
    for p in range(n_reps):
        hp[p] = h_p_vec_flat(p, z)

    exp_kPhi = np.exp(kappa * Phi)
    n_y = len(y_values)
    Ap = np.zeros((n_reps, n_y), dtype=complex)

    for iy, y in enumerate(y_values):
        w = exp_kPhi * np.exp(1j * y * Phi) / n_samples
        for p in range(n_reps):
            Ap[p, iy] = np.sum(hp[p] * w)

    return Ap


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    np.random.seed(42)

    print()
    print("=" * 90)
    print("  Interference Mechanism of Fisher Zeros")
    print("  Three Critical Tests")
    print("=" * 90)

    # ======================================================================
    # PART 1: Fourier Spectrum of Z(y)
    # ======================================================================
    print(f"\n  PART 1: Fourier Spectrum |Ẑ(ω)|")
    print("  " + "-" * 70)

    N = 4
    n_quad = 40
    n_reps = 14
    kappa = 1.0
    n_plaq = 2

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
    }

    # Dense y scan for FFT
    n_fft = 4096
    y_max = 50.0
    y_fft = np.linspace(0, y_max, n_fft)
    dy = y_fft[1] - y_fft[0]

    for act_name, Phi_action in actions.items():
        print(f"\n  {act_name}:")
        sys.stdout.flush()

        # Compute Z_n(κ+iy) along scan
        Ap = compute_Ap_scan(Phi_action, measure, hp_list, n_reps, kappa, y_fft)
        Z = compute_Z_from_Ap(Ap, dims, n_plaq)

        # FFT of Z(y) — use full fft since Z is complex, take positive freqs
        Z_centered = Z - np.mean(Z)
        spectrum_full = np.abs(np.fft.fft(Z_centered))
        freqs_full = np.fft.fftfreq(n_fft, d=dy)
        # Keep positive frequencies only
        pos = freqs_full > 0
        spectrum = spectrum_full[pos]
        freqs = freqs_full[pos]
        omega = 2 * pi * freqs

        # Find dominant peaks
        # Normalize
        spectrum_norm = spectrum / np.max(spectrum)
        peaks = []
        for i in range(1, len(spectrum) - 1):
            if (spectrum_norm[i] > spectrum_norm[i - 1] and
                    spectrum_norm[i] > spectrum_norm[i + 1] and
                    spectrum_norm[i] > 0.05):
                peaks.append((omega[i], spectrum_norm[i]))

        peaks.sort(key=lambda x: -x[1])
        n_dom = sum(1 for _, s in peaks if s > 0.1)

        print(f"    Total FFT peaks (>5% of max): {len(peaks)}")
        print(f"    Dominant peaks (>10% of max): {n_dom}")
        print(f"    Top peaks:")
        print(f"      {'ω':>8} {'|Ẑ|/max':>10}  {'2π/ω (=Δy_pred)':>16}")
        for omega_k, amp in peaks[:8]:
            if omega_k > 0.01:
                print(f"      {omega_k:8.3f} {amp:10.4f}  {2*pi/omega_k:16.4f}")

        # Also compute individual term spectra: T_p(y) = d_p A_p(κ+iy)^n
        print(f"\n    Individual term |T_p| at y=0 and phases:")
        print(f"      {'p':>3} {'d_p':>6} {'|A_p|^n':>12} {'|T_p|':>12} "
              f"{'|T_p|/|T_0|':>12}")
        T0_mag = abs(dims[0] * Ap[0, 0] ** n_plaq)
        for p in range(min(8, n_reps)):
            Tp = dims[p] * Ap[p, 0] ** n_plaq
            print(f"      {p:3d} {dims[p]:6d} {abs(Ap[p,0])**n_plaq:12.6e} "
                  f"{abs(Tp):12.6e} {abs(Tp)/T0_mag:12.6e}")

        # Phase velocities dθ_p/dy at y=5
        iy5 = np.argmin(np.abs(y_fft - 5.0))
        iy5p = min(iy5 + 1, len(y_fft) - 1)
        print(f"\n    Phase velocities dθ_p/dy at y=5:")
        print(f"      {'p':>3} {'θ_p(5)':>10} {'dθ_p/dy':>10} "
              f"{'n·dθ_p/dy':>10} {'C_2(p)':>8}")
        for p in range(min(8, n_reps)):
            phase_p = np.angle(Ap[p, iy5] ** n_plaq)
            phase_p1 = np.angle(Ap[p, iy5p] ** n_plaq)
            dphi = (phase_p1 - phase_p) / (y_fft[iy5p] - y_fft[iy5])
            c2 = casimir_suN(p, N)
            theta_raw = np.angle(Ap[p, iy5])
            dtheta = np.angle(Ap[p, iy5p]) - theta_raw
            # unwrap
            dtheta = (dtheta + pi) % (2 * pi) - pi
            dtheta_dy = dtheta / (y_fft[iy5p] - y_fft[iy5])
            print(f"      {p:3d} {theta_raw:10.4f} {dtheta_dy:10.4f} "
                  f"{n_plaq*dtheta_dy:10.4f} {c2:8.4f}")

    # ======================================================================
    # PART 2: Spacing vs Representation Gap
    # ======================================================================
    print(f"\n\n  {'='*90}")
    print(f"  PART 2: Spacing vs Representation Gap")
    print("  " + "-" * 70)
    print(f"  Test: Δy ≈ 2π / ω_{{pq}} where ω_{{pq}} = |dθ_p/dy - dθ_q/dy|")

    y_dense = np.linspace(0.1, 25.0, 2000)
    dy_dense = y_dense[1] - y_dense[0]

    for act_name in ['Wilson', 'Symanzik']:
        Phi_action = actions[act_name]
        print(f"\n  {act_name} (SU({N}), n={n_plaq}, κ={kappa}):")
        sys.stdout.flush()

        Ap = compute_Ap_scan(Phi_action, measure, hp_list, n_reps, kappa,
                             y_dense)
        Z = compute_Z_from_Ap(Ap, dims, n_plaq)
        zeros = find_zeros_from_Z(Z, y_dense)

        if len(zeros) < 3:
            print(f"    Only {len(zeros)} zeros found, skipping.")
            continue

        gaps = np.diff(zeros)
        print(f"    {len(zeros)} zeros, ⟨Δy⟩ = {np.mean(gaps):.4f}")

        # At each zero, find dominant interfering pair
        print(f"\n    Zero-by-zero analysis:")
        print(f"    {'y_zero':>8} {'Δy':>8} {'dom_pair':>10} "
              f"{'ω_pq':>8} {'2π/ω_pq':>8} {'ratio':>8}")

        predicted_gaps = []
        for iz, y0 in enumerate(zeros):
            iy = np.argmin(np.abs(y_dense - y0))

            # Compute |T_p| = d_p |A_p|^n at this zero
            Tp = np.array([dims[p] * abs(Ap[p, iy]) ** n_plaq
                           for p in range(n_reps)])

            # Find two largest |T_p|
            sorted_p = np.argsort(-Tp)
            p1, p2 = sorted_p[0], sorted_p[1]

            # Phase velocity of A_p at this y
            iy_prev = max(0, iy - 1)
            iy_next = min(len(y_dense) - 1, iy + 1)
            delta_y = y_dense[iy_next] - y_dense[iy_prev]
            if delta_y < 1e-10:
                continue

            phases = np.zeros(n_reps)
            for p in range(n_reps):
                ph_next = np.angle(Ap[p, iy_next])
                ph_prev = np.angle(Ap[p, iy_prev])
                dph = (ph_next - ph_prev + pi) % (2 * pi) - pi
                phases[p] = dph / delta_y

            omega_pq = abs(phases[p1] - phases[p2])
            if omega_pq > 0.01:
                predicted = 2 * pi / omega_pq
            else:
                predicted = float('inf')

            actual = gaps[iz] if iz < len(gaps) else float('nan')
            ratio = actual / predicted if (predicted < 100 and
                                           not np.isnan(actual)) else float('nan')
            predicted_gaps.append((actual, predicted, ratio))

            if iz < 20 or iz == len(zeros) - 1:
                pair_str = f"({p1},{p2})"
                print(f"    {y0:8.3f} {actual:8.4f} {pair_str:>10} "
                      f"{omega_pq:8.4f} {predicted:8.4f} {ratio:8.4f}")

        # Summary statistics
        valid = [(a, p, r) for a, p, r in predicted_gaps
                 if not np.isnan(r) and r < 10 and r > 0.1]
        if valid:
            ratios = [r for _, _, r in valid]
            print(f"\n    Summary: {len(valid)}/{len(predicted_gaps)} "
                  f"zeros have valid predictions")
            print(f"    ⟨ratio⟩ = {np.mean(ratios):.4f} ± "
                  f"{np.std(ratios):.4f}")
            print(f"    Median ratio = {np.median(ratios):.4f}")

    # Also test for n=3,4
    for n_test in [3, 4]:
        Phi_action = actions['Wilson']
        print(f"\n  Wilson (SU({N}), n={n_test}, κ={kappa}):")

        Ap = compute_Ap_scan(Phi_action, measure, hp_list, n_reps, kappa,
                             y_dense)
        Z = compute_Z_from_Ap(Ap, dims, n_test)
        zeros = find_zeros_from_Z(Z, y_dense)

        if len(zeros) < 3:
            print(f"    Only {len(zeros)} zeros, skipping.")
            continue

        gaps = np.diff(zeros)
        print(f"    {len(zeros)} zeros, ⟨Δy⟩ = {np.mean(gaps):.4f}, "
              f"C_f = {np.mean(gaps)*n_test:.4f}")

        # Casimir frequency prediction
        c2_fund = casimir_suN(1, N)  # fundamental rep
        c2_0 = casimir_suN(0, N)     # trivial rep
        omega_01 = n_test * (c2_fund - c2_0)
        pred_01 = 2 * pi / omega_01

        c2_2 = casimir_suN(2, N)
        omega_12 = n_test * (c2_2 - c2_fund)
        pred_12 = 2 * pi / omega_12

        omega_02 = n_test * (c2_2 - c2_0)
        pred_02 = 2 * pi / omega_02

        print(f"    Casimir predictions:")
        print(f"      ω_{{0,1}} = n·C_2(1) = {omega_01:.4f}, "
              f"Δy_pred = 2π/ω = {pred_01:.4f}")
        print(f"      ω_{{1,2}} = n·ΔC_2 = {omega_12:.4f}, "
              f"Δy_pred = 2π/ω = {pred_12:.4f}")
        print(f"      ω_{{0,2}} = n·C_2(2) = {omega_02:.4f}, "
              f"Δy_pred = 2π/ω = {pred_02:.4f}")
        print(f"    Actual ⟨Δy⟩ = {np.mean(gaps):.4f}")
        print(f"    Ratios: ⟨Δy⟩/pred_01 = {np.mean(gaps)/pred_01:.4f}, "
              f"⟨Δy⟩/pred_12 = {np.mean(gaps)/pred_12:.4f}")

    # ======================================================================
    # PART 3: Large N — C_f(N) at N=24,32
    # ======================================================================
    print(f"\n\n  {'='*90}")
    print(f"  PART 3: C_f(N) at Large N (N=8,12,16,24,32)")
    print("  " + "-" * 70)

    n_plaq = 2
    kappa = 1.0
    y_scan_mc = np.linspace(0.1, 25.0, 2000)

    large_N_values = [8, 12, 16, 24, 32]
    n_samples_dict = {8: 80000, 12: 80000, 16: 80000, 24: 60000, 32: 40000}
    n_reps_dict = {8: 6, 12: 5, 16: 4, 24: 4, 32: 3}

    print(f"\n  {'N':>4} {'n_samp':>7} {'n_reps':>6} {'#zeros':>7} "
          f"{'⟨Δy⟩':>8} {'C_f':>8}")
    print(f"  {'-'*4} {'-'*7} {'-'*6} {'-'*7} {'-'*8} {'-'*8}")

    Cf_large = {}
    for N_val in large_N_values:
        n_samp = n_samples_dict[N_val]
        n_rep = n_reps_dict[N_val]
        print(f"  {N_val:4d} {n_samp:7d} {n_rep:6d}", end="", flush=True)

        dims_N = [dim_rep(p, N_val) for p in range(n_rep)]
        Ap = mc_Ap_scan(N_val, kappa, y_scan_mc, n_samp, n_rep)
        Z = compute_Z_from_Ap(Ap, dims_N, n_plaq)
        zeros = find_zeros_from_Z(Z, y_scan_mc)

        if len(zeros) >= 2:
            gaps = np.diff(zeros)
            avg_gap = np.mean(gaps)
            Cf = avg_gap * n_plaq
            Cf_large[N_val] = Cf
            print(f" {len(zeros):7d} {avg_gap:8.4f} {Cf:8.4f}")
        else:
            Cf_large[N_val] = float('nan')
            print(f" {len(zeros):7d}      N/A      N/A")

    # Fit C_f = a + b/N
    Ns = np.array([N for N in Cf_large if not np.isnan(Cf_large[N])])
    Cfs = np.array([Cf_large[N] for N in Ns])
    if len(Ns) >= 3:
        A = np.column_stack([np.ones(len(Ns)), 1.0 / Ns])
        result = np.linalg.lstsq(A, Cfs, rcond=None)
        a, b = result[0]
        print(f"\n  Fit: C_f(N) ≈ {a:.4f} + {b:.4f}/N")
        print(f"  C_f(N→∞) ≈ {a:.4f}")
        print(f"  π/2 = {pi/2:.4f}")
        print(f"  Ratio C_f(∞)/(π/2) = {a/(pi/2):.4f}")

        # Also check Casimir gap prediction
        print(f"\n  Casimir gap check:")
        for N_val in Ns:
            c2_1 = casimir_suN(1, int(N_val))
            omega_01 = n_plaq * c2_1
            pred = 2 * pi / omega_01
            actual = Cf_large[int(N_val)] / n_plaq
            print(f"    N={int(N_val):3d}: C_2(fund)={c2_1:.4f}, "
                  f"ω_01=n·C_2={omega_01:.4f}, "
                  f"2π/ω_01={pred:.4f}, "
                  f"actual ⟨Δy⟩={actual:.4f}, "
                  f"ratio={actual/pred:.4f}")

    # ======================================================================
    # SUMMARY
    # ======================================================================
    print(f"\n\n  {'='*90}")
    print(f"  SUMMARY")
    print(f"  {'='*90}")

    print(f"""
  TEST 1: Fourier Spectrum
    How many modes dominate Z(y)?
    [See Part 1 peak tables above]

  TEST 2: Spacing vs Rep Gap
    Δy ≈ 2π/ω_pq for dominant pair?
    [See Part 2 zero-by-zero analysis]

  TEST 3: C_f(N→∞)
    [See Part 3 fit above]
    π/2 = {pi/2:.6f}
""")

    print(f"  [Completed in {time.time()-t0:.1f}s]")
    print("=" * 90)


if __name__ == '__main__':
    main()
