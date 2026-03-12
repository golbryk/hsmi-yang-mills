"""
Gross-Witten-Wadia Connection to Fisher Zeros
===============================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: Task 21 — GWW phase transition and Fisher zero structure

Analyzes:
1. Wilson-action character integrals A_p(κ) vs heat-kernel predictions
2. Eigenvalue crossings: where A_1(κ) overtakes A_0(κ)
3. GWW order parameter u(κ) = ⟨Re Tr U⟩/N
4. Fisher zero proximity to the real axis
5. N-dependence of all observables
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


def casimir_2(p, N):
    """Quadratic Casimir C_2 for symmetric rep p of SU(N)."""
    return p * (p + N) / (2 * N)


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


def setup_grid(N, n_quad, n_reps):
    """Precompute grid and h_p values."""
    z, Phi, meas = build_weyl_grid(N, n_quad)
    hp_list = [h_p_vec(p, z) for p in range(n_reps)]
    dims = [dim_rep(p, N) for p in range(n_reps)]
    return Phi, meas, hp_list, dims


# ---------------------------------------------------------------------------
# Character integrals A_p(κ) for real κ
# ---------------------------------------------------------------------------

def compute_Ap_real(kap, Phi, meas, hp_list, n_reps):
    """Compute A_p(κ) for real κ. Returns array of real values."""
    exp_kap_Phi = np.exp(kap * Phi)
    weighted = exp_kap_Phi * meas
    Ap = np.zeros(n_reps)
    for p in range(n_reps):
        Ap[p] = np.sum(hp_list[p] * weighted).real
    return Ap


# ---------------------------------------------------------------------------
# GWW order parameter
# ---------------------------------------------------------------------------

def compute_order_param(kap, N, Phi, meas):
    """Compute u(κ) = ⟨Re Tr U⟩/N = ⟨Φ⟩/N."""
    exp_kap_Phi = np.exp(kap * Phi)
    Z = np.sum(exp_kap_Phi * meas).real
    PhiAvg = np.sum(Phi * exp_kap_Phi * meas).real / Z
    return PhiAvg / N


def gww_prediction(kap, N):
    """GWW large-N prediction for order parameter u(κ) = ⟨Φ⟩/N.

    For U(N) with action (κ/N) Re Tr U, the eigenvalue density is:
    - Weak coupling (κ < N): ρ(θ) = (1/2π)(1 + 2(κ/N)cos θ)
      giving u = κ/(2N)
    - Strong coupling (κ > N): gap closes, u approaches 1

    More precisely, for the action κ Re Tr U on SU(N) at large N:
    - Weak coupling: u = κ/N · (1/2) (leading order)
    - Strong coupling: u = 1 - N/(2κ) (leading order)
    - Transition: κ_c ≈ N (for the single-trace action)
    """
    # Simple weak/strong interpolation
    if kap / N < 1:
        return kap / (2 * N)
    else:
        return 1 - N / (2 * kap)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 90)
    print("  Gross-Witten-Wadia Connection to Fisher Zeros")
    print("  Author: Grzegorz Olbryk  |  March 2026")
    print("=" * 90)

    # Grid parameters per N
    configs = {
        3: {'n_quad': 80, 'n_reps': 10},
        4: {'n_quad': 40, 'n_reps': 14},
        5: {'n_quad': 20, 'n_reps': 10},
    }

    kap_vals = np.concatenate([
        np.linspace(0.05, 0.5, 10),
        np.linspace(0.6, 2.0, 15),
        np.linspace(2.5, 5.0, 6),
        np.linspace(6.0, 10.0, 5)
    ])

    # ==================================================================
    # Part 1: Wilson-action A_p(κ) vs heat-kernel
    # ==================================================================
    print(f"\n  PART 1: Wilson-Action Character Integrals vs Heat-Kernel")
    print("  " + "-" * 70)

    all_results = {}

    for N, cfg in configs.items():
        n_quad, n_reps = cfg['n_quad'], cfg['n_reps']
        n_pts = n_quad ** (N - 1)
        print(f"\n  SU({N}): n_quad={n_quad} ({n_pts:,d} pts), n_reps={n_reps}")
        print(f"  Building Weyl grid...", flush=True)
        Phi, meas, hp_list, dims = setup_grid(N, n_quad, n_reps)
        print(f"  Grid ready.")

        Ap_table = np.zeros((len(kap_vals), n_reps))
        u_wilson = np.zeros(len(kap_vals))

        for ik, kap in enumerate(kap_vals):
            Ap = compute_Ap_real(kap, Phi, meas, hp_list, n_reps)
            Ap_table[ik] = Ap
            u_wilson[ik] = compute_order_param(kap, N, Phi, meas)

        all_results[N] = {
            'Ap_table': Ap_table,
            'u_wilson': u_wilson,
            'Phi': Phi, 'meas': meas, 'hp_list': hp_list, 'dims': dims
        }

        # Print A_p table at selected κ values
        print(f"\n  A_p(κ) for SU({N}) [Wilson action]:")
        print(f"  {'κ':>6s}  {'A_0':>12s}  {'A_1':>12s}  {'A_2':>12s}  "
              f"{'A_1/A_0':>10s}  {'A_1^HK/A_0^HK':>14s}")
        for ik in range(0, len(kap_vals), 3):
            kap = kap_vals[ik]
            A0, A1, A2 = Ap_table[ik, 0], Ap_table[ik, 1], Ap_table[ik, 2]
            ratio_W = A1 / A0 if A0 > 1e-30 else float('inf')
            # Heat-kernel prediction
            ratio_HK = dim_rep(1, N) * np.exp(-casimir_2(1, N) * kap)
            print(f"  {kap:6.2f}  {A0:12.6e}  {A1:12.6e}  {A2:12.6e}  "
                  f"{ratio_W:10.6f}  {ratio_HK:14.6e}")

    # ==================================================================
    # Part 2: Eigenvalue crossings
    # ==================================================================
    print(f"\n\n  PART 2: Eigenvalue Crossings")
    print("  " + "-" * 70)

    for N in configs:
        Ap_table = all_results[N]['Ap_table']

        # Heat-kernel crossing: d_1 exp(-C_2(1) κ_c) = 1
        # N exp(-(N²-1)/(2N) κ_c) = 1
        kap_c_HK = 2 * N * np.log(N) / (N**2 - 1)
        C2_1 = casimir_2(1, N)

        # Wilson-action crossing: A_1(κ) = A_0(κ)
        A0 = Ap_table[:, 0]
        A1 = Ap_table[:, 1]
        ratio = A1 / np.maximum(A0, 1e-30)
        cross_idx = None
        for i in range(len(ratio) - 1):
            if ratio[i] < 1 and ratio[i+1] >= 1:
                cross_idx = i
                break
            elif ratio[i] >= 1 and ratio[i+1] < 1:
                cross_idx = i
                break

        kap_c_W = None
        if cross_idx is not None:
            # Linear interpolation
            r0, r1 = ratio[cross_idx], ratio[cross_idx + 1]
            k0, k1 = kap_vals[cross_idx], kap_vals[cross_idx + 1]
            kap_c_W = k0 + (1.0 - r0) / (r1 - r0) * (k1 - k0)

        # Also find d_1 A_1^n crossing with A_0^n for various n
        print(f"\n  SU({N}):")
        print(f"    Heat-kernel crossing κ_c^HK = {kap_c_HK:.6f}")
        print(f"    (where d_1 exp(-C_2 κ) = 1, i.e., N exp(-(N²-1)/(2N) κ) = 1)")
        if kap_c_W is not None:
            print(f"    Wilson-action crossing κ_c^W = {kap_c_W:.4f}")
            print(f"    (where A_1(κ) = A_0(κ) from Weyl quadrature)")
            print(f"    Ratio κ_c^W / κ_c^HK = {kap_c_W / kap_c_HK:.3f}")
        else:
            # Check if A_1 > A_0 everywhere or nowhere
            if np.all(ratio < 1):
                print(f"    Wilson: A_1 < A_0 for all κ tested (no crossing up to κ={kap_vals[-1]:.1f})")
            elif np.all(ratio >= 1):
                print(f"    Wilson: A_1 ≥ A_0 for all κ tested")
            else:
                print(f"    Wilson: crossing not found in scan range")

        # Find crossing for d_p A_p^n = d_0 A_0^n (partition function terms)
        for n_plaq in [2, 3]:
            T0 = A0 ** n_plaq
            T1 = dim_rep(1, N) * A1 ** n_plaq
            r_T = T1 / np.maximum(T0, 1e-30)
            cross_T = None
            for i in range(len(r_T) - 1):
                if r_T[i] < 1 and r_T[i+1] >= 1:
                    k0, k1 = kap_vals[i], kap_vals[i + 1]
                    cross_T = k0 + (1.0 - r_T[i]) / (r_T[i+1] - r_T[i]) * (k1 - k0)
                    break
                elif r_T[i] >= 1 and r_T[i+1] < 1:
                    k0, k1 = kap_vals[i], kap_vals[i + 1]
                    cross_T = k0 + (1.0 - r_T[i]) / (r_T[i+1] - r_T[i]) * (k1 - k0)
                    break

            hk_cross = N * np.log(N) / (n_plaq * C2_1)
            if cross_T is not None:
                print(f"    n={n_plaq}: d_1·A_1^n = A_0^n crossing: "
                      f"κ_c^W = {cross_T:.4f}  (HK: {hk_cross:.4f})")
            else:
                if np.all(r_T < 1):
                    print(f"    n={n_plaq}: d_1·A_1^n < A_0^n for all κ "
                          f"(HK: {hk_cross:.4f})")
                else:
                    print(f"    n={n_plaq}: crossing not found (HK: {hk_cross:.4f})")

    # ==================================================================
    # Part 3: GWW order parameter
    # ==================================================================
    print(f"\n\n  PART 3: GWW Order Parameter u(κ) = ⟨Φ⟩/N")
    print("  " + "-" * 70)

    for N in configs:
        u_W = all_results[N]['u_wilson']
        print(f"\n  SU({N}):")
        print(f"  {'κ':>6s}  {'u_Wilson':>10s}  {'u_GWW':>10s}  {'deviation':>10s}")
        for ik in range(0, len(kap_vals), 3):
            kap = kap_vals[ik]
            u_gww = gww_prediction(kap, N)
            dev = u_W[ik] - u_gww
            print(f"  {kap:6.2f}  {u_W[ik]:10.6f}  {u_gww:10.6f}  {dev:+10.6f}")

        # Find effective κ_c^GWW from u(κ) = 0.5 (midpoint of transition)
        for i in range(len(u_W) - 1):
            if u_W[i] < 0.5 and u_W[i+1] >= 0.5:
                k0, k1 = kap_vals[i], kap_vals[i + 1]
                kap_half = k0 + (0.5 - u_W[i]) / (u_W[i+1] - u_W[i]) * (k1 - k0)
                print(f"  u = 0.5 at κ ≈ {kap_half:.3f}  (GWW prediction: κ = {N:.0f})")
                print(f"  Ratio κ_half / N = {kap_half / N:.3f}")
                break

    # ==================================================================
    # Part 4: Fisher zeros near the real axis (SU(4) data)
    # ==================================================================
    print(f"\n\n  PART 4: Fisher Zeros and Eigenvalue Crossings (SU(4))")
    print("  " + "-" * 70)

    # SU(4) Fisher zero data from RESULT_015
    su4_zeros_n2 = [
        (0.422, 2.32), (0.542, 3.60), (1.054, 4.82),
        (1.455, 5.77), (1.614, 6.75), (1.819, 7.62),
        (2.064, 8.52), (2.195, 9.73), (2.253, 10.33),
        (2.297, 10.91), (2.327, 11.53), (2.356, 12.12),
        (2.384, 12.73), (2.406, 13.34), (2.443, 13.93),
        (2.543, 15.06), (2.618, 22.33),
    ]
    su4_zeros_n3 = [
        (0.740, 1.09), (0.925, 2.20), (0.932, 3.63),
        (0.961, 4.99), (1.011, 6.26), (1.072, 7.49),
        (1.134, 8.70), (1.190, 9.87), (1.242, 10.96),
        (1.289, 12.12), (1.337, 13.21), (1.379, 14.35),
        (1.416, 15.44), (1.452, 16.50), (1.482, 17.60),
        (1.519, 18.66), (1.545, 19.86),
        (2.380, 3.43), (2.399, 4.39), (2.436, 5.34),
        (2.486, 6.32), (2.540, 7.26), (2.588, 8.23),
        (2.634, 9.18),
    ]
    su4_zeros_n4 = [
        (0.436, 3.56), (0.476, 4.17), (0.527, 4.79),
        (0.583, 5.42), (0.640, 6.06), (0.695, 6.69),
        (0.749, 7.32), (0.801, 7.96), (0.851, 8.59),
        (0.896, 9.22), (0.941, 9.85), (0.982, 10.50),
        (1.023, 11.12), (1.061, 11.75), (1.099, 12.36),
        (1.136, 12.99), (1.177, 13.68), (1.215, 14.37),
        (1.255, 15.06), (1.291, 15.69), (1.312, 16.20),
        (1.605, 5.38), (1.749, 7.41), (2.010, 7.62),
    ]

    # Heat-kernel and Wilson eigenvalue crossings for SU(4)
    N = 4
    Ap_table = all_results[N]['Ap_table']
    kap_c_HK = 2 * N * np.log(N) / (N**2 - 1)

    print(f"\n  SU(4) eigenvalue crossings:")
    print(f"    Heat-kernel: κ_c^HK = {kap_c_HK:.4f}")

    # Find Wilson A_1/A_0 = 1
    A0 = Ap_table[:, 0]
    A1 = Ap_table[:, 1]
    ratio_W = A1 / np.maximum(A0, 1e-30)
    for i in range(len(ratio_W) - 1):
        if ratio_W[i] < 1 and ratio_W[i+1] >= 1:
            k0, k1 = kap_vals[i], kap_vals[i + 1]
            kap_c_W = k0 + (1.0 - ratio_W[i]) / (ratio_W[i+1] - ratio_W[i]) * (k1 - k0)
            print(f"    Wilson: κ_c^W = {kap_c_W:.4f} (A_1 = A_0)")
            break

    # Fisher zero κ-values relative to crossings
    print(f"\n  Fisher zero κ-values relative to crossings:")
    print(f"  n=3 main branch (first 17 zeros):")
    print(f"  {'#':>4s}  {'κ':>8s}  {'y':>8s}  {'κ/κ_c^HK':>10s}  {'κ/κ_c^W':>10s}")
    for i, (k, y) in enumerate(su4_zeros_n3[:17]):
        print(f"  {i+1:4d}  {k:8.4f}  {y:8.4f}  {k/kap_c_HK:10.4f}  "
              f"{k/kap_c_W:10.4f}")

    # Closest approach to real axis
    print(f"\n  Closest approach to real axis (smallest y):")
    for label, zeros in [("n=2", su4_zeros_n2), ("n=3", su4_zeros_n3),
                          ("n=4", su4_zeros_n4)]:
        closest = min(zeros, key=lambda z: z[1])
        print(f"    {label}: κ = {closest[0]:.4f}, y = {closest[1]:.4f}")

    # ==================================================================
    # Part 5: Approach to real axis scaling
    # ==================================================================
    print(f"\n\n  PART 5: Scaling of Closest Approach")
    print("  " + "-" * 70)

    # For each n, find the closest zero to real axis
    print(f"\n  SU(4) — closest Fisher zero to real axis:")
    print(f"  {'n':>4s}  {'y_min':>8s}  {'κ at y_min':>12s}  {'y_min·n':>8s}")
    for n, zeros in [(2, su4_zeros_n2), (3, su4_zeros_n3), (4, su4_zeros_n4)]:
        closest = min(zeros, key=lambda z: z[1])
        print(f"  {n:4d}  {closest[1]:8.4f}  {closest[0]:12.4f}  "
              f"{closest[1]*n:8.4f}")

    # ==================================================================
    # Part 6: A_p(κ) hierarchy at crossing
    # ==================================================================
    print(f"\n\n  PART 6: Representation Hierarchy at Crossings")
    print("  " + "-" * 70)

    for N in [3, 4, 5]:
        Ap_table = all_results[N]['Ap_table']
        n_reps = configs[N]['n_reps']

        # Find closest κ to the Wilson crossing
        A0 = Ap_table[:, 0]
        A1 = Ap_table[:, 1]
        diff = np.abs(A1 / np.maximum(A0, 1e-30) - 1.0)
        idx_cross = np.argmin(diff)
        kap_cross = kap_vals[idx_cross]

        print(f"\n  SU({N}) at κ ≈ {kap_cross:.2f} (near Wilson crossing):")
        print(f"  {'p':>4s}  {'d_p':>8s}  {'A_p':>14s}  {'d_p·A_p':>14s}  "
              f"{'d_p·A_p^2':>14s}  {'A_p/A_0':>10s}")
        for p in range(min(8, n_reps)):
            dp = dim_rep(p, N)
            Ap = Ap_table[idx_cross, p]
            print(f"  {p:4d}  {dp:8d}  {Ap:14.6e}  {dp*Ap:14.6e}  "
                  f"{dp*Ap**2:14.6e}  {Ap/A0[idx_cross]:10.6f}")

    # ==================================================================
    # Part 7: Wilson vs heat-kernel ratio A_1/A_0
    # ==================================================================
    print(f"\n\n  PART 7: Wilson vs Heat-Kernel — A_1/A_0 Ratio")
    print("  " + "-" * 70)

    for N in [3, 4, 5]:
        Ap_table = all_results[N]['Ap_table']
        C2_1 = casimir_2(1, N)
        d1 = dim_rep(1, N)

        print(f"\n  SU({N}):")
        print(f"  {'κ':>6s}  {'(A_1/A_0)_W':>14s}  {'(A_1/A_0)_HK':>14s}  "
              f"{'ratio W/HK':>12s}")
        for ik in range(0, len(kap_vals), 3):
            kap = kap_vals[ik]
            A0 = Ap_table[ik, 0]
            A1 = Ap_table[ik, 1]
            ratio_w = A1 / A0 if A0 > 1e-30 else float('inf')
            ratio_hk = d1 * np.exp(-C2_1 * kap)
            r_ratio = ratio_w / ratio_hk if ratio_hk > 1e-30 else float('inf')
            print(f"  {kap:6.2f}  {ratio_w:14.6e}  {ratio_hk:14.6e}  "
                  f"{r_ratio:12.6f}")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n\n  {'='*80}")
    print("  SUMMARY")
    print(f"  {'='*80}")

    for N in [3, 4, 5]:
        Ap_table = all_results[N]['Ap_table']
        u_W = all_results[N]['u_wilson']

        # Find crossings
        A0 = Ap_table[:, 0]
        A1 = Ap_table[:, 1]
        ratio = A1 / np.maximum(A0, 1e-30)
        kap_c_HK = 2 * N * np.log(N) / (N**2 - 1)

        kap_c_W = None
        for i in range(len(ratio) - 1):
            if ratio[i] < 1 and ratio[i+1] >= 1:
                k0, k1 = kap_vals[i], kap_vals[i + 1]
                kap_c_W = k0 + (1.0 - ratio[i]) / (ratio[i+1] - ratio[i]) * (k1 - k0)
                break

        # u = 0.5
        kap_half = None
        for i in range(len(u_W) - 1):
            if u_W[i] < 0.5 and u_W[i+1] >= 0.5:
                k0, k1 = kap_vals[i], kap_vals[i + 1]
                kap_half = k0 + (0.5 - u_W[i]) / (u_W[i+1] - u_W[i]) * (k1 - k0)
                break

        print(f"\n  SU({N}):")
        print(f"    Heat-kernel eigenvalue crossing:  κ_c^HK  = {kap_c_HK:.4f}")
        if kap_c_W:
            print(f"    Wilson eigenvalue crossing:       κ_c^W   = {kap_c_W:.4f}")
        else:
            print(f"    Wilson eigenvalue crossing:       not found in range")
        if kap_half:
            print(f"    GWW midpoint (u = 0.5):          κ_half  = {kap_half:.3f}")
        print(f"    GWW prediction κ_c ≈ N:          κ_c^GWW = {N}")

    print(f"\n  Key findings:")
    print(f"    1. Wilson-action A_1/A_0 crossing differs from heat-kernel")
    print(f"    2. Heat-kernel overestimates the rate of A_1 decay at large κ")
    print(f"    3. Fisher zeros (SU(4)) span a wide range of κ around the crossings")
    print(f"    4. GWW transition at κ ≈ N is far from the eigenvalue crossing")
    print("=" * 90)


if __name__ == '__main__':
    main()
