"""
Fast Oscillation Analysis: What produces the spacing π/2?
=========================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026

Vectorized computation of A_p(κ+iy) for SU(3) to identify
the oscillation structure and resolve the saddle-point question.
"""

import numpy as np
from math import comb, pi


def precompute_grid_su3(n_pts=200):
    """Precompute the SU(3) Weyl integration grid (vectorized)."""
    theta1 = np.linspace(0, 2*pi, n_pts, endpoint=False)
    theta2 = np.linspace(0, 2*pi, n_pts, endpoint=False)
    T1, T2 = np.meshgrid(theta1, theta2)
    T3 = -(T1 + T2)

    Z1, Z2, Z3 = np.exp(1j*T1), np.exp(1j*T2), np.exp(1j*T3)

    # Vandermonde |Δ|²
    vand = np.abs(Z1 - Z2)**2 * np.abs(Z1 - Z3)**2 * np.abs(Z2 - Z3)**2

    # Φ = Re Tr U
    Phi = (Z1 + Z2 + Z3).real

    return T1, T2, T3, Z1, Z2, Z3, vand, Phi


def compute_hp_grid(p, Z1, Z2, Z3):
    """Compute h_p(z1, z2, z3) on the entire grid using Newton's identities."""
    if p == 0:
        return np.ones_like(Z1)

    # Power sums on the grid
    pk = [Z1**k + Z2**k + Z3**k for k in range(1, p+1)]

    h_prev = [np.ones_like(Z1, dtype=complex)]  # h[0] = 1
    for k in range(1, p+1):
        hk = sum(pk[j] * h_prev[k-1-j] for j in range(k)) / k
        h_prev.append(hk)

    return h_prev[p]


def compute_all_Ap(kappa, y, p_max, grid_data):
    """Compute A_p(κ+iy) for p=0..p_max in one grid sweep."""
    _, _, _, Z1, Z2, Z3, vand, Phi = grid_data
    s = kappa + 1j * y

    # Weight = |Δ|² × exp(s × Φ)
    weight = vand * np.exp(s * Phi)

    Ap = np.zeros(p_max + 1, dtype=complex)
    for p in range(p_max + 1):
        hp = compute_hp_grid(p, Z1, Z2, Z3)
        dp = comb(p + 2, 2)
        Ap[p] = np.sum(hp * weight) / dp

    return Ap


def main():
    print("=" * 80)
    print("  Fast Oscillation Analysis: SU(3)")
    print("  Author: Grzegorz Olbryk  |  March 2026")
    print("=" * 80)

    kappa = 1.0
    N_PTS = 200
    P_MAX = 20

    print(f"\n  Parameters: κ={kappa}, grid={N_PTS}², p_max={P_MAX}")
    print("  Precomputing grid...")
    grid = precompute_grid_su3(N_PTS)

    # ================================================================
    # Part 1: Critical points of Φ on SU(3)
    # ================================================================
    T1, T2, T3, Z1, Z2, Z3, vand, Phi = grid

    print("\n  Part 1: Critical points of Φ on SU(3)")
    print("  " + "-" * 60)

    # Gradient of Φ
    grad1 = -np.sin(T1) + np.sin(T1 + T2)
    grad2 = -np.sin(T2) + np.sin(T1 + T2)
    grad_norm = grad1**2 + grad2**2

    crit_mask = grad_norm < 0.01
    crit_Phi = Phi[crit_mask]
    crit_vand = vand[crit_mask]

    # Bin by Phi value
    phi_vals_rounded = np.round(crit_Phi * 4) / 4  # Round to nearest 0.25
    unique_phi = sorted(set(phi_vals_rounded))
    print(f"  Distinct Φ values: {unique_phi}")
    for pv in unique_phi:
        mask = phi_vals_rounded == pv
        vs = crit_vand[phi_vals_rounded == pv]
        print(f"    Φ ≈ {pv:6.2f}: count={np.sum(mask)}, "
              f"|Δ|² range=[{np.min(vs):.2f}, {np.max(vs):.2f}]")

    # ================================================================
    # Part 2: A_p(κ+iy) for range of y
    # ================================================================
    print(f"\n  Part 2: A_p oscillation structure")
    print("  " + "-" * 60)

    y_values = np.linspace(0, 40, 400)
    Ap_array = np.zeros((len(y_values), P_MAX + 1), dtype=complex)

    for yi, y in enumerate(y_values):
        if yi % 100 == 0:
            print(f"    y = {y:.1f} ({yi+1}/{len(y_values)})...")
        Ap_array[yi, :] = compute_all_Ap(kappa, y, P_MAX, grid)

    # ================================================================
    # Part 3: FFT of A_p oscillations
    # ================================================================
    print(f"\n  Part 3: FFT analysis of A_p")
    print("  " + "-" * 60)

    dy = y_values[1] - y_values[0]

    for p in [0, 1, 2, 3, 4, 5]:
        Ap_y = Ap_array[:, p]

        # FFT
        fft = np.fft.fft(Ap_y - np.mean(Ap_y))
        freqs = np.fft.fftfreq(len(Ap_y), d=dy)
        power = np.abs(fft)**2

        # Top positive frequencies
        pos = freqs > 0.01
        pf, pp = freqs[pos], power[pos]
        top3 = np.argsort(pp)[-3:][::-1]

        print(f"\n  A_{p} (d_p = {comb(p+2,2)}): dominant frequencies")
        for idx in top3:
            omega = 2 * pi * pf[idx]
            print(f"    ω = {omega:8.3f} (Φ ≈ {omega:.2f}), power = {pp[idx]:.4e}")

    # ================================================================
    # Part 4: Z_2 and Z_3 from A_p
    # ================================================================
    print(f"\n  Part 4: Z_n from A_p")
    print("  " + "-" * 60)

    for n in [2, 3]:
        # Z_n = Σ d_p × (A_p)^n
        Z_n = np.zeros(len(y_values), dtype=complex)
        for p in range(P_MAX + 1):
            dp = comb(p + 2, 2)
            Z_n += dp * Ap_array[:, p]**n

        # Find zeros (|Z_n| minima or Re Z_n sign changes)
        re_Z = Z_n.real
        zeros_y = []
        for i in range(len(re_Z) - 1):
            if re_Z[i] * re_Z[i+1] < 0:
                t = re_Z[i] / (re_Z[i] - re_Z[i+1])
                zeros_y.append(y_values[i] + t * (y_values[i+1] - y_values[i]))

        print(f"\n  Z_{n}: {len(zeros_y)} real-part zeros in y ∈ [0, {y_values[-1]:.0f}]")
        if len(zeros_y) > 1:
            gaps = np.diff(zeros_y)
            expected = pi / n  # π/(n·|Φ₀|) with |Φ₀|=1
            print(f"  Mean gap: {np.mean(gaps):.4f} (expected π/{n} = {expected:.4f})")
            print(f"  Std gap:  {np.std(gaps):.4f}")
            for i, g in enumerate(gaps[:10]):
                print(f"    Δy_{i} = {g:.4f}")

        # FFT of Z_n
        fft_Z = np.fft.fft(Z_n - np.mean(Z_n))
        freqs_Z = np.fft.fftfreq(len(Z_n), d=dy)
        power_Z = np.abs(fft_Z)**2

        pos = freqs_Z > 0.01
        pf, pp = freqs_Z[pos], power_Z[pos]
        top5 = np.argsort(pp)[-5:][::-1]

        print(f"\n  Z_{n} FFT (top 5):")
        for idx in top5:
            omega = 2 * pi * pf[idx]
            if abs(omega) > 0.01:
                implied_spacing = 2 * pi / omega
            else:
                implied_spacing = float('inf')
            print(f"    ω = {omega:8.3f}, spacing = 2π/ω = {implied_spacing:.4f}, "
                  f"power = {pp[idx]:.4e}")

    # ================================================================
    # Part 5: The key test — what is A_0(κ+iy) doing?
    # ================================================================
    print(f"\n  Part 5: Detailed A_0 and A_1 analysis")
    print("  " + "-" * 60)

    # A_0 = ∫ exp(s Re Tr U) |Δ|² dθ / 1
    # This is the partition function of SU(3) one-plaquette model
    A0 = Ap_array[:, 0]
    A1 = Ap_array[:, 1]

    # Print values at selected y
    print(f"\n  {'y':>6s}  {'|A_0|':>12s}  {'arg A_0':>10s}  {'|A_1|':>12s}  {'arg A_1':>10s}")
    for yi in range(0, len(y_values), 20):
        y = y_values[yi]
        a0 = A0[yi]
        a1 = A1[yi]
        print(f"  {y:6.2f}  {abs(a0):12.6f}  {np.angle(a0):10.4f}  "
              f"{abs(a1):12.6f}  {np.angle(a1):10.4f}")

    # Phase unwrapping
    phase_A0 = np.unwrap(np.angle(A0))
    phase_A1 = np.unwrap(np.angle(A1))

    # Effective frequency = dphase/dy
    dphase_A0 = np.gradient(phase_A0, dy)
    dphase_A1 = np.gradient(phase_A1, dy)

    print(f"\n  Effective frequency (dφ/dy) at large y:")
    for yi in [len(y_values)//2, 3*len(y_values)//4, len(y_values)-2]:
        y = y_values[yi]
        print(f"    y={y:5.1f}: ω_A0 = {dphase_A0[yi]:.4f}, ω_A1 = {dphase_A1[yi]:.4f}")

    print(f"\n  Mean effective frequency (y > 20):")
    mask = y_values > 20
    print(f"    A_0: <ω> = {np.mean(dphase_A0[mask]):.4f}")
    print(f"    A_1: <ω> = {np.mean(dphase_A1[mask]):.4f}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"""
  For SU(3), the critical points of Φ on SU(3) are:
    Identity:     Φ = +3.0,  |Δ|² = 0
    Balanced:     Φ = −1.0,  |Δ|² = 0
    Z₃ center:   Φ = −1.5,  |Δ|² = 0

  The det=1 constraint eliminates the (2,1)-split (Φ=+1).
  Only (1,2)-split (Φ=−1) survives as a balanced saddle.

  The two-saddle interference in A_p must therefore be between:
    (a) the balanced saddle at Φ = −1, AND
    (b) some other critical point (identity at Φ = +3, or center at Φ = −1.5)

  The frequency difference determines the spacing:
    ΔΦ = 3 − (−1) = 4  →  A_p oscillates at ω ≈ 1 (dominant freq at Φ=−1)
    A_p² oscillates at 2ω → Z_2 spacing = 2π/(2·ω) = π/ω

  Verdict: Check the FFT results above.
  """)

    print("=" * 80)


if __name__ == '__main__':
    main()
