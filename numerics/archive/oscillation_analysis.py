"""
Oscillation Analysis: What produces the spacing π/2 in Z_2?
===========================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: Referee response — the {0,π}^N saddles with det=1 for N odd all have
         the SAME Phi value. So the two-saddle interference mechanism must
         involve different critical points than previously assumed.

This script:
1. Computes A_p(κ+iy) by direct numerical quadrature on SU(N)
2. Extracts oscillation frequencies via FFT
3. Identifies ALL critical points of Φ = Re Tr U on SU(N)
4. Determines which contribute to the oscillation
"""

import numpy as np
from math import comb, pi, factorial
from itertools import product


def weyl_integral_su3(func, s, n_pts=200):
    """Compute ∫_{SU(3)} func(z1,z2,z3) exp(s Re Tr U) dU via Weyl formula.

    Uses θ_3 = -(θ_1 + θ_2) constraint.
    Returns the normalized integral.
    """
    dtheta = 2 * pi / n_pts
    total = 0.0 + 0.0j
    norm = 0.0 + 0.0j

    for i1 in range(n_pts):
        t1 = i1 * dtheta
        for i2 in range(n_pts):
            t2 = i2 * dtheta
            t3 = -(t1 + t2)
            z1, z2, z3 = np.exp(1j * t1), np.exp(1j * t2), np.exp(1j * t3)

            # Vandermonde |Δ|²
            vand = abs(z1 - z2)**2 * abs(z1 - z3)**2 * abs(z2 - z3)**2

            # Phase + amplitude
            phi = (z1 + z2 + z3).real  # Re Tr U
            weight = vand * np.exp(s * phi)

            f_val = func(z1, z2, z3)
            total += f_val * weight
            norm += weight

    # Normalize: Weyl formula includes 1/(N!) × (2π)^{-(N-1)} × |Δ|²
    # We compute ratio to avoid normalization issues
    return total / norm if abs(norm) > 1e-30 else 0.0


def h_p_from_eigenvalues(p, z1, z2, z3):
    """Complete homogeneous symmetric polynomial h_p(z1, z2, z3)."""
    if p == 0:
        return 1.0
    z = np.array([z1, z2, z3])
    pk = [np.sum(z ** k) for k in range(1, p + 1)]
    h = [0.0 + 0.0j] * (p + 1)
    h[0] = 1.0
    for k in range(1, p + 1):
        h[k] = sum(pk[j] * h[k - 1 - j] for j in range(k)) / k
    return h[p]


def compute_Ap_direct_su3(p, s, n_pts=200):
    """Compute A_p(s) for SU(3) by direct quadrature.

    A_p(s) = ∫_{SU(3)} (h_p(U)/d_p) exp(s Re Tr U) dU / ∫ exp(s Re Tr U) dU
    """
    dp = comb(p + 2, 2)  # dim for SU(3)
    dtheta = 2 * pi / n_pts
    numerator = 0.0 + 0.0j
    denominator = 0.0 + 0.0j

    for i1 in range(n_pts):
        t1 = i1 * dtheta
        for i2 in range(n_pts):
            t2 = i2 * dtheta
            t3 = -(t1 + t2)
            z1, z2, z3 = np.exp(1j * t1), np.exp(1j * t2), np.exp(1j * t3)

            vand = abs(z1 - z2)**2 * abs(z1 - z3)**2 * abs(z2 - z3)**2
            phi = (z1 + z2 + z3).real
            weight = vand * np.exp(s * phi)

            hp = h_p_from_eigenvalues(p, z1, z2, z3)
            numerator += hp * weight
            denominator += weight

    return (numerator / denominator) / dp if abs(denominator) > 1e-30 else 0.0


def compute_Ap_unnorm_su3(p, s, n_pts=200):
    """Compute unnormalized A_p(s) = ∫ h_p(U) exp(s Re Tr U) |Δ|² dθ for SU(3)."""
    dp = comb(p + 2, 2)
    dtheta = 2 * pi / n_pts
    integral = 0.0 + 0.0j

    for i1 in range(n_pts):
        t1 = i1 * dtheta
        for i2 in range(n_pts):
            t2 = i2 * dtheta
            t3 = -(t1 + t2)
            z1, z2, z3 = np.exp(1j * t1), np.exp(1j * t2), np.exp(1j * t3)

            vand = abs(z1 - z2)**2 * abs(z1 - z3)**2 * abs(z2 - z3)**2
            phi = (z1 + z2 + z3).real
            weight = vand * np.exp(s * phi)

            hp = h_p_from_eigenvalues(p, z1, z2, z3)
            integral += hp * weight * dtheta * dtheta

    return integral / dp


def compute_Z2_su3(s, p_max=30, n_pts=200):
    """Compute Z_2(s) = Σ d_p [A_p(s)]² for SU(3)."""
    # First compute all A_p
    Z2 = 0.0 + 0.0j
    for p in range(p_max + 1):
        dp = comb(p + 2, 2)
        Ap = compute_Ap_unnorm_su3(p, s, n_pts)
        Z2 += dp * Ap**2
    return Z2


def main():
    print("=" * 80)
    print("  Oscillation Analysis: What produces spacing π/2?")
    print("  Author: Grzegorz Olbryk  |  March 2026  |  Referee Response")
    print("=" * 80)

    N = 3
    kappa = 1.0
    N_PTS = 150  # quadrature points per dimension
    P_MAX = 15

    # ================================================================
    # Part 1: Critical points of Φ = Re Tr U on SU(3)
    # ================================================================
    print("\n  Part 1: Critical points of Φ on SU(3)")
    print("  " + "-" * 60)

    # Find all critical points by scanning
    n_scan = 300
    dtheta = 2 * pi / n_scan
    crits = []

    for i1 in range(n_scan):
        t1 = i1 * dtheta
        for i2 in range(n_scan):
            t2 = i2 * dtheta
            t3 = -(t1 + t2)
            # Check gradient ≈ 0
            g1 = -np.sin(t1) + np.sin(t1 + t2)
            g2 = -np.sin(t2) + np.sin(t1 + t2)
            if g1**2 + g2**2 < 0.001:
                phi = np.cos(t1) + np.cos(t2) + np.cos(t3)
                z = [np.exp(1j * t1), np.exp(1j * t2), np.exp(1j * t3)]
                vand = abs(z[0]-z[1])**2 * abs(z[0]-z[2])**2 * abs(z[1]-z[2])**2
                crits.append({
                    'theta': (t1 % (2*pi), t2 % (2*pi)),
                    'Phi': round(phi, 4),
                    'vand_sq': round(vand, 4)
                })

    # Deduplicate by Phi
    phi_vals = sorted(set(c['Phi'] for c in crits))
    print(f"  Distinct Φ values at critical points: {phi_vals}")
    for pv in phi_vals:
        group = [c for c in crits if c['Phi'] == pv]
        vands = set(c['vand_sq'] for c in group)
        print(f"    Φ = {pv:6.2f}: {len(group)} critical points, |Δ|² ∈ {vands}")

    # ================================================================
    # Part 2: Compute A_p(κ+iy) for a range of y values
    # ================================================================
    print(f"\n  Part 2: A_p(κ+iy) oscillation structure (SU({N}), κ={kappa})")
    print("  " + "-" * 60)

    y_values = np.linspace(0, 40, 200)
    p_test = [0, 1, 2, 3]

    print(f"  Computing A_p for p = {p_test}, y ∈ [0, {y_values[-1]:.0f}]...")
    print(f"  Quadrature: {N_PTS}×{N_PTS} grid")

    Ap_data = {}
    for p in p_test:
        Ap_data[p] = []

    for yi, y in enumerate(y_values):
        s = kappa + 1j * y
        if yi % 50 == 0:
            print(f"    y = {y:.1f} ({yi+1}/{len(y_values)})...")
        for p in p_test:
            Ap = compute_Ap_unnorm_su3(p, s, N_PTS)
            Ap_data[p].append(Ap)

    # Convert to arrays
    for p in p_test:
        Ap_data[p] = np.array(Ap_data[p])

    # ================================================================
    # Part 3: FFT analysis of A_p oscillations
    # ================================================================
    print(f"\n  Part 3: FFT frequency analysis")
    print("  " + "-" * 60)

    dy = y_values[1] - y_values[0]
    for p in p_test:
        # FFT of A_p
        Ap_arr = Ap_data[p]
        # Remove DC component
        Ap_centered = Ap_arr - np.mean(Ap_arr)
        fft = np.fft.fft(Ap_centered)
        freqs = np.fft.fftfreq(len(Ap_arr), d=dy)
        power = np.abs(fft)**2

        # Find top 3 frequencies (positive only)
        pos_mask = freqs > 0.01
        pos_freqs = freqs[pos_mask]
        pos_power = power[pos_mask]
        top_idx = np.argsort(pos_power)[-3:][::-1]

        print(f"  A_{p}: top frequencies (as ω = 2π·freq):")
        for idx in top_idx:
            omega = 2 * pi * pos_freqs[idx]
            print(f"    ω = {omega:8.3f} (freq = {pos_freqs[idx]:8.4f}), "
                  f"power = {pos_power[idx]:.4e}")

    # ================================================================
    # Part 4: Compute Z_2(κ+iy) and find zeros
    # ================================================================
    print(f"\n  Part 4: Z_2(κ+iy) zeros and spacing")
    print("  " + "-" * 60)

    # Compute Z_2 from A_p data
    print(f"  Computing Z_2 from A_p, p=0..{P_MAX}...")

    y_fine = np.linspace(0, 30, 300)
    Z2_vals = []

    for yi, y in enumerate(y_fine):
        s = kappa + 1j * y
        if yi % 60 == 0:
            print(f"    y = {y:.1f} ({yi+1}/{len(y_fine)})...")
        Z2 = 0.0 + 0.0j
        for p in range(P_MAX + 1):
            dp = comb(p + 2, 2)
            Ap = compute_Ap_unnorm_su3(p, s, N_PTS)
            Z2 += dp * Ap**2
        Z2_vals.append(Z2)

    Z2_arr = np.array(Z2_vals)

    # Find zeros (sign changes in real part)
    zeros_y = []
    for i in range(len(Z2_arr) - 1):
        if Z2_arr[i].real * Z2_arr[i+1].real < 0:
            # Linear interpolation
            t = Z2_arr[i].real / (Z2_arr[i].real - Z2_arr[i+1].real)
            y_zero = y_fine[i] + t * (y_fine[i+1] - y_fine[i])
            zeros_y.append(y_zero)

    print(f"\n  Found {len(zeros_y)} real-part sign changes in y ∈ [0, 30]:")
    if len(zeros_y) > 0:
        for i, yz in enumerate(zeros_y[:15]):
            print(f"    y_{i} = {yz:.4f}")

    if len(zeros_y) > 1:
        gaps = np.diff(zeros_y)
        print(f"\n  Gaps between zeros:")
        for i, g in enumerate(gaps[:14]):
            print(f"    Δy_{i} = {g:.4f}  (π/2 = {pi/2:.4f})")
        print(f"\n  Mean gap: {np.mean(gaps):.4f}")
        print(f"  Expected (π/2): {pi/2:.4f}")
        print(f"  Std of gaps: {np.std(gaps):.4f}")

    # FFT of Z_2
    dy_fine = y_fine[1] - y_fine[0]
    Z2_centered = Z2_arr - np.mean(Z2_arr)
    fft_Z2 = np.fft.fft(Z2_centered)
    freqs_Z2 = np.fft.fftfreq(len(Z2_arr), d=dy_fine)
    power_Z2 = np.abs(fft_Z2)**2

    pos_mask = freqs_Z2 > 0.01
    pos_freqs = freqs_Z2[pos_mask]
    pos_power = power_Z2[pos_mask]
    top_idx = np.argsort(pos_power)[-5:][::-1]

    print(f"\n  Z_2 top FFT frequencies:")
    for idx in top_idx:
        omega = 2 * pi * pos_freqs[idx]
        spacing = pi / (omega / 2) if abs(omega) > 0.01 else float('inf')
        print(f"    ω = {omega:8.3f}, implied spacing = π/ω·2π = {spacing:.4f}, "
              f"power = {pos_power[idx]:.4e}")

    print("\n" + "=" * 80)
    print("  ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"""
  For SU(3), the critical points of Φ = Re Tr U on SU(3) are:
  - Identity: Φ = 3, |Δ|² = 0 (N-fold degenerate)
  - Balanced (1,-1,-1): Φ = -1, |Δ|² = 0 (2-fold degenerate)
  - Center ω·I (ω = e^{{2πi/3}}): Φ = -3/2, |Δ|² = 0 (3-fold degenerate)

  Key insight: For N odd, the det=1 constraint means only ONE balanced
  split survives (e.g., Φ = -1 for SU(3), not Φ = ±1). The TWO-SADDLE
  interference must be between the balanced saddle (Φ = -1) and a
  DIFFERENT saddle, not the complementary balanced split.

  If the dominant A_p oscillation is at frequency ω₁, then:
  - A_p² oscillates at 2ω₁
  - Z_2 = Σ d_p A_p² has dominant frequency 2ω₁
  - Spacing = 2π/(2ω₁) = π/ω₁
  """)


if __name__ == '__main__':
    main()
