"""
Wilson-Action Fisher Zero Spacing Analysis
============================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: Task 22 — Wilson-action spacing formula

Derives and verifies the spacing formula for Wilson-action Fisher zeros:
- For |Φ₀| ≠ 0 (N odd, N ≡ 2 mod 4): saddle-point → Δy = π/(n|Φ₀|)
- For |Φ₀| = 0 (N ≡ 0 mod 4): representation interference → Δy = 2π/(n|ω_{p*} - ω_0|)

Computes phase velocities ω_p(y) = d/dy [arg A_p(κ+iy)] at Fisher zero locations
and tests whether the spacing is predicted by the representation interference formula.
"""

import numpy as np
from math import comb, pi
import sys


# ---------------------------------------------------------------------------
# SU(N) utilities (from su4_newton_search.py)
# ---------------------------------------------------------------------------

def dim_rep(p, N):
    return comb(p + N - 1, N - 1)

def casimir_2(p, N):
    return p * (p + N) / (2 * N)

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

def setup_grid(N, n_quad, n_reps):
    z, Phi, meas = build_weyl_grid(N, n_quad)
    hp_list = [h_p_vec(p, z) for p in range(n_reps)]
    dims = [dim_rep(p, N) for p in range(n_reps)]
    return Phi, meas, hp_list, dims


# ---------------------------------------------------------------------------
# Compute A_p(κ + iy) complex
# ---------------------------------------------------------------------------

def compute_Ap_complex(kap, y, Phi, meas, hp_list, n_reps):
    """Compute A_p(κ + iy) as complex values."""
    exp_sPhi = np.exp(kap * Phi) * np.exp(1j * y * Phi)
    weighted = exp_sPhi * meas
    Ap = np.zeros(n_reps, dtype=complex)
    for p in range(n_reps):
        Ap[p] = np.sum(hp_list[p] * weighted)
    return Ap


def compute_phase_velocity(kap, y, Phi, meas, hp_list, n_reps, dy=0.01):
    """Compute ω_p(y) = d/dy [arg A_p(κ+iy)] via finite difference."""
    Ap_minus = compute_Ap_complex(kap, y - dy, Phi, meas, hp_list, n_reps)
    Ap_plus = compute_Ap_complex(kap, y + dy, Phi, meas, hp_list, n_reps)

    omega = np.zeros(n_reps)
    for p in range(n_reps):
        if abs(Ap_minus[p]) > 1e-30 and abs(Ap_plus[p]) > 1e-30:
            dphase = np.angle(Ap_plus[p]) - np.angle(Ap_minus[p])
            # Unwrap
            dphase = (dphase + pi) % (2 * pi) - pi
            omega[p] = dphase / (2 * dy)
    return omega


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 90)
    print("  Wilson-Action Fisher Zero Spacing Analysis")
    print("  Author: Grzegorz Olbryk  |  March 2026  |  Task 22")
    print("=" * 90)

    # ===================================================================
    # Part A: SU(4) n=3 main branch — representation interference
    # ===================================================================
    print(f"\n  PART A: SU(4) n=3 Main Branch — Phase Velocity Analysis")
    print("  " + "-" * 70)

    N = 4
    n_quad = 40
    n_reps = 16
    n_plaq = 3

    print(f"  Building SU({N}) grid: n_quad={n_quad}, n_reps={n_reps}...")
    sys.stdout.flush()
    Phi, meas, hp_list, dims = setup_grid(N, n_quad, n_reps)
    print(f"  Grid ready.")

    # n=3 main branch zeros from RESULT_015
    zeros_n3_main = [
        (0.7445, 1.0898),
        (0.8970, 3.6279),
        (0.9681, 4.9890),
        (0.9991, 6.2593),
        (1.0232, 7.4922),
        (1.0388, 8.6978),
        (1.0459, 9.8658),
        (1.1682, 10.9575),
    ]

    # Extended zeros (from the summary, approximately)
    zeros_n3_extended = [
        (0.7445, 1.0898),
        (0.8970, 3.6279),
        (0.9681, 4.9890),
        (0.9991, 6.2593),
        (1.0232, 7.4922),
        (1.0388, 8.6978),
        (1.0459, 9.8658),
        (1.1682, 10.9575),
        (1.2420, 12.1200),
        (1.2890, 13.2100),
        (1.3370, 14.3500),
        (1.3790, 15.4400),
        (1.4160, 16.5000),
        (1.4520, 17.6000),
        (1.4820, 18.6600),
        (1.5190, 19.8600),
    ]

    print(f"\n  Phase velocities at zero locations (n={n_plaq}):")
    print(f"  {'#':>3s}  {'κ':>7s}  {'y':>7s}  {'ω_0':>10s}  {'ω_1':>10s}  "
          f"{'ω_2':>10s}  {'ω_3':>10s}  {'partner':>8s}  {'Δy_obs':>8s}  "
          f"{'Δy_pred':>8s}")

    prev_y = None
    for i, (kap, y) in enumerate(zeros_n3_extended):
        # Compute phase velocities
        omega = compute_phase_velocity(kap, y, Phi, meas, hp_list, n_reps,
                                       dy=0.005)

        # Compute |d_p A_p^n| to find partner
        Ap = compute_Ap_complex(kap, y, Phi, meas, hp_list, n_reps)
        T = np.array([dims[p] * abs(Ap[p])**n_plaq for p in range(n_reps)])

        # Find partner: largest T_p for p > 0
        partner = np.argmax(T[1:]) + 1

        # Phase difference rate
        delta_omega = abs(omega[partner] - omega[0])
        Dy_pred = 2 * pi / (n_plaq * delta_omega) if delta_omega > 1e-10 else float('inf')

        # Observed gap
        Dy_obs = y - prev_y if prev_y is not None else float('nan')
        prev_y = y

        print(f"  {i+1:3d}  {kap:7.4f}  {y:7.4f}  {omega[0]:10.4f}  "
              f"{omega[1]:10.4f}  {omega[2]:10.4f}  {omega[3]:10.4f}  "
              f"p={partner:2d}      {Dy_obs:8.4f}  {Dy_pred:8.4f}")

    # ===================================================================
    # Part B: n=3 secondary branch
    # ===================================================================
    print(f"\n\n  PART B: SU(4) n=3 Secondary Branch")
    print("  " + "-" * 70)

    zeros_n3_sec = [
        (1.7976, 1.9165),
        (2.2236, 2.7531),
        (2.5289, 3.7407),
        (2.7358, 4.7214),
        (2.9200, 5.7007),
    ]

    print(f"  {'#':>3s}  {'κ':>7s}  {'y':>7s}  {'ω_0':>10s}  {'ω_1':>10s}  "
          f"{'ω_2':>10s}  {'partner':>8s}  {'Δy_obs':>8s}  {'Δy_pred':>8s}")

    prev_y = None
    for i, (kap, y) in enumerate(zeros_n3_sec):
        omega = compute_phase_velocity(kap, y, Phi, meas, hp_list, n_reps,
                                       dy=0.005)
        Ap = compute_Ap_complex(kap, y, Phi, meas, hp_list, n_reps)
        T = np.array([dims[p] * abs(Ap[p])**n_plaq for p in range(n_reps)])
        partner = np.argmax(T[1:]) + 1
        delta_omega = abs(omega[partner] - omega[0])
        Dy_pred = 2 * pi / (n_plaq * delta_omega) if delta_omega > 1e-10 else float('inf')
        Dy_obs = y - prev_y if prev_y is not None else float('nan')
        prev_y = y

        print(f"  {i+1:3d}  {kap:7.4f}  {y:7.4f}  {omega[0]:10.4f}  "
              f"{omega[1]:10.4f}  {omega[2]:10.4f}  "
              f"p={partner:2d}      {Dy_obs:8.4f}  {Dy_pred:8.4f}")

    # ===================================================================
    # Part C: n=4 main branch
    # ===================================================================
    print(f"\n\n  PART C: SU(4) n=4 Main Branch")
    print("  " + "-" * 70)

    n_plaq_4 = 4
    zeros_n4 = [
        (1.0391, 3.5610),
        (1.1807, 4.3123),
        (1.1348, 4.9189),
        (1.2010, 5.5869),
        (1.1916, 6.1947),
        (1.2366, 6.8454),
        (1.2390, 7.4535),
        (1.2691, 8.0970),
        (1.2704, 8.7018),
        (1.2942, 9.3382),
        (1.2988, 9.9518),
        (1.3453, 10.6106),
        (1.4519, 11.2400),
        (1.6077, 11.7783),
        (1.6931, 12.2970),
    ]

    print(f"  {'#':>3s}  {'κ':>7s}  {'y':>7s}  {'ω_0':>10s}  {'ω_1':>10s}  "
          f"{'partner':>8s}  {'Δy_obs':>8s}  {'Δy_pred':>8s}")

    prev_y = None
    for i, (kap, y) in enumerate(zeros_n4):
        omega = compute_phase_velocity(kap, y, Phi, meas, hp_list, n_reps,
                                       dy=0.005)
        Ap = compute_Ap_complex(kap, y, Phi, meas, hp_list, n_reps)
        T = np.array([dims[p] * abs(Ap[p])**n_plaq_4 for p in range(n_reps)])
        partner = np.argmax(T[1:]) + 1
        delta_omega = abs(omega[partner] - omega[0])
        Dy_pred = 2 * pi / (n_plaq_4 * delta_omega) if delta_omega > 1e-10 else float('inf')
        Dy_obs = y - prev_y if prev_y is not None else float('nan')
        prev_y = y

        print(f"  {i+1:3d}  {kap:7.4f}  {y:7.4f}  {omega[0]:10.4f}  "
              f"{omega[1]:10.4f}  "
              f"p={partner:2d}      {Dy_obs:8.4f}  {Dy_pred:8.4f}")

    # ===================================================================
    # Part D: Saddle-point verification for N=3 (|Φ₀| = 1)
    # ===================================================================
    print(f"\n\n  PART D: Saddle-Point Mechanism — SU(3) n=2 (|Φ₀| = 1)")
    print("  " + "-" * 70)

    N3 = 3
    n_quad3 = 80
    n_reps3 = 10
    n_plaq_2 = 2

    print(f"  Building SU(3) grid...", flush=True)
    Phi3, meas3, hp3, dims3 = setup_grid(N3, n_quad3, n_reps3)
    print(f"  Grid ready.")

    # Compute Z_{2P}^{SU(3)}(1.0 + iy) for a range of y to find zeros
    kap3 = 1.0
    ys = np.linspace(0.5, 20, 2000)
    Z_vals = np.zeros(len(ys), dtype=complex)
    for iy, y in enumerate(ys):
        Ap = compute_Ap_complex(kap3, y, Phi3, meas3, hp3, n_reps3)
        Z = sum(dims3[p] * Ap[p]**n_plaq_2 for p in range(n_reps3))
        Z_vals[iy] = Z

    # Find sign changes in Re Z (approximate zeros)
    zeros_su3 = []
    for i in range(len(ys) - 1):
        if Z_vals[i].real * Z_vals[i+1].real < 0 or Z_vals[i].imag * Z_vals[i+1].imag < 0:
            # Interpolate
            y_zero = ys[i] + (ys[i+1] - ys[i]) * abs(Z_vals[i]) / (abs(Z_vals[i]) + abs(Z_vals[i+1]))
            zeros_su3.append(y_zero)

    # Remove duplicates (within 0.1)
    unique_zeros = [zeros_su3[0]] if zeros_su3 else []
    for z in zeros_su3[1:]:
        if z - unique_zeros[-1] > 0.1:
            unique_zeros.append(z)

    print(f"\n  SU(3) n=2 zeros at κ={kap3} (|Φ₀| = 1, predicted Δy = π/2 = {pi/2:.4f}):")
    print(f"  {'#':>3s}  {'y':>8s}  {'Δy':>8s}  {'Δy/(π/2)':>10s}")
    prev = None
    for i, y in enumerate(unique_zeros[:15]):
        gap = y - prev if prev else float('nan')
        ratio = gap / (pi/2) if prev else float('nan')
        print(f"  {i+1:3d}  {y:8.4f}  {gap:8.4f}  {ratio:10.4f}")
        prev = y

    # Phase velocities at SU(3) zeros
    print(f"\n  Phase velocities at SU(3) zeros:")
    print(f"  {'#':>3s}  {'y':>7s}  {'ω_0':>10s}  {'ω_1':>10s}  {'ω_2':>10s}  "
          f"{'n|ω_1-ω_0|':>12s}  {'2π/n|Δω|':>12s}")
    for i, y in enumerate(unique_zeros[:8]):
        omega = compute_phase_velocity(kap3, y, Phi3, meas3, hp3, n_reps3,
                                       dy=0.005)
        dw = abs(omega[1] - omega[0])
        Dy_pred = 2 * pi / (n_plaq_2 * dw) if dw > 1e-10 else float('inf')
        print(f"  {i+1:3d}  {y:7.4f}  {omega[0]:10.4f}  {omega[1]:10.4f}  "
              f"{omega[2]:10.4f}  {n_plaq_2*dw:12.4f}  {Dy_pred:12.4f}")

    # ===================================================================
    # Part E: Effective frequency from Fourier analysis
    # ===================================================================
    print(f"\n\n  PART E: Effective Φ-Centroids of ρ_p(Φ)")
    print("  " + "-" * 70)
    print(f"  The effective frequency ω_p ≈ ⟨Φ⟩_p where ⟨Φ⟩_p is the mean of")
    print(f"  ρ_p(Φ; κ) = h_p(U) exp(κ Φ) / ∫ h_p exp(κΦ) dμ")

    for N, n_q, n_r, kap_val in [(3, 80, 10, 1.0), (4, 40, 16, 1.0)]:
        print(f"\n  SU({N}) at κ = {kap_val}:")
        z, PhiN, measN = build_weyl_grid(N, n_q)
        hpN = [h_p_vec(p, z) for p in range(n_r)]

        # exp(κΦ) weight
        w_kap = np.exp(kap_val * PhiN) * measN

        print(f"  {'p':>4s}  {'⟨Φ⟩_p':>10s}  {'⟨Φ²⟩_p':>10s}  {'σ_Φ':>10s}  "
              f"{'ω_p(y=5)':>10s}  {'C_2(p)':>8s}")
        for p in range(min(8, n_r)):
            hp_real = hpN[p].real  # for symmetric reps, h_p can be complex
            hp_abs = np.abs(hpN[p])
            # Use |h_p| as weight (h_p can be complex for SU(N>=3))
            w_p = hp_abs * np.exp(kap_val * PhiN) * measN
            norm = np.sum(w_p)
            if norm > 1e-30:
                mean_Phi = np.sum(PhiN * w_p) / norm
                mean_Phi2 = np.sum(PhiN**2 * w_p) / norm
                sigma = np.sqrt(max(0, mean_Phi2 - mean_Phi**2))
            else:
                mean_Phi = 0
                sigma = 0

            # Actual phase velocity at y=5
            Phi_full, meas_full = PhiN, measN
            omega_5 = compute_phase_velocity(kap_val, 5.0, Phi_full, meas_full,
                                              hpN, n_r, dy=0.005)
            c2 = casimir_2(p, N)
            print(f"  {p:4d}  {mean_Phi:10.4f}  {mean_Phi2:10.4f}  "
                  f"{sigma:10.4f}  {omega_5[p]:10.4f}  {c2:8.4f}")

    # ===================================================================
    # Part F: Summary — Two-mechanism spacing formula
    # ===================================================================
    print(f"\n\n  {'='*80}")
    print(f"  SUMMARY: Two-Mechanism Spacing Formula")
    print(f"  {'='*80}")
    print(f"""
  Wilson-action Fisher zero spacing has TWO distinct mechanisms:

  MECHANISM 1 (|Φ₀| ≠ 0, saddle-point):
    A_0(κ+iy) oscillates due to contributions at Φ = ±|Φ₀|
    Z_n ~ A_0^n oscillates at frequency n|Φ₀|
    Spacing: Δy = π / (n|Φ₀|)
    Valid for: N odd (|Φ₀| = 1), N ≡ 2 mod 4 (|Φ₀| = 2)

  MECHANISM 2 (|Φ₀| = 0, representation interference):
    A_0(κ+iy) decays without oscillation (stationary point at Φ=0)
    Cancellation comes from A_0^n vs d_p* A_p*^n for partner p*
    Spacing: Δy = 2π / (n|ω_p*(y) - ω_0(y)|)
    where ω_p(y) = d/dy [arg A_p(κ+iy)]
    Valid for: N ≡ 0 mod 4

  The two mechanisms are unified by the observation that the spacing
  is always determined by the rate of phase accumulation between the
  two dominant interfering contributions to Z_n.
""")
    print("=" * 90)


if __name__ == '__main__':
    main()
