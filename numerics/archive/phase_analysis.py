"""
Phase Analysis of Fisher Zero Spacing
=======================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026

Direct verification of the two-term interference model:
At a zero, Z_n = Σ T_p = 0 where T_p = d_p A_p^n.
Between consecutive zeros, the phase of the ratio T_{dominant}/T_{partner}
should advance by 2π.

Also: smooth A_p(κ+iy) profiles as functions of y to visualize
oscillation frequencies.
"""

import numpy as np
from math import comb, pi
import sys

# Reuse infrastructure
def dim_rep(p, N): return comb(p + N - 1, N - 1)
def casimir_2(p, N): return p * (p + N) / (2 * N)
def h_p_vec(p, z):
    n_pts = z.shape[0]
    if p == 0: return np.ones(n_pts, dtype=complex)
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
def compute_Ap_complex(kap, y, Phi, meas, hp_list, n_reps):
    exp_sPhi = np.exp(kap * Phi) * np.exp(1j * y * Phi)
    weighted = exp_sPhi * meas
    Ap = np.zeros(n_reps, dtype=complex)
    for p in range(n_reps):
        Ap[p] = np.sum(hp_list[p] * weighted)
    return Ap


def main():
    print()
    print("=" * 90)
    print("  Phase Analysis of Fisher Zero Spacing")
    print("  Author: Grzegorz Olbryk  |  March 2026")
    print("=" * 90)

    N = 4
    n_quad = 40
    n_reps = 16
    print(f"\n  Building SU({N}) grid (n_quad={n_quad}, n_reps={n_reps})...",
          flush=True)
    Phi, meas, hp_list, dims = setup_grid(N, n_quad, n_reps)
    print(f"  Grid ready.")

    # ================================================================
    # Part 1: A_p profiles along vertical lines
    # ================================================================
    print(f"\n  PART 1: |A_p(κ+iy)| profiles at κ = 1.0")
    print("  " + "-" * 70)

    kap = 1.0
    ys = np.linspace(0, 15, 300)

    Ap_profile = np.zeros((len(ys), n_reps), dtype=complex)
    for iy, y in enumerate(ys):
        Ap_profile[iy] = compute_Ap_complex(kap, y, Phi, meas, hp_list, n_reps)

    # Print |A_p| at selected y values
    print(f"\n  |A_p(1.0 + iy)| for p = 0, 1, 2, 3, 5, 8:")
    print(f"  {'y':>6s}  {'|A_0|':>10s}  {'|A_1|':>10s}  {'|A_2|':>10s}  "
          f"{'|A_3|':>10s}  {'|A_5|':>10s}  {'|A_8|':>10s}")
    for iy in range(0, len(ys), 20):
        y = ys[iy]
        print(f"  {y:6.2f}  {abs(Ap_profile[iy,0]):10.4e}  "
              f"{abs(Ap_profile[iy,1]):10.4e}  {abs(Ap_profile[iy,2]):10.4e}  "
              f"{abs(Ap_profile[iy,3]):10.4e}  {abs(Ap_profile[iy,5]):10.4e}  "
              f"{abs(Ap_profile[iy,8]):10.4e}")

    # ================================================================
    # Part 2: Z_n profiles and zero detection
    # ================================================================
    print(f"\n\n  PART 2: Z_3(1.0 + iy) along imaginary axis")
    print("  " + "-" * 70)

    n_plaq = 3
    Z_profile = np.zeros(len(ys), dtype=complex)
    for iy in range(len(ys)):
        Ap = Ap_profile[iy]
        Z = sum(dims[p] * Ap[p]**n_plaq for p in range(n_reps))
        Z_profile[iy] = Z

    # Find zeros (|Z| minima)
    absZ = np.abs(Z_profile)
    zeros_y = []
    for i in range(1, len(ys) - 1):
        if absZ[i] < absZ[i-1] and absZ[i] < absZ[i+1] and absZ[i] < 0.01 * absZ[0]:
            zeros_y.append((ys[i], absZ[i]))

    print(f"  Found {len(zeros_y)} approximate zeros:")
    for i, (y, az) in enumerate(zeros_y):
        print(f"    #{i+1}: y = {y:.4f}, |Z| = {az:.4e}")

    # ================================================================
    # Part 3: Phase at exact zeros — direct term decomposition
    # ================================================================
    print(f"\n\n  PART 3: Term Decomposition at n=3 Main Branch Zeros")
    print("  " + "-" * 70)

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

    # At each zero, compute T_p = d_p A_p^n and its phase
    for i, (kap, y) in enumerate(zeros_n3_main):
        Ap = compute_Ap_complex(kap, y, Phi, meas, hp_list, n_reps)
        Tp = np.array([dims[p] * Ap[p]**n_plaq for p in range(n_reps)])
        Z = np.sum(Tp)

        # Find the two largest |T_p|
        absT = np.abs(Tp)
        top2_idx = np.argsort(absT)[-2:][::-1]
        p_dom = top2_idx[0]
        p_part = top2_idx[1]

        # Phase of each term
        phase_dom = np.angle(Tp[p_dom])
        phase_part = np.angle(Tp[p_part])
        delta_phase = phase_part - phase_dom
        delta_phase = (delta_phase + pi) % (2*pi) - pi

        print(f"\n  Zero #{i+1}: κ={kap:.4f}, y={y:.4f}, |Z|={abs(Z):.2e}")
        print(f"    Top terms:")
        for j in range(min(5, n_reps)):
            idx = np.argsort(absT)[::-1][j]
            phase = np.angle(Tp[idx])
            print(f"      p={idx:2d}: |T_p| = {absT[idx]:.4e}, "
                  f"arg(T_p) = {phase:+.4f} rad ({np.degrees(phase):+.1f}°)")
        print(f"    Dominant pair: p={p_dom}, p={p_part}")
        print(f"    Phase difference: {delta_phase:.4f} rad "
              f"({np.degrees(delta_phase):.1f}°)")

    # ================================================================
    # Part 4: Phase tracking between consecutive zeros
    # ================================================================
    print(f"\n\n  PART 4: Phase Tracking Between Consecutive Zeros")
    print("  " + "-" * 70)

    # For each pair of consecutive zeros, track arg(T_0) and arg(T_{partner})
    # at many y points between them
    print(f"\n  Accumulated phase of T_0 = A_0^3 between consecutive zeros:")
    print(f"  {'pair':>8s}  {'y1':>7s}  {'y2':>7s}  {'Δ(arg T_0)':>12s}  "
          f"{'Δ(arg T_0)/π':>14s}  {'Δ(arg T_0)/2π':>14s}")

    for i in range(len(zeros_n3_main) - 1):
        kap1, y1 = zeros_n3_main[i]
        kap2, y2 = zeros_n3_main[i+1]

        # Evaluate at many points along the branch (linear interpolation in κ)
        n_steps = 100
        phases_T0 = []
        for j in range(n_steps + 1):
            t = j / n_steps
            kap_j = kap1 + t * (kap2 - kap1)
            y_j = y1 + t * (y2 - y1)
            Ap = compute_Ap_complex(kap_j, y_j, Phi, meas, hp_list, n_reps)
            T0 = Ap[0]**n_plaq
            phases_T0.append(np.angle(T0))

        # Unwrap phases
        phases_T0 = np.unwrap(phases_T0)
        delta_phase_T0 = phases_T0[-1] - phases_T0[0]

        print(f"  {i+1}→{i+2}     {y1:7.4f}  {y2:7.4f}  "
              f"{delta_phase_T0:12.4f}  {delta_phase_T0/pi:14.4f}  "
              f"{delta_phase_T0/(2*pi):14.4f}")

    # ================================================================
    # Part 5: Effective oscillation frequency from phase accumulation
    # ================================================================
    print(f"\n\n  PART 5: Effective Oscillation Frequencies")
    print("  " + "-" * 70)

    # Evaluate arg(A_0^3) along the main branch from y=1 to y=11
    kap_interp = np.interp(
        np.linspace(1, 11, 200),
        [z[1] for z in zeros_n3_main],
        [z[0] for z in zeros_n3_main]
    )
    ys_branch = np.linspace(1, 11, 200)

    phase_A0n = []
    phase_A1n = []
    phase_A3n = []
    for j in range(len(ys_branch)):
        Ap = compute_Ap_complex(kap_interp[j], ys_branch[j], Phi, meas,
                                hp_list, n_reps)
        phase_A0n.append(np.angle(Ap[0]**n_plaq))
        phase_A1n.append(np.angle(dims[1] * Ap[1]**n_plaq))
        phase_A3n.append(np.angle(dims[3] * Ap[3]**n_plaq))

    phase_A0n = np.unwrap(phase_A0n)
    phase_A1n = np.unwrap(phase_A1n)
    phase_A3n = np.unwrap(phase_A3n)

    # Compute effective frequency as slope
    dy = ys_branch[1] - ys_branch[0]
    omega_A0n = np.gradient(phase_A0n, dy)
    omega_A1n = np.gradient(phase_A1n, dy)

    print(f"  Total phase accumulation of A_0^3 from y=1 to y=11:")
    print(f"    Δ(arg A_0^3) = {phase_A0n[-1] - phase_A0n[0]:.4f} rad "
          f"= {(phase_A0n[-1] - phase_A0n[0])/pi:.2f}π "
          f"= {(phase_A0n[-1] - phase_A0n[0])/(2*pi):.2f} turns")
    print(f"    Average ω_0 = {np.mean(omega_A0n):.4f} rad/y")
    print(f"  Total phase accumulation of d_1·A_1^3:")
    print(f"    Δ(arg T_1) = {phase_A1n[-1] - phase_A1n[0]:.4f} rad "
          f"= {(phase_A1n[-1] - phase_A1n[0])/pi:.2f}π")
    print(f"    Average ω_1 = {np.mean(omega_A1n):.4f} rad/y")

    print(f"\n  Number of zeros expected from A_0^3 oscillation: "
          f"{abs(phase_A0n[-1] - phase_A0n[0])/(2*pi):.1f}")
    print(f"  Number of zeros observed in y ∈ [1, 11]: "
          f"{len([z for z in zeros_n3_main if 1 < z[1] < 11])}")

    # ================================================================
    # Part 6: Fixed-κ vertical scan
    # ================================================================
    print(f"\n\n  PART 6: Phase of A_0^3 Along Fixed κ = 1.0")
    print("  " + "-" * 70)

    kap_fix = 1.0
    ys_fix = np.linspace(0, 15, 500)
    phase_fix = []
    abs_A0n_fix = []
    for y in ys_fix:
        Ap = compute_Ap_complex(kap_fix, y, Phi, meas, hp_list, n_reps)
        T0 = Ap[0]**n_plaq
        phase_fix.append(np.angle(T0))
        abs_A0n_fix.append(abs(T0))

    phase_fix = np.unwrap(phase_fix)
    dy = ys_fix[1] - ys_fix[0]
    omega_fix = np.gradient(phase_fix, dy)

    print(f"  Total phase of A_0^3 from y=0 to y=15: "
          f"{phase_fix[-1] - phase_fix[0]:.4f} rad = "
          f"{(phase_fix[-1] - phase_fix[0])/(2*pi):.2f} turns")
    print(f"  Expected #zeros if spacing = π/(n|Φ₀|) = π/3 ≈ 1.05: "
          f"{15/1.05:.0f}")
    print(f"  Expected from phase accumulation: "
          f"{abs(phase_fix[-1] - phase_fix[0])/(2*pi):.1f}")

    # Effective frequency at selected y
    print(f"\n  Effective frequency ω₀ = d/dy [arg A_0^3] at κ=1.0:")
    print(f"  {'y':>6s}  {'ω₀':>10s}  {'|A_0|^3':>12s}")
    for iy in range(0, len(ys_fix), 33):
        print(f"  {ys_fix[iy]:6.2f}  {omega_fix[iy]:10.4f}  "
              f"{abs_A0n_fix[iy]:12.4e}")

    # Predicted average spacing from average ω₀
    omega_avg = np.mean(omega_fix[50:400])  # skip early transient
    print(f"\n  Average ω₀ (y ∈ [1.5, 12]): {omega_avg:.4f} rad/y")
    print(f"  If ALL phase comes from A_0^3: Δy ≈ 2π/|ω₀| = {2*pi/abs(omega_avg):.4f}")
    print(f"  If dominant pair: Δy ≈ π/|ω₀| = {pi/abs(omega_avg):.4f}")
    print(f"  Observed: Δy ≈ 1.22 (n=3 main branch)")

    # ================================================================
    # Part 7: Summary
    # ================================================================
    print(f"\n\n  {'='*80}")
    print(f"  SUMMARY")
    print(f"  {'='*80}")
    print(f"""
  The Fisher zero spacing for N ≡ 0 mod 4 is determined by the rate of
  phase accumulation of A_0^n(κ+iy) along the zero branch.

  For A_0(κ+iy) = ∫ ρ_0(Φ; κ) exp(iyΦ) dΦ:
  - The phase of A_0^n accumulates as arg(A_0^n) ~ n·⟨Φ⟩_eff · y
  - But ⟨Φ⟩_eff is NOT constant — it depends on y through stationary phase
  - At moderate y, ω_0 ≈ ⟨Φ⟩_κ (the κ-weighted mean of Φ)
  - At large y, ω_0 → Φ_max = N (boundary contribution)

  The spacing emerges from the two-term interference:
  Z_n = A_0^n + d_partner * A_partner^n + [smaller terms]
  Zeros when the dominant pair has opposite phase.
  Consecutive zeros correspond to a ~2π phase advance of A_0^n.
""")
    print("=" * 90)


if __name__ == '__main__':
    main()
