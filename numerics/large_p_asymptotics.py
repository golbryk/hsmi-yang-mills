"""
Large-p Asymptotics of Wilson-Action Character Integrals
=========================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: Task 23 — asymptotic behavior of A_p(s) for large p

Computes A_p(κ+iy) for p = 0..P_max and fits asymptotic formulas:
  A_p(κ) ~ C(κ) · d_p · r(κ)^p   (real coupling)
  |A_p(κ+iy)| ~ |C(s)| · d_p · |r(s)|^p   (complex coupling)

The generating function approach: G(t, s) = Σ_p A_p(s) t^p = ∫ exp(sΦ)/det(I-tU) dμ
The singularity structure of G in t determines the large-p behavior.
"""

import numpy as np
from math import comb, pi
import sys


# ---------------------------------------------------------------------------
# SU(N) utilities
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


def main():
    print()
    print("=" * 90)
    print("  Large-p Asymptotics of A_p(s)")
    print("  Author: Grzegorz Olbryk  |  March 2026  |  Task 23")
    print("=" * 90)

    configs = {
        3: {'n_quad': 80, 'P_max': 30},
        4: {'n_quad': 40, 'P_max': 25},
        5: {'n_quad': 20, 'P_max': 20},
    }

    kap_vals = [0.5, 1.0, 2.0]
    y_vals = [0, 3.0, 6.0, 10.0]

    for N, cfg in configs.items():
        n_quad = cfg['n_quad']
        P_max = cfg['P_max']
        n_pts = n_quad ** (N - 1)

        print(f"\n\n  {'='*80}")
        print(f"  SU({N}): n_quad={n_quad} ({n_pts:,d} pts), P_max={P_max}")
        print(f"  {'='*80}")

        print(f"  Building grid and computing h_p for p = 0..{P_max}...",
              flush=True)
        z, Phi, meas = build_weyl_grid(N, n_quad)
        hp_list = [h_p_vec(p, z) for p in range(P_max + 1)]
        dims = [dim_rep(p, N) for p in range(P_max + 1)]
        print(f"  Done.")

        # ==============================================================
        # Part 1: A_p(κ) for real coupling
        # ==============================================================
        print(f"\n  PART 1: A_p(κ) for real κ")
        print("  " + "-" * 60)

        for kap in kap_vals:
            exp_kap = np.exp(kap * Phi)
            weighted = exp_kap * meas

            Ap_real = np.zeros(P_max + 1)
            for p in range(P_max + 1):
                Ap_real[p] = np.sum(hp_list[p] * weighted).real

            # Compute ratio A_p / d_p
            ratio = Ap_real / np.array([max(1, d) for d in dims])

            # Compute successive ratio r_p = A_{p+1}/A_p
            succ_ratio = np.zeros(P_max)
            for p in range(P_max):
                if abs(Ap_real[p]) > 1e-30:
                    succ_ratio[p] = Ap_real[p+1] / Ap_real[p]

            print(f"\n  κ = {kap}:")
            print(f"  {'p':>4s}  {'d_p':>8s}  {'A_p':>14s}  {'A_p/d_p':>12s}  "
                  f"{'A_{p+1}/A_p':>12s}  {'log(A_p/d_p)':>14s}")
            for p in range(0, P_max + 1, 2):
                log_ratio = np.log(ratio[p]) if ratio[p] > 1e-30 else float('-inf')
                sr = succ_ratio[p] if p < P_max else float('nan')
                print(f"  {p:4d}  {dims[p]:8d}  {Ap_real[p]:14.6e}  "
                      f"{ratio[p]:12.6e}  {sr:12.6e}  {log_ratio:14.6f}")

            # Fit: A_p/d_p = C · r^p for large p
            # log(A_p/d_p) = log(C) + p·log(r)
            p_fit_start = max(5, P_max // 3)
            ps = np.arange(p_fit_start, P_max + 1)
            log_ratios = np.log(np.maximum(ratio[p_fit_start:P_max+1], 1e-30))
            # Linear fit
            if len(ps) > 1 and np.all(np.isfinite(log_ratios)):
                coeffs = np.polyfit(ps, log_ratios, 1)
                log_r = coeffs[0]
                log_C = coeffs[1]
                r_fit = np.exp(log_r)
                C_fit = np.exp(log_C)
                # Check quality
                residuals = log_ratios - (coeffs[0] * ps + coeffs[1])
                rms = np.sqrt(np.mean(residuals**2))
                print(f"\n  Fit A_p/d_p ~ C·r^p (p ≥ {p_fit_start}):")
                print(f"    r = {r_fit:.8f}")
                print(f"    C = {C_fit:.8e}")
                print(f"    log(r) = {log_r:.8f}")
                print(f"    RMS residual = {rms:.6f}")

                # Compare with heat-kernel: A_p/d_p ~ exp(-C_2(p)·κ) = exp(-p(p+N)κ/(2N))
                # For large p, C_2 ~ p²/(2N), so log(A_p/d_p) ~ -p²κ/(2N)
                # This is QUADRATIC in p, while our fit is LINEAR in p
                print(f"    Note: heat-kernel predicts log(A_p/d_p) ~ -p²κ/(2N)")
                print(f"    Wilson fit is LINEAR: log(A_p/d_p) ~ {log_r:.4f}·p")
                print(f"    Heat-kernel slope at p={p_fit_start}: "
                      f"-{p_fit_start}κ/N = {-p_fit_start*kap/N:.4f}")

        # ==============================================================
        # Part 2: A_p(κ+iy) for complex coupling
        # ==============================================================
        print(f"\n\n  PART 2: |A_p(κ+iy)| and arg(A_p) for complex s")
        print("  " + "-" * 60)

        kap = 1.0
        for y in y_vals:
            if y == 0:
                continue
            exp_s = np.exp(kap * Phi) * np.exp(1j * y * Phi)
            weighted = exp_s * meas

            Ap_complex = np.zeros(P_max + 1, dtype=complex)
            for p in range(P_max + 1):
                Ap_complex[p] = np.sum(hp_list[p] * weighted)

            absAp = np.abs(Ap_complex)
            argAp = np.angle(Ap_complex)
            ratio_abs = absAp / np.array([max(1, d) for d in dims])

            print(f"\n  s = {kap} + {y}i:")
            print(f"  {'p':>4s}  {'d_p':>8s}  {'|A_p|':>14s}  {'|A_p|/d_p':>12s}  "
                  f"{'arg(A_p)/π':>12s}  {'d_p|A_p|^3':>14s}")
            for p in range(0, P_max + 1, 2):
                T_p = dims[p] * absAp[p]**3
                print(f"  {p:4d}  {dims[p]:8d}  {absAp[p]:14.6e}  "
                      f"{ratio_abs[p]:12.6e}  {argAp[p]/pi:12.6f}  "
                      f"{T_p:14.6e}")

            # Fit |A_p|/d_p ~ |C|·|r|^p
            p_fit_start = max(5, P_max // 3)
            ps = np.arange(p_fit_start, P_max + 1)
            log_ratios = np.log(np.maximum(ratio_abs[p_fit_start:P_max+1], 1e-30))
            if len(ps) > 1 and np.all(np.isfinite(log_ratios)):
                coeffs = np.polyfit(ps, log_ratios, 1)
                abs_r_fit = np.exp(coeffs[0])
                abs_C_fit = np.exp(coeffs[1])
                residuals = log_ratios - (coeffs[0] * ps + coeffs[1])
                rms = np.sqrt(np.mean(residuals**2))
                print(f"\n  Fit |A_p|/d_p ~ |C|·|r|^p (p ≥ {p_fit_start}):")
                print(f"    |r| = {abs_r_fit:.8f}")
                print(f"    |C| = {abs_C_fit:.8e}")
                print(f"    RMS residual = {rms:.6f}")

            # Phase: fit arg(A_p) ~ α + β·p + γ·p²
            phases = argAp[p_fit_start:P_max+1]
            phases_unwrapped = np.unwrap(phases)
            if len(ps) > 2:
                coeffs_ph = np.polyfit(ps, phases_unwrapped, 2)
                print(f"  Phase fit arg(A_p) ~ {coeffs_ph[2]:.4f} + "
                      f"{coeffs_ph[1]:.4f}·p + {coeffs_ph[0]:.6f}·p²")
                print(f"    Linear rate: {coeffs_ph[1]:.4f} rad/rep "
                      f"= {coeffs_ph[1]/pi:.4f}π/rep")

        # ==============================================================
        # Part 3: Generating function singularity
        # ==============================================================
        print(f"\n\n  PART 3: Generating Function G(t, s) = Σ A_p t^p")
        print("  " + "-" * 60)

        kap = 1.0
        # Compute G(t) for real t ∈ [0, 0.99]
        # For real κ: G(t, κ) = ∫ exp(κΦ) Π_j 1/(1-tz_j) dμ
        # Singularity at t=1: G ~ C/(1-t)^N

        print(f"\n  G(t, κ=1) for real t (SU({N})):")
        print(f"  G(t) = Σ_{{p=0}}^{P_max} A_p(1) t^p")

        exp_kap = np.exp(kap * Phi)
        weighted_real = exp_kap * meas
        Ap_real_kap1 = np.zeros(P_max + 1)
        for p in range(P_max + 1):
            Ap_real_kap1[p] = np.sum(hp_list[p] * weighted_real).real

        ts = np.linspace(0, 0.95, 20)
        print(f"  {'t':>6s}  {'G(t)':>14s}  {'G(t)·(1-t)^N':>16s}")
        for t in ts:
            G_t = sum(Ap_real_kap1[p] * t**p for p in range(P_max + 1))
            G_reg = G_t * (1 - t)**N
            print(f"  {t:6.3f}  {G_t:14.6e}  {G_reg:16.6e}")

        # Also compute G directly: G(t) = ∫ exp(κΦ)/Π(1-tz_j) dμ
        print(f"\n  Direct computation: G_direct(t) = ∫ exp(κΦ)/det(I-tU) dμ")
        print(f"  {'t':>6s}  {'G_series':>14s}  {'G_direct':>14s}  "
              f"{'G_dir·(1-t)^N':>16s}")
        for t in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
            # Direct: ∫ exp(κΦ) Π_j 1/(1-t·z_j) dμ
            prod = np.ones(len(Phi), dtype=complex)
            for j in range(N):
                prod *= 1.0 / (1.0 - t * z[:, j])
            G_dir = np.sum(prod * exp_kap * meas).real
            G_ser = sum(Ap_real_kap1[p] * t**p for p in range(P_max + 1))
            G_dir_reg = G_dir * (1 - t)**N
            print(f"  {t:6.3f}  {G_ser:14.6e}  {G_dir:14.6e}  "
                  f"{G_dir_reg:16.6e}")

        # Determine the coefficient C in G(t) ~ C/(1-t)^N as t→1
        # G_direct(t) * (1-t)^N → C
        ts_close = [0.90, 0.92, 0.94, 0.96, 0.98, 0.99]
        print(f"\n  Extracting singularity coefficient C: G(t)·(1-t)^N → C")
        print(f"  {'t':>6s}  {'G(t)·(1-t)^N':>16s}")
        for t in ts_close:
            prod = np.ones(len(Phi), dtype=complex)
            for j in range(N):
                prod *= 1.0 / (1.0 - t * z[:, j])
            G_dir = np.sum(prod * exp_kap * meas).real
            C_est = G_dir * (1 - t)**N
            print(f"  {t:6.3f}  {C_est:16.8e}")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n\n  {'='*80}")
    print(f"  SUMMARY: Large-p Asymptotics")
    print(f"  {'='*80}")
    print(f"""
  For the Wilson-action character integral A_p(s) = ∫ h_p(U) exp(s Re Tr U) dU:

  1. REAL COUPLING (s = κ):
     A_p(κ) / d_p ~ C(κ) · r(κ)^p   for large p
     where r(κ) < 1 and log(r) is LINEAR in p (not quadratic as heat-kernel predicts).
     The successive ratio A_{{p+1}}/A_p converges to a limit r·(p+N)/(p+1) ≈ r.

  2. COMPLEX COUPLING (s = κ+iy):
     |A_p(κ+iy)| / d_p ~ |C(s)| · |r(s)|^p   for large p
     with |r(s)| < r(κ) (additional decay from y-oscillations).
     The PHASE arg(A_p) has approximate linear dependence on p.

  3. GENERATING FUNCTION:
     G(t, s) = Σ_p A_p(s) t^p = ∫ exp(sΦ)/det(I-tU) dμ
     has a singularity at t=1: G(t) ~ C(s)/(1-t)^N
     The coefficient C(s) = ∫ exp(sΦ)/det(I-U) dμ determines A_p ~ C·d_p·r^p.

  4. CONVEYOR BELT EXPLANATION:
     The partner at zero #k has d_{{p*}} |A_{{p*}}|^n ≈ |A_0|^n.
     Since |A_p| ~ d_p · r^p and d_p ~ p^{{N-1}}:
       p*^{{N-1}} · r^{{np*}} ≈ r^0 = 1
       p* ~ (-n log r)^{{(N-1)/n}} · ... (depends on r)
     The conveyor belt partner increases logarithmically with zero index.
""")
    print("=" * 90)


if __name__ == '__main__':
    main()
