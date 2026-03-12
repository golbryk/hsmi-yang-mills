"""
p*(y) Scaling Theory from Generating Function
===============================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 42 — Analytical scaling law

Derives the conveyor belt scaling p*(y) analytically from the generating
function radius r(s):

    A_p(s) / d_p ~ r(s)^p   for large p

The "crossing" p* where |A_{p*}(s)| first dominates over |A_0(s)| satisfies:
    p* ~ log(|A_0(s)|^n) / (-n log|r(s)|) + corrections

More precisely, the peak of |A_p(s)|/|A_p(kappa)| occurs where
d/dp [p log|r(s)/r(kappa)| + (N-1) log p] = 0 (saddle-point in p).

Verifies against numerical p* from conveyor_belt_scaling.py.
"""

import numpy as np
from math import comb, pi
import time
import sys


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
    Phi = np.sum(np.cos(theta_all), axis=1)
    measure = w * V2 / (2 * np.pi) ** dim
    norm = np.sum(measure).real
    measure = measure / norm
    return z, Phi, measure


def main():
    t0 = time.time()

    N = 4
    n_quad = 40
    p_max = 25  # need enough reps for asymptotic regime but not too many (aliasing)
    n_plaq = 2

    print()
    print("=" * 80)
    print("  p*(y) Scaling Theory from Generating Function — Task 42")
    print(f"  SU({N}), n_quad={n_quad}, p_max={p_max}")
    print("=" * 80)

    # Build grid
    print(f"\n  Building Weyl grid: {n_quad}^{N-1} = {n_quad**(N-1)} points...")
    sys.stdout.flush()
    z, Phi, measure = build_weyl_grid(N, n_quad)

    print(f"  Precomputing h_p for p=0..{p_max}...")
    hp_list = [h_p_vec(p, z) for p in range(p_max + 1)]
    dims = [dim_rep(p, N) for p in range(p_max + 1)]
    exp_kPhi_cache = {}

    # ======================================================================
    # Part 1: Compute r(kappa+iy) — asymptotic ratio
    # ======================================================================
    print(f"\n  PART 1: Generating Function Radius r(s)")
    print("  " + "-" * 60)

    kappa = 1.0
    y_values = np.arange(0.0, 14.0, 0.5)

    print(f"\n  r(s) = lim_{{p->inf}} (A_{{p+1}}/A_p) * (d_p/d_{{p+1}})")
    print(f"  Computed from p={p_max-5}..{p_max} ratio average\n")

    r_values = []
    print(f"  {'y':>5}  {'|r(s)|':>10}  {'|r(kap)|':>10}  "
          f"{'|r(s)/r(kap)|':>14}  {'arg(r(s))':>10}  {'phase/pi':>10}")

    for y in y_values:
        s = kappa + 1j * y
        exp_sPhi = np.exp(kappa * Phi) * np.exp(1j * y * Phi)
        weighted = exp_sPhi * measure

        # Compute A_p for p = p_max-10 .. p_max
        Ap_vals = {}
        for p in range(max(0, p_max - 10), p_max + 1):
            Ap_vals[p] = np.sum(hp_list[p] * weighted)

        # Ratio r_p = (A_{p+1}/A_p) * (d_p/d_{p+1})
        r_estimates = []
        for p in range(max(1, p_max - 8), p_max):
            if abs(Ap_vals[p]) > 1e-30:
                r_p = (Ap_vals[p + 1] / Ap_vals[p]) * (dims[p] / dims[p + 1])
                r_estimates.append(r_p)

        if r_estimates:
            r_avg = np.mean(r_estimates[-3:])  # use last 3 for stability
            r_values.append((y, r_avg))

            # Also compute r at real kappa
            if y == 0:
                r_kap = abs(r_avg)
            phase_str = f"{np.angle(r_avg)/pi:.4f}"
            print(f"  {y:5.1f}  {abs(r_avg):10.6f}  {r_kap:10.6f}  "
                  f"{abs(r_avg)/r_kap:14.6f}  {np.angle(r_avg):10.4f}  "
                  f"{phase_str:>10}")

    # ======================================================================
    # Part 2: Predicted p* from r(s)
    # ======================================================================
    print(f"\n\n  PART 2: Predicted p* from Ratio Table Peak")
    print("  " + "-" * 60)

    # The ratio |A_p(s)|/|A_p(kappa)| ~ |r(s)/r(kappa)|^p * correction
    # Peak in p occurs where d/dp [p log|r(s)/r(kap)| + (N-1) log p] = 0
    # => p* = -(N-1) / log|r(s)/r(kap)|

    print(f"\n  Analytical prediction: p_peak = (N-1) / (-log|r(s)/r(kappa)|)")
    print(f"  where N-1 = {N-1} accounts for d_p ~ p^{{N-1}} growth\n")

    print(f"  {'y':>5}  {'|r(s)/r(kap)|':>14}  {'log ratio':>10}  "
          f"{'p_peak(theory)':>14}  {'p_peak(num)':>11}")

    # Need numerical p_peak from actual computation
    exp_kPhi_real = np.exp(kappa * Phi)
    w_real = exp_kPhi_real * measure
    Ap_real_dict = {}
    for p in range(p_max + 1):
        Ap_real_dict[p] = abs(np.sum(hp_list[p] * w_real))

    for y_val, r_s in r_values:
        if y_val < 0.5:
            continue

        ratio_r = abs(r_s) / r_kap
        log_ratio = np.log(ratio_r)

        if log_ratio > 0:
            p_theory = (N - 1) / log_ratio
        else:
            p_theory = float('inf')

        # Numerical p_peak: find p that maximizes |A_p(s)|/|A_p(kappa)|
        exp_sPhi = np.exp(kappa * Phi) * np.exp(1j * y_val * Phi)
        weighted = exp_sPhi * measure
        max_ratio = 0
        p_num = 0
        for p in range(1, p_max + 1):
            Ap_s = abs(np.sum(hp_list[p] * weighted))
            if Ap_real_dict[p] > 1e-30:
                ratio = Ap_s / Ap_real_dict[p]
                if ratio > max_ratio:
                    max_ratio = ratio
                    p_num = p

        if p_theory < 100:
            print(f"  {y_val:5.1f}  {ratio_r:14.6f}  {log_ratio:10.4f}  "
                  f"{p_theory:14.1f}  {p_num:11d}")
        else:
            print(f"  {y_val:5.1f}  {ratio_r:14.6f}  {log_ratio:10.4f}  "
                  f"{'inf':>14}  {p_num:11d}")

    # ======================================================================
    # Part 3: dp*/dy slope from r(s)
    # ======================================================================
    print(f"\n\n  PART 3: dp*/dy Slope from r(s)")
    print("  " + "-" * 60)

    # If |r(s)| ~ |r(kappa)| + c * y for small y, then
    # p* ~ (N-1) / (c * y / |r(kappa)|) = (N-1) |r(kappa)| / (c * y)
    # But actually for larger y, |r(s)| saturates and the scaling becomes:
    # p* = (N-1) / log(|r(s)|/|r(kappa)|) which is not linear in y.

    # Let's compute the numerical slope dp*/dy from the data
    y_pstar_pairs = []
    for y_val in np.arange(1.0, 12.0, 0.25):
        exp_sPhi = np.exp(kappa * Phi) * np.exp(1j * y_val * Phi)
        weighted = exp_sPhi * measure

        max_ratio = 0
        p_num = 0
        for p in range(1, p_max + 1):
            Ap_s = abs(np.sum(hp_list[p] * weighted))
            if Ap_real_dict[p] > 1e-30:
                ratio = Ap_s / Ap_real_dict[p]
                if ratio > max_ratio:
                    max_ratio = ratio
                    p_num = p

        if p_num < p_max - 3:  # avoid ceiling
            y_pstar_pairs.append((y_val, p_num))

    if len(y_pstar_pairs) > 5:
        yr = np.array([p[0] for p in y_pstar_pairs])
        pr = np.array([p[1] for p in y_pstar_pairs], dtype=float)

        # Linear fit
        c = np.polyfit(yr, pr, 1)
        rmse = np.sqrt(np.mean((pr - np.polyval(c, yr))**2))

        print(f"\n  Numerical (from peak of |A_p(s)|/|A_p(kap)|):")
        print(f"  Data points: {len(y_pstar_pairs)}")
        print(f"  Linear fit: p_peak = {c[0]:.3f} y + ({c[1]:.1f})")
        print(f"  RMSE = {rmse:.2f}")
        print(f"  Slope dp*/dy = {c[0]:.3f}")

        # Compare with r-based prediction
        # For moderate y (2-8), |r(s)/r(kap)| is approximately linear in y
        # Fit |r(s)/r(kap)| vs y
        r_ratios = [(y, abs(r_s) / r_kap) for y, r_s in r_values if 2 <= y <= 10]
        if len(r_ratios) > 3:
            y_r = np.array([r[0] for r in r_ratios])
            rr = np.array([r[1] for r in r_ratios])
            c_r = np.polyfit(y_r, np.log(rr), 1)
            print(f"\n  Generating function prediction:")
            print(f"  log|r(s)/r(kap)| ~ {c_r[0]:.4f} y + ({c_r[1]:.2f})")
            if c_r[0] > 0:
                predicted_slope = (N - 1) / c_r[0]
                print(f"  => dp*/dy ~ (N-1) / slope = {N-1} / {c_r[0]:.4f} "
                      f"= {predicted_slope:.2f}")
                print(f"  Numerical dp*/dy = {c[0]:.2f}")
                print(f"  Agreement: {abs(predicted_slope - c[0]) / c[0] * 100:.1f}% difference")

    # Print the data
    print(f"\n  Raw data:")
    print(f"  {'y':>6}  {'p_peak':>7}")
    for y_val, p_num in y_pstar_pairs:
        if int(y_val * 4) % 4 == 0:  # print every 1.0
            print(f"  {y_val:6.2f}  {p_num:7d}")

    # ======================================================================
    # Part 4: Phase structure of r(s)
    # ======================================================================
    print(f"\n\n  PART 4: Phase Structure of r(s)")
    print("  " + "-" * 60)

    print(f"\n  arg(r(kappa+iy)) determines which p oscillates fastest:")
    print(f"  A_p(s) ~ d_p r(s)^p => phase = p * arg(r(s))")
    print(f"  Anti-phase with A_0 when p * arg(r) ~ pi (mod 2pi)\n")

    for y_val, r_s in r_values:
        if y_val < 0.5:
            continue
        arg_r = np.angle(r_s)
        if abs(arg_r) > 0.01:
            p_antiphase = pi / abs(arg_r)
            print(f"  y={y_val:5.1f}: arg(r) = {arg_r:+.4f} ({arg_r/pi:+.3f} pi), "
                  f"p_antiphase = pi/|arg(r)| = {p_antiphase:.1f}")

    elapsed = time.time() - t0
    print(f"\n\n  [Completed in {elapsed:.1f}s]")

    # Summary
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print("  1. r(s) = asymptotic ratio of consecutive A_p/d_p")
    print("  2. |r(kappa+iy)| > |r(kappa)| for y > 0 (conveyor belt driver)")
    print(f"  3. p_peak = (N-1) / log|r(s)/r(kappa)| predicts ratio table peaks")
    print(f"  4. dp*/dy determined by d(log|r|)/dy")
    print(f"  5. Phase arg(r(s)) determines which p is anti-phase with A_0")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
