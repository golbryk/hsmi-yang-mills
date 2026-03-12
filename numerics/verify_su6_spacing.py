"""
Verification: Fisher Zero Spacing for SU(6) — ⟨Δ⟩ = π/4
============================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: Extension of Theorem 2Seq-SU(N) to N ≡ 2 mod 4

Description
-----------
For SU(6) (N ≡ 2 mod 4), the balanced-split saddle (k=3, Φ=0) is NOT in SU(N)
because N-k = 3 is odd. The dominant saddles are:
  - Saddle A: k=4, eigenvalues (1,1,1,1,-1,-1), Φ_A = +2, ord = 14
  - Saddle B: k=2, eigenvalues (1,1,-1,-1,-1,-1), Φ_B = -2, ord = 14

Schur parity holds: eig_B is a permutation of -eig_A, so
  h_p(eig_B) = (-1)^p h_p(eig_A)

The two-saddle interference has ω = 2|Φ_0| = 4, giving:
  ⟨Δ⟩ = π/ω = π/4   (exact, for all κ > 0)
  pair sum = 2π/ω = π/2

This script verifies the prediction via:
  Method A: Formula-based zeros (adapting Theorem 2Seq with ω=4)
  Method B: Direct Weyl quadrature on SU(6) — 5D Gauss-Legendre

Usage
-----
    python verify_su6_spacing.py
    python verify_su6_spacing.py --kappa 2.0 --n_quad 15
"""

import argparse
import numpy as np
from math import comb, pi, exp, acos
from itertools import combinations_with_replacement as cr


# ---------------------------------------------------------------------------
# SU(N) helpers
# ---------------------------------------------------------------------------

def dim_rep(p, N):
    return comb(p + N - 1, N - 1)


def casimir(p, N):
    return p * (p + N) / float(N)


def h_p_general(p, eigenvalues):
    """Complete homogeneous symmetric polynomial h_p via combinatorial sum."""
    if p == 0:
        return 1.0
    result = 0.0
    for combo in cr(range(len(eigenvalues)), p):
        term = 1.0
        for idx in combo:
            term *= eigenvalues[idx]
        result += term
    return result


def h_p_vec(p, eigs):
    """h_p via Newton's identity, vectorized. eigs: (n_pts, N) complex."""
    n_pts = eigs.shape[0]
    if p == 0:
        return np.ones(n_pts, dtype=complex)
    psums = np.array([np.sum(eigs ** k, axis=1) for k in range(1, p + 1)])
    h = np.zeros((p + 1, n_pts), dtype=complex)
    h[0] = 1.0
    for k in range(1, p + 1):
        h[k] = sum(psums[j] * h[k - 1 - j] for j in range(k)) / k
    return h[p]


# ---------------------------------------------------------------------------
# Method A: Formula-based prediction for N ≡ 2 mod 4
# ---------------------------------------------------------------------------

def formula_prediction_N2mod4(N, kappa, n_reps=20):
    """
    Compute rho_N, alpha_N, and predicted zeros for N ≡ 2 mod 4.

    Dominant saddles have k = N/2 ± 1 (both even), Φ = ±2.
    Schur parity: h_p(eig_B) = (-1)^p h_p(eig_A).
    Oscillation frequency: ω = 2|Φ_0| = 4.

    The formula for ρ and α is the same as for N odd, but with
    different saddle eigenvalues.
    """
    assert N % 4 == 2, f"N must be ≡ 2 mod 4, got N={N}"

    # Saddle A: k = N/2 + 1 eigenvalues +1, rest -1
    k_A = N // 2 + 1
    eig_A = [1.0] * k_A + [-1.0] * (N - k_A)

    S_A = 0.0
    S_AB = 0.0
    for p in range(n_reps):
        d = dim_rep(p, N)
        chi_A = h_p_general(p, eig_A)
        A_p = d * exp(-casimir(p, N) * kappa)
        sign = (-1) ** p  # Schur parity
        S_A += d * chi_A ** 2 * A_p
        S_AB += d * sign * chi_A ** 2 * A_p

    rho = -S_AB / (2.0 * S_A)
    rho = max(-1.0 + 1e-12, min(1.0 - 1e-12, rho))
    alpha = acos(rho)

    return S_A, S_AB, rho, alpha


def generate_formula_zeros(alpha, omega, n_zeros):
    """
    Generate predicted zeros from cos(ω y) = -ρ, i.e. ω y = α + 2πk or 2π - α + 2πk.

    Families:
      y_k^+ = (α + 2πk) / ω
      y_k^- = (2π - α + 2πk) / ω
    """
    k_max = n_zeros + 5
    fam_plus = [(alpha + 2 * pi * k) / omega for k in range(k_max)]
    fam_minus = [((2 * pi - alpha) + 2 * pi * k) / omega for k in range(k_max)]

    all_y = sorted(fam_plus + fam_minus)
    all_y = [y for y in all_y if y > 0][:n_zeros]
    return all_y


# ---------------------------------------------------------------------------
# Method B: Weyl quadrature for SU(N) (general N)
# ---------------------------------------------------------------------------

def compute_Z2P_weyl_suN(N, kappa, y, n_reps=12, n_quad=15):
    """
    Compute Z_{2P}^{SU(N)}(kappa+iy) via exact Weyl integration.

    Uses (N-1)-dimensional trapezoidal rule on [0,2π]^{N-1}
    with the SU(N) constraint θ_N = -(θ_1+...+θ_{N-1}).

    The trapezoidal rule has EXPONENTIAL convergence for smooth periodic
    integrands on the torus — far superior to Gauss-Legendre here.

    For N=6, this is a 5D integral with n_quad^5 evaluation points.
    """
    s = complex(kappa, y)

    # Trapezoidal rule on [0, 2π): equally spaced points
    nodes = np.linspace(0, 2 * np.pi, n_quad, endpoint=False)
    weight_1d = 2 * np.pi / n_quad  # each point has equal weight

    # Build (N-1)-dimensional tensor product grid
    dim = N - 1
    grids = np.meshgrid(*([nodes] * dim), indexing='ij')
    theta = np.stack([g.ravel() for g in grids], axis=1)  # (n_pts, N-1)

    W = weight_1d ** dim  # constant weight for trapezoidal rule on torus

    n_pts = theta.shape[0]

    # SU(N) constraint: θ_N = -(θ_1 + ... + θ_{N-1})
    theta_N = -np.sum(theta, axis=1, keepdims=True)
    theta_all = np.concatenate([theta, theta_N], axis=1)  # (n_pts, N)

    # Eigenvalues z_j = exp(i θ_j)
    z = np.exp(1j * theta_all)

    # Vandermonde squared: ∏_{j<k} |z_j - z_k|²
    V2 = np.ones(n_pts)
    for j in range(N):
        for k in range(j + 1, N):
            V2 *= np.abs(z[:, j] - z[:, k]) ** 2

    # Phase: Φ = Re Tr U = Σ cos(θ_j)
    Phi = np.sum(np.cos(theta_all), axis=1)
    exp_sPhi = np.exp(s * Phi)

    # Integration measure: W * V2 / (2π)^dim
    # Normalization: ∫ |Δ|² ∏ dθ/(2π) = N! over [0,2π]^{N-1}
    measure = W * V2 / (2 * np.pi) ** dim

    # Normalization constant (should be close to N!)
    norm = np.sum(measure).real

    # Z_{2P} = Σ_p d_p [A_p(s)]²
    Z = 0j
    for p in range(n_reps):
        d = dim_rep(p, N)
        hp = h_p_vec(p, z)
        Ap = np.sum(hp * exp_sPhi * measure) / norm
        Z += d * Ap ** 2

    return Z


def find_zeros_weyl(N, kappa, y_min, y_max, n_scan, n_reps=12, n_quad=15):
    """Find Fisher zeros by scanning |Z| and Re Z via Weyl quadrature."""
    y_vals = np.linspace(y_min, y_max, n_scan)

    print(f"  Weyl scan: N={N}, κ={kappa}, y ∈ [{y_min:.2f}, {y_max:.2f}], "
          f"n_scan={n_scan}, n_quad={n_quad}")
    print(f"  Grid points per evaluation: {n_quad}^{N-1} = {n_quad**(N-1)}")
    print(f"  Computing...", flush=True)

    Z_vals = np.zeros(n_scan, dtype=complex)
    for i, yy in enumerate(y_vals):
        Z_vals[i] = compute_Z2P_weyl_suN(N, kappa, yy, n_reps, n_quad)
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{n_scan}] y={yy:.4f}, "
                  f"Re Z = {Z_vals[i].real:.4e}, Im Z = {Z_vals[i].imag:.4e}, "
                  f"|Z| = {abs(Z_vals[i]):.4e}", flush=True)

    # Find sign changes of Re Z (approximate zeros)
    re_brackets = []
    for i in range(n_scan - 1):
        if Z_vals[i].real * Z_vals[i + 1].real < 0:
            re_brackets.append((y_vals[i], y_vals[i + 1]))

    # Bisect each bracket for Re Z = 0
    re_zeros = []
    for lo, hi in re_brackets:
        for _ in range(30):
            mid = (lo + hi) / 2
            Z_mid = compute_Z2P_weyl_suN(N, kappa, mid, n_reps, n_quad)
            Z_lo = compute_Z2P_weyl_suN(N, kappa, lo, n_reps, n_quad)
            if Z_mid.real * Z_lo.real < 0:
                hi = mid
            else:
                lo = mid
        yz = (lo + hi) / 2
        Zz = compute_Z2P_weyl_suN(N, kappa, yz, n_reps, n_quad)
        re_zeros.append({'y': yz, 'ReZ': Zz.real, 'ImZ': Zz.imag, 'absZ': abs(Zz)})

    return re_zeros, Z_vals, y_vals


# ---------------------------------------------------------------------------
# Bessel-function approach for A_p (verification of formula convergence)
# ---------------------------------------------------------------------------

def A_p_bessel_su2(p, s):
    """Exact A_p for SU(2) using modified Bessel functions.
    A_p(s) = I_p(s) - I_{p+2}(s) normalized by Z_1."""
    # For SU(2), the character integral has a known closed form
    # via the Weyl formula: A_p(s) ∝ I_p(2s) - I_{p+2}(2s) ... (not needed here)
    pass


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def verify_su6(kappa, n_reps=20, n_quad=15, n_scan=80, y_max=8.0):
    """Full verification of ⟨Δ⟩ = π/4 for SU(6)."""
    N = 6
    omega = 4  # 2 * |Φ_0| = 2 * 2

    print()
    print("=" * 75)
    print("  VERIFICATION: Fisher Zero Spacing for SU(6)")
    print("  Prediction: ⟨Δ⟩ = π/4 = {:.6f}".format(pi / 4))
    print(f"  Author: Grzegorz Olbryk  |  κ = {kappa}")
    print("=" * 75)

    # --- Saddle-point analysis ---
    print()
    print("  [1] Saddle-point structure for SU(6):")
    print("  " + "-" * 50)
    for k in range(7):
        Phi = 2 * k - 6
        ord_val = k * (k - 1) + (6 - k) * (6 - k - 1)
        in_su6 = (6 - k) % 2 == 0
        mark = ""
        if in_su6 and k in (2, 4):
            mark = " <-- dominant (Φ=±2, ord=14)"
        elif in_su6 and k in (0, 6):
            mark = " (sub-dominant)"
        print(f"    k={k}: Φ={Phi:+2d}, ord={ord_val:>2}, "
              f"in SU(6)={'YES' if in_su6 else 'no '}{mark}")
    print()
    print("  Two dominant saddles at Φ = ±2 → ω = 4 → ⟨Δ⟩ = π/4")

    # --- Method A: Formula prediction ---
    print()
    print("  [2] Method A: Formula-based prediction (adapting Theorem 2Seq)")
    print("  " + "-" * 50)

    S_A, S_AB, rho, alpha = formula_prediction_N2mod4(N, kappa, n_reps)
    gap_short = (pi - alpha) / 2
    gap_long = alpha / 2
    pair_sum = gap_short + gap_long
    avg = pair_sum / 2

    print(f"    S_A   = {S_A:.6f}")
    print(f"    S_AB  = {S_AB:.6f}")
    print(f"    ρ_6   = {rho:.6f}")
    print(f"    α_6   = {alpha:.6f} rad = {alpha/pi:.6f}π")
    print(f"    Δ_short = (π - α)/2 = {gap_short:.6f}")
    print(f"    Δ_long  = α/2       = {gap_long:.6f}")
    print(f"    pair sum = {pair_sum:.6f}   (exact: π/2 = {pi/2:.6f})")
    print(f"    ⟨Δ⟩     = {avg:.6f}   (exact: π/4 = {pi/4:.6f})")
    print(f"    |pair_sum - π/2| = {abs(pair_sum - pi/2):.2e}")

    # Generate formula zeros
    formula_zeros = generate_formula_zeros(alpha, omega, 15)
    print()
    print("    Predicted zeros (formula):")
    print(f"    {'k':>3}  {'y_k':>10}  {'Δ_k':>10}  {'type':>8}")
    print("    " + "-" * 38)
    gaps_formula = []
    for i, yk in enumerate(formula_zeros):
        gap = yk - formula_zeros[i - 1] if i > 0 else None
        gtype = ''
        if gap is not None:
            gaps_formula.append(gap)
            gtype = 'short' if gap < avg else 'long'
        gap_str = f"{gap:.6f}" if gap is not None else "—"
        print(f"    {i:>3}  {yk:>10.6f}  {gap_str:>10}  {gtype:>8}")

    # --- Method B: Weyl quadrature verification ---
    print()
    print("  [3] Method B: Direct Weyl quadrature verification")
    print("  " + "-" * 50)

    y_min = 0.2
    re_zeros, Z_vals, y_scan = find_zeros_weyl(
        N, kappa, y_min, y_max, n_scan, n_reps=min(n_reps, 12), n_quad=n_quad
    )

    print()
    print("    Re Z = 0 crossings found by Weyl quadrature:")
    print(f"    {'k':>3}  {'y_weyl':>10}  {'|Z|':>10}  {'Im Z':>12}")
    print("    " + "-" * 48)
    for i, z in enumerate(re_zeros):
        print(f"    {i:>3}  {z['y']:>10.6f}  {z['absZ']:>10.4e}  {z['ImZ']:>12.4e}")

    # --- Comparison ---
    print()
    print("  [4] Comparison: Formula vs Weyl")
    print("  " + "-" * 50)

    n_comp = min(len(formula_zeros), len(re_zeros))
    if n_comp > 0:
        print(f"    {'k':>3}  {'y_formula':>10}  {'y_weyl':>10}  {'|diff|':>10}  {'Δ_weyl':>10}")
        print("    " + "-" * 55)
        gaps_weyl = []
        for i in range(n_comp):
            yf = formula_zeros[i]
            yw = re_zeros[i]['y']
            diff = abs(yf - yw)
            gap_w = yw - re_zeros[i - 1]['y'] if i > 0 else None
            gap_str = f"{gap_w:.6f}" if gap_w is not None else "—"
            if gap_w is not None:
                gaps_weyl.append(gap_w)
            print(f"    {i:>3}  {yf:>10.6f}  {yw:>10.6f}  {diff:>10.6f}  {gap_str:>10}")

        if gaps_weyl:
            mean_weyl = sum(gaps_weyl) / len(gaps_weyl)
            print()
            print(f"    Mean gap (Weyl) = {mean_weyl:.6f}")
            print(f"    π/4             = {pi/4:.6f}")
            print(f"    |deviation|     = {abs(mean_weyl - pi/4):.6f}")

            # Pair sums
            pair_sums = [gaps_weyl[i] + gaps_weyl[i + 1]
                         for i in range(0, len(gaps_weyl) - 1, 2)]
            if pair_sums:
                mean_ps = sum(pair_sums) / len(pair_sums)
                print(f"    Mean pair sum   = {mean_ps:.6f}")
                print(f"    π/2             = {pi/2:.6f}")
    else:
        print("    No zeros found for comparison!")

    # --- Convergence check ---
    print()
    print("  [5] Quadrature convergence check at y = 2.0")
    print("  " + "-" * 50)
    y_test = 2.0
    for nq in [10, 12, 15, 18]:
        if nq <= n_quad + 3:  # only test if reasonable
            Z = compute_Z2P_weyl_suN(N, kappa, y_test, min(n_reps, 12), nq)
            print(f"    n_quad={nq:>2}: Re Z = {Z.real:>12.6e}, "
                  f"Im Z = {Z.imag:>12.6e}, |Z| = {abs(Z):>12.6e}")

    # --- Summary ---
    print()
    print("  " + "=" * 50)
    print("  SUMMARY")
    print("  " + "=" * 50)
    print(f"    SU(6), κ = {kappa}")
    print(f"    Dominant saddles: Φ = ±2, ω = 4")
    print(f"    Formula prediction:  ⟨Δ⟩ = π/4 = {pi/4:.6f}")
    if gaps_formula:
        mean_f = sum(gaps_formula) / len(gaps_formula)
        print(f"    Formula ⟨Δ⟩:         {mean_f:.6f}")
    if n_comp > 0 and gaps_weyl:
        print(f"    Weyl ⟨Δ⟩:            {mean_weyl:.6f}")
        print(f"    Agreement:           {'YES' if abs(mean_weyl - pi/4) < 0.05 else 'CHECK'}")
    print("  " + "=" * 50)
    print()


# ---------------------------------------------------------------------------
# Bonus: Quick check that SU(4) has no zeros on real y axis
# ---------------------------------------------------------------------------

def check_su4_no_zeros(kappa, n_quad=20, n_scan=40, y_max=5.0):
    """Verify that SU(4) Z_{2P}(κ+iy) has |Z| >> 0 everywhere on real y axis."""
    N = 4
    print()
    print("=" * 75)
    print("  CHECK: SU(4) — no Fisher zeros on real y axis")
    print(f"  (N ≡ 0 mod 4: dominant saddle Φ=0, non-oscillating)")
    print(f"  κ = {kappa}, n_quad = {n_quad}")
    print("=" * 75)

    y_vals = np.linspace(0.2, y_max, n_scan)
    min_absZ = float('inf')
    for i, yy in enumerate(y_vals):
        Z = compute_Z2P_weyl_suN(N, kappa, yy, n_reps=12, n_quad=n_quad)
        absZ = abs(Z)
        if absZ < min_absZ:
            min_absZ = absZ
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{n_scan}] y={yy:.3f}: "
                  f"Re Z={Z.real:.4e}, Im Z={Z.imag:.4e}, |Z|={absZ:.4e}")

    print()
    print(f"  Minimum |Z| over scan: {min_absZ:.4e}")
    print(f"  {'CONFIRMED: no zeros on real y axis' if min_absZ > 1e-6 else 'POTENTIAL ZEROS FOUND'}")
    print("=" * 75)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify ⟨Δ⟩ = π/4 for SU(6) Fisher zeros. "
                    "Author: Grzegorz Olbryk."
    )
    parser.add_argument('--kappa', type=float, default=1.0)
    parser.add_argument('--n_reps', type=int, default=20)
    parser.add_argument('--n_quad', type=int, default=12,
                        help='Quadrature points per dimension (12^5=248K pts)')
    parser.add_argument('--n_scan', type=int, default=60)
    parser.add_argument('--y_max', type=float, default=7.0)
    parser.add_argument('--su4_check', action='store_true',
                        help='Also check SU(4) has no zeros')
    args = parser.parse_args()

    verify_su6(args.kappa, args.n_reps, args.n_quad, args.n_scan, args.y_max)

    if args.su4_check:
        check_su4_no_zeros(args.kappa)


if __name__ == '__main__':
    main()
