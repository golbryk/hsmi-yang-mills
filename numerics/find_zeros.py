"""
Independent Zero Finder for Z_{2P}^{SU(N)}(kappa + iy)
========================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Paper  : Pi v16 — "Fisher Zeros of the Two-Plaquette SU(N) Model for Odd N"

Description
-----------
Finds Fisher zeros of Z_{2P}^{SU(N)}(kappa + iy) by two independent methods
and compares them:

Method A (Formula, from Theorem 2Seq-SU(N)):
    Computes rho_N = -S_AB / (2 S_A), alpha_N = arccos(rho_N),
    and generates zeros y_k from the two-sequence formula.
    This ASSUMES the theorem.

Method B (Direct character expansion, independent):
    Evaluates Z_{2P}(kappa + iy) = Sigma_p d_p A_p(kappa+iy)^2 directly
    using the character expansion with enough terms, then finds sign changes
    of Re Z_{2P} by scanning y.  This does NOT assume the theorem.

For large k (y_k >> y_0), the two methods should agree closely.
Discrepancies at small k are expected (Rouché threshold k_0).

Usage
-----
    python find_zeros.py --N 5 --kappa 1.0 --n_zeros 10 --y_max 35
    python find_zeros.py --N 3 --kappa 1.0 --n_zeros 8 --y_max 25
"""

import argparse
import sys
import numpy as np
from math import comb, pi, acos, exp
from itertools import combinations_with_replacement as cr


# ---------------------------------------------------------------------------
# Method A: Formula zeros (uses theorem)
# ---------------------------------------------------------------------------

def _h_real(p, eigs_real):
    """Author: Grzegorz Olbryk"""
    result = 0.0
    for combo in cr(range(len(eigs_real)), p):
        t = 1.0
        for i in combo: t *= eigs_real[i]
        result += t
    return result


def formula_zeros(N, kappa, n_zeros, n_reps=20):
    """
    Compute Fisher zeros via Theorem 2Seq-SU(N) (Method A).
    Requires N odd.

    Author: Grzegorz Olbryk
    """
    assert N % 2 == 1, f"N must be odd for Theorem 2Seq; got N={N}"
    n_plus = (N + 1) // 2
    n_minus = (N - 1) // 2
    eig_A = [1.0] * n_plus + [-1.0] * n_minus

    S_A = S_AB = 0.0
    for p in range(n_reps):
        d = comb(p + N - 1, N - 1)
        chi_A = _h_real(p, eig_A)
        A_p = d * exp(-p * (p + N) / float(N) * kappa)
        S_A += d * chi_A ** 2 * A_p
        S_AB += d * ((-1) ** p) * chi_A ** 2 * A_p

    rho = -S_AB / (2.0 * S_A)
    rho = max(-1 + 1e-12, min(1 - 1e-12, rho))
    alpha = acos(rho)

    k_max = n_zeros + 5
    fam_plus  = [(alpha / 2.0 + pi * k, 'A+') for k in range(k_max)]
    fam_minus = [((2 * pi - alpha) / 2.0 + pi * k, 'A-') for k in range(k_max)]
    all_y = sorted(fam_plus + fam_minus, key=lambda x: x[0])
    all_y = [z for z in all_y if z[0] > 0][:n_zeros + 1]

    zeros = []
    for i, (y, fam) in enumerate(all_y):
        if i == 0:
            zeros.append({'k': i, 'y': y, 'method': 'formula', 'family': fam, 'gap': None})
        else:
            zeros.append({'k': i, 'y': y, 'method': 'formula', 'family': fam,
                          'gap': y - all_y[i-1][0]})

    return zeros[:n_zeros], S_A, S_AB, rho, alpha


# ---------------------------------------------------------------------------
# Method B: Direct sign-change finder (does NOT use theorem)
# ---------------------------------------------------------------------------

def _char_symmetric_p(p, eigs):
    """
    Character of rep (p,0,...,0) of SU(N) via Newton's identity.
    eigs: array (n_samples, N) complex.
    Returns array (n_samples,) complex.
    Author: Grzegorz Olbryk
    """
    if p == 0:
        return np.ones(eigs.shape[0], dtype=complex)
    # Power sums p_k = Sigma_i eigs_i^k
    psums = np.array([np.sum(eigs ** k, axis=1) for k in range(1, p + 1)])
    h = np.zeros((p + 1, eigs.shape[0]), dtype=complex)
    h[0] = 1.0
    for k in range(1, p + 1):
        h[k] = (1.0 / k) * sum(psums[j] * h[k - 1 - j] for j in range(k))
    return h[p]


def _Z2P_char_series(N, kappa, y, n_reps, n_MC=40000, seed=42):
    """
    Evaluate Z_{2P}(kappa+iy) directly via character expansion + Haar MC.
    Does NOT use the two-sequence formula.
    Author: Grzegorz Olbryk
    """
    rng = np.random.default_rng(seed)
    Z_gaus = (rng.standard_normal((n_MC, N, N)) +
              1j * rng.standard_normal((n_MC, N, N)))
    Q, R = np.linalg.qr(Z_gaus)
    D = np.diagonal(R, axis1=1, axis2=2)
    D = D / np.abs(D)
    Q = Q * D[:, np.newaxis, :]
    dets = np.linalg.det(Q)
    Q = Q / (dets ** (1.0 / N))[:, np.newaxis, np.newaxis]

    eigs = np.linalg.eigvals(Q)              # (n_MC, N)
    Phi  = np.sum(eigs.real, axis=1)         # (n_MC,)
    exp_s_Phi = np.exp(complex(kappa, y) * Phi)

    Z = 0j
    for p in range(n_reps):
        d = comb(p + N - 1, N - 1)
        chi = _char_symmetric_p(p, eigs)
        Z += d * np.mean(chi * exp_s_Phi) ** 2

    return Z


def direct_zeros(N, kappa, n_zeros, y_max, n_reps=6, n_MC=40000, n_scan=120):
    """
    Find Fisher zeros by direct sign-change search on Re Z_{2P}(kappa+iy).
    Does NOT use the two-sequence formula (Method B).

    Author: Grzegorz Olbryk
    """
    y_vals = np.linspace(0.4, y_max, n_scan)
    Z_vals = [_Z2P_char_series(N, kappa, y, n_reps, n_MC, seed=42 + i)
              for i, y in enumerate(y_vals)]
    ReZ = np.array([Z.real for Z in Z_vals])

    # Sign-change bracketing
    brackets = []
    for i in range(len(ReZ) - 1):
        if ReZ[i] * ReZ[i+1] < 0:
            brackets.append((y_vals[i], y_vals[i+1]))

    zeros_direct = []
    for lo, hi in brackets:
        if len(zeros_direct) >= n_zeros:
            break
        for _ in range(30):
            mid = (lo + hi) / 2
            Z_mid = _Z2P_char_series(N, kappa, mid, n_reps, n_MC)
            Z_lo  = _Z2P_char_series(N, kappa, lo,  n_reps, n_MC)
            if Z_mid.real * Z_lo.real < 0:
                hi = mid
            else:
                lo = mid
        zeros_direct.append((lo + hi) / 2)

    result = []
    for i, y in enumerate(zeros_direct):
        gap = y - zeros_direct[i-1] if i > 0 else None
        result.append({'k': i, 'y': y, 'method': 'direct', 'family': '?', 'gap': gap})
    return result


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def compare_and_print(N, kappa, n_zeros, y_max, n_reps_formula, n_reps_direct,
                      n_MC, n_scan):
    """Author: Grzegorz Olbryk"""
    print()
    print("=" * 75)
    print(f"  Fisher Zeros Comparison: Formula vs Direct  [SU({N}), kappa={kappa}]")
    print(f"  Author: Grzegorz Olbryk  |  Paper Pi v16")
    print("=" * 75)

    # Method A
    if N % 2 == 1:
        zeros_A, S_A, S_AB, rho, alpha = formula_zeros(N, kappa, n_zeros, n_reps_formula)
        print(f"  Method A (formula): rho={rho:.4f}, alpha={alpha:.4f} rad = {alpha/np.pi:.4f}*pi")
    else:
        zeros_A = []
        print(f"  Method A: N={N} is even — Theorem 2Seq not applicable.")

    # Method B
    print(f"  Method B (direct MC): scanning y in [0.4, {y_max}], n_MC={n_MC}, n_scan={n_scan}")
    print(f"  Computing...", flush=True)
    zeros_B = direct_zeros(N, kappa, n_zeros, y_max, n_reps_direct, n_MC, n_scan)

    print()
    print(f"  {'k':>3}  {'Method A (formula)':>20}  {'Method B (direct)':>20}  {'|diff|':>8}")
    print("  " + "-" * 57)

    n_compare = min(len(zeros_A), len(zeros_B))
    for i in range(n_compare):
        yA = zeros_A[i]['y']
        yB = zeros_B[i]['y']
        diff = abs(yA - yB)
        print(f"  {i:>3}  {yA:>20.5f}  {yB:>20.5f}  {diff:>8.5f}")

    if zeros_B:
        gaps_B = [z['gap'] for z in zeros_B if z['gap'] is not None]
        if gaps_B:
            mean_gap_B = sum(gaps_B) / len(gaps_B)
            print()
            print(f"  Method B mean gap = {mean_gap_B:.4f}  (pi/2 = {pi/2:.4f})")
            if zeros_A:
                gaps_A = [z['gap'] for z in zeros_A if z['gap'] is not None]
                mean_gap_A = sum(gaps_A) / len(gaps_A)
                print(f"  Method A mean gap = {mean_gap_A:.4f}  (exact by Theorem 2Seq)")

    print()
    print("  INTERPRETATION:")
    print("  Agreement for k >= k_0 confirms the asymptotic theorem.")
    print("  Discrepancies at small k are expected (below Rouché threshold y_0).")
    print("  Method B has MC variance; increase n_MC for better accuracy.")
    print("=" * 75)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare formula zeros (Theorem 2Seq) with direct MC zeros. "
            "Author: Grzegorz Olbryk. Paper: Pi v16."
        )
    )
    parser.add_argument('--N',              type=int,   default=5)
    parser.add_argument('--kappa',          type=float, default=1.0)
    parser.add_argument('--n_zeros',        type=int,   default=8)
    parser.add_argument('--y_max',          type=float, default=28.0)
    parser.add_argument('--n_reps_formula', type=int,   default=20)
    parser.add_argument('--n_reps_direct',  type=int,   default=5,
                        help='Reps for direct method (fewer = faster)')
    parser.add_argument('--n_MC',           type=int,   default=40000)
    parser.add_argument('--n_scan',         type=int,   default=120)
    args = parser.parse_args()

    compare_and_print(
        args.N, args.kappa, args.n_zeros, args.y_max,
        args.n_reps_formula, args.n_reps_direct,
        args.n_MC, args.n_scan
    )


if __name__ == '__main__':
    main()
