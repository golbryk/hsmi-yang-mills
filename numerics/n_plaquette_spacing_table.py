"""
n-Plaquette Fisher Zero Spacing Table
======================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 5 — n-Plaquette conjecture

Theorem (n-plaquette, N odd)
----------------------------
For N odd >= 3, n >= 2, kappa > 0:

    <Delta> = pi/n    (exact, for all kappa)

where <Delta> is the average spacing of Fisher zeros of Z_{nP}^{SU(N)}.

This is a corollary of Theorem 2Seq-SU(N):
- The saddle-point premises (Schur parity, equal orders) are properties
  of the single-link integral A_p(s), independent of n.
- A_p^n has oscillation frequency n*Phi_0 (from n-th power of e^{iy*Phi_0}).
- Therefore the pair sum becomes 2*pi/(2n*|Phi_0|) for each pair,
  giving average spacing pi/(n*|Phi_0|) = pi/n for N odd (|Phi_0|=1).

The parameter rho_N (from the character expansion) determines the
individual gap PATTERN (short/long alternation), while the AVERAGE
is always pi/n regardless of rho.

Predicted zero positions for the n-plaquette model:
    y_k^+ = (alpha + 2*pi*k) / n
    y_k^- = (2*pi - alpha + 2*pi*k) / n

where alpha = arccos(rho_N) and n is the number of plaquettes.

Usage
-----
    python n_plaquette_spacing_table.py
    python n_plaquette_spacing_table.py --N_list 3 5 7 --n_plaq_list 2 3 4 5
"""

import argparse
from math import comb, pi, exp, acos
from itertools import combinations_with_replacement as cr


# ---------------------------------------------------------------------------
# SU(N) helpers (shared with spacing_table.py)
# ---------------------------------------------------------------------------

def h_p_general(p, eigenvalues):
    """Complete homogeneous symmetric polynomial h_p."""
    if p == 0:
        return 1.0
    result = 0.0
    for combo in cr(range(len(eigenvalues)), p):
        term = 1.0
        for idx in combo:
            term *= eigenvalues[idx]
        result += term
    return result


def casimir_suN(p, N):
    return p * (p + N) / float(N)


def dimension_suN(p, N):
    return comb(p + N - 1, N - 1)


def balanced_saddles(N):
    """Return saddle eigenvalues for N odd."""
    assert N % 2 == 1
    n_plus = (N + 1) // 2
    return [1.0] * n_plus + [-1.0] * (N - n_plus)


# ---------------------------------------------------------------------------
# Core computation: rho_N and alpha_N
# ---------------------------------------------------------------------------

def compute_rho_alpha(N, kappa, n_reps=20):
    """Compute rho_N(kappa) and alpha_N(kappa) for SU(N) odd.

    These parameters determine the gap PATTERN (short/long alternation)
    but NOT the average spacing, which is always pi/n for n plaquettes.
    """
    eig_A = balanced_saddles(N)
    S_A = 0.0
    S_AB = 0.0

    for p in range(n_reps):
        d = dimension_suN(p, N)
        chi_A = h_p_general(p, eig_A)
        A_p = d * exp(-casimir_suN(p, N) * kappa)
        S_A += d * chi_A ** 2 * A_p
        S_AB += d * ((-1) ** p) * chi_A ** 2 * A_p

    if S_A < 1e-30:
        return None, None
    rho = max(-1 + 1e-12, min(1 - 1e-12, -S_AB / (2.0 * S_A)))
    alpha = acos(rho)
    return rho, alpha


# ---------------------------------------------------------------------------
# n-plaquette gap computation
# ---------------------------------------------------------------------------

def compute_gaps_nplaq(alpha, n_plaq):
    """Compute the n individual gaps for n-plaquette model.

    Zero positions (in one period [0, pi)):
        y_k^+ = (alpha + 2*pi*k) / n  for k such that y in [0, pi)
        y_k^- = (2*pi - alpha + 2*pi*k) / n  for k such that y in [0, pi)

    Returns sorted list of gaps.
    """
    n = n_plaq
    zeros = []

    # Generate zeros in [0, pi)
    for k in range(n + 1):
        y_plus = (alpha + 2 * pi * k) / n
        y_minus = (2 * pi - alpha + 2 * pi * k) / n
        if 0 <= y_plus < pi + 1e-10:
            zeros.append(y_plus)
        if 0 <= y_minus < pi + 1e-10:
            zeros.append(y_minus)

    zeros = sorted(set(round(z, 12) for z in zeros if z < pi + 1e-10))

    if len(zeros) < 2:
        return [], zeros

    # Compute gaps (including wrap-around)
    gaps = []
    for i in range(len(zeros) - 1):
        gaps.append(zeros[i + 1] - zeros[i])
    # Wrap-around gap: from last zero to first zero + pi
    if len(zeros) >= 2:
        gaps.append((zeros[0] + pi) - zeros[-1])

    return gaps, zeros


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def build_table(N_list, kappa_list, n_plaq_list, n_reps=20):
    """Build the n-plaquette spacing table."""
    rows = []
    for N in N_list:
        if N % 2 == 0:
            continue
        for kappa in kappa_list:
            rho, alpha = compute_rho_alpha(N, kappa, n_reps)
            if rho is None:
                continue
            for n_plaq in n_plaq_list:
                gaps, zeros = compute_gaps_nplaq(alpha, n_plaq)
                if not gaps:
                    continue
                avg = sum(gaps) / len(gaps)
                nfold_sum = sum(gaps)
                rows.append({
                    'N': N, 'kappa': kappa, 'n': n_plaq,
                    'rho': rho, 'alpha': alpha,
                    'n_zeros': len(zeros), 'n_gaps': len(gaps),
                    'gaps': gaps, 'avg': avg,
                    'nfold_sum': nfold_sum,
                    'expected_avg': pi / n_plaq,
                    'expected_sum': pi,
                })
    return rows


def print_table(rows):
    """Print the n-plaquette spacing table."""
    print()
    print("=" * 90)
    print("  n-Plaquette Fisher Zero Spacing: <Delta> = pi/n")
    print("  Author: Grzegorz Olbryk  |  March 2026  |  Theorem (n-plaquette, N odd)")
    print("=" * 90)

    header = (
        f"  {'N':>3}  {'k':>4}  {'n':>3}  {'rho':>8}  {'alpha':>8}  "
        f"{'<Delta>':>10}  {'pi/n':>10}  {'n-sum':>8}  {'pi':>8}  {'ok'}"
    )
    print(header)
    print("  " + "-" * 86)

    prev_N = None
    prev_kappa = None
    for r in rows:
        if r['N'] != prev_N and prev_N is not None:
            print()
        if r['N'] != prev_N or r['kappa'] != prev_kappa:
            prev_N = r['N']
            prev_kappa = r['kappa']

        ok_avg = abs(r['avg'] - r['expected_avg']) < 1e-10
        ok_sum = abs(r['nfold_sum'] - r['expected_sum']) < 1e-10
        ok = ('Y' if ok_avg and ok_sum else 'N')
        print(
            f"  {r['N']:>3}  {r['kappa']:>4.1f}  {r['n']:>3}  "
            f"{r['rho']:>8.5f}  {r['alpha']:>8.5f}  "
            f"{r['avg']:>10.6f}  {r['expected_avg']:>10.6f}  "
            f"{r['nfold_sum']:>8.5f}  {r['expected_sum']:>8.5f}  {ok}"
        )

    print()
    all_ok = all(abs(r['avg'] - r['expected_avg']) < 1e-10 and
                 abs(r['nfold_sum'] - r['expected_sum']) < 1e-10
                 for r in rows)
    print(f"  All {len(rows)} cases: <Delta> = pi/n ? "
          f"{'YES' if all_ok else 'NO'}")
    print()

    # Detailed gap patterns
    print("  Gap patterns (first occurrence of each (N, n) pair):")
    print("  " + "-" * 86)
    seen = set()
    for r in rows:
        key = (r['N'], r['n'])
        if key in seen:
            continue
        seen.add(key)
        gap_str = ", ".join(f"{g:.5f}" for g in r['gaps'])
        print(f"  SU({r['N']}), n={r['n']}, kappa={r['kappa']}: "
              f"gaps = [{gap_str}]")
        print(f"    {'':>15} sum = {r['nfold_sum']:.10f}, "
              f"avg = {r['avg']:.10f}")

    print("=" * 90)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="n-Plaquette spacing table. Author: Grzegorz Olbryk."
    )
    parser.add_argument('--N_list', type=int, nargs='+', default=[3, 5, 7])
    parser.add_argument('--kappa_list', type=float, nargs='+',
                        default=[0.5, 1.0, 2.0])
    parser.add_argument('--n_plaq_list', type=int, nargs='+',
                        default=[2, 3, 4, 5])
    parser.add_argument('--n_reps', type=int, default=20)
    args = parser.parse_args()

    for N in args.N_list:
        if N % 2 == 0:
            print(f"  Warning: N={N} even — skipped.")

    rows = build_table(args.N_list, args.kappa_list, args.n_plaq_list,
                       args.n_reps)
    print_table(rows)


if __name__ == '__main__':
    main()
