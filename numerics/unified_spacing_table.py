"""
Unified Fisher Zero Spacing Table
===================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 6 — Unified Fisher zero theory

Unified theorem: For N with Phi_0 != 0 and n >= 2 plaquettes:

    <Delta> = pi / (n * |Phi_0|)

where |Phi_0| = 1 (N odd), |Phi_0| = 2 (N equiv 2 mod 4).

This script verifies the formula for both N odd and N equiv 2 mod 4,
for n = 2, 3, 4, 5, and multiple kappa values.
"""

import argparse
from math import comb, pi, exp, acos
from itertools import combinations_with_replacement as cr


# ---------------------------------------------------------------------------
# SU(N) helpers
# ---------------------------------------------------------------------------

def h_p_general(p, eigenvalues):
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


def dominant_saddle_eigenvalues(N):
    """Return eigenvalues of the dominant balanced-split saddle.

    N odd: k = (N+1)/2 ones, (N-1)/2 minus-ones. Phi = +1.
    N equiv 2 mod 4: k = N/2 + 1 ones, N/2 - 1 minus-ones. Phi = +2.
    N equiv 0 mod 4: k = N/2 ones, N/2 minus-ones. Phi = 0. (degenerate)
    """
    if N % 2 == 1:
        n_plus = (N + 1) // 2
        return [1.0] * n_plus + [-1.0] * (N - n_plus)
    elif N % 4 == 2:
        n_plus = N // 2 + 1
        return [1.0] * n_plus + [-1.0] * (N - n_plus)
    else:
        n_plus = N // 2
        return [1.0] * n_plus + [-1.0] * (N - n_plus)


def phi_0(N):
    """Dominant saddle phase |Phi_0|."""
    if N % 2 == 1:
        return 1
    elif N % 4 == 2:
        return 2
    else:
        return 0


# ---------------------------------------------------------------------------
# Core computation: rho_N and alpha_N
# ---------------------------------------------------------------------------

def compute_rho_alpha(N, kappa, n_reps=20):
    """Compute rho_N(kappa) and alpha_N(kappa)."""
    eig_A = dominant_saddle_eigenvalues(N)
    phi = phi_0(N)

    if phi == 0:
        return None, None  # degenerate case

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
# Gap computation for general n and |Phi_0|
# ---------------------------------------------------------------------------

def compute_gaps(alpha, n_plaq, phi0):
    """Compute gaps for n plaquettes with phase |Phi_0|.

    Zero positions:
        y_k^+ = (alpha + 2*pi*k) / (n * phi0)
        y_k^- = (2*pi - alpha + 2*pi*k) / (n * phi0)

    Period: pi (always — zeros repeat with period pi for all N)
    Zeros per period: n * phi0
    Average spacing: pi / (n * phi0)
    """
    n = n_plaq
    nf = n * phi0  # number of zeros per period [0, pi)
    period = pi

    zeros = []
    for k in range(nf + 1):
        y_plus = (alpha + 2 * pi * k) / nf
        y_minus = (2 * pi - alpha + 2 * pi * k) / nf
        if -1e-10 <= y_plus < period + 1e-10:
            zeros.append(y_plus)
        if -1e-10 <= y_minus < period + 1e-10:
            zeros.append(y_minus)

    zeros = sorted(set(round(z, 12) for z in zeros
                       if -1e-12 <= z < period + 1e-10))

    if len(zeros) < 2:
        return [], zeros

    gaps = []
    for i in range(len(zeros) - 1):
        gaps.append(zeros[i + 1] - zeros[i])
    # Wrap-around gap
    if len(zeros) >= 2:
        gaps.append((zeros[0] + period) - zeros[-1])

    return gaps, zeros


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def build_table(N_list, kappa_list, n_plaq_list, n_reps=20):
    rows = []
    for N in N_list:
        phi = phi_0(N)
        if phi == 0:
            continue  # skip N equiv 0 mod 4

        for kappa in kappa_list:
            rho, alpha = compute_rho_alpha(N, kappa, n_reps)
            if rho is None:
                continue

            for n_plaq in n_plaq_list:
                gaps, zeros = compute_gaps(alpha, n_plaq, phi)
                if not gaps:
                    continue
                avg = sum(gaps) / len(gaps)
                total = sum(gaps)
                expected_avg = pi / (n_plaq * phi)
                expected_sum = pi  # sum of all gaps in [0, pi) is always pi

                rows.append({
                    'N': N, 'kappa': kappa, 'n': n_plaq,
                    'phi0': phi, 'rho': rho, 'alpha': alpha,
                    'n_zeros': len(zeros), 'n_gaps': len(gaps),
                    'gaps': gaps, 'avg': avg,
                    'total': total,
                    'expected_avg': expected_avg,
                    'expected_sum': expected_sum,
                    'N_class': 'odd' if N % 2 == 1 else f'{N}≡2(4)',
                })
    return rows


def print_table(rows):
    print()
    print("=" * 100)
    print("  Unified Fisher Zero Spacing: <Delta> = pi / (n * |Phi_0|)")
    print("  Author: Grzegorz Olbryk  |  March 2026  |  Task 6: Unified Theorem")
    print("=" * 100)

    header = (
        f"  {'N':>3}  {'class':>8}  {'Φ₀':>3}  {'κ':>4}  {'n':>3}  "
        f"{'<Δ>':>10}  {'π/(nΦ₀)':>10}  {'Σ':>8}  {'π/Φ₀':>8}  {'ok'}"
    )
    print(header)
    print("  " + "-" * 96)

    prev_N = None
    for r in rows:
        if r['N'] != prev_N and prev_N is not None:
            print()
        prev_N = r['N']

        ok_avg = abs(r['avg'] - r['expected_avg']) < 1e-10
        ok_sum = abs(r['total'] - r['expected_sum']) < 1e-10
        ok = 'Y' if ok_avg and ok_sum else 'N'
        print(
            f"  {r['N']:>3}  {r['N_class']:>8}  {r['phi0']:>3}  "
            f"{r['kappa']:>4.1f}  {r['n']:>3}  "
            f"{r['avg']:>10.6f}  {r['expected_avg']:>10.6f}  "
            f"{r['total']:>8.5f}  {r['expected_sum']:>8.5f}  {ok}"
        )

    print()
    all_ok = all(abs(r['avg'] - r['expected_avg']) < 1e-10 and
                 abs(r['total'] - r['expected_sum']) < 1e-10
                 for r in rows)
    n_odd = sum(1 for r in rows if r['N'] % 2 == 1)
    n_even = sum(1 for r in rows if r['N'] % 2 == 0)
    print(f"  Total: {len(rows)} cases ({n_odd} N-odd, {n_even} N≡2(4))")
    print(f"  All <Δ> = π/(n·|Φ₀|) ? {'YES' if all_ok else 'NO'}")
    print("=" * 100)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified spacing table. Author: Grzegorz Olbryk."
    )
    parser.add_argument('--N_list', type=int, nargs='+',
                        default=[3, 5, 6, 7, 10, 14])
    parser.add_argument('--kappa_list', type=float, nargs='+',
                        default=[0.5, 1.0, 2.0])
    parser.add_argument('--n_plaq_list', type=int, nargs='+',
                        default=[2, 3, 4, 5])
    parser.add_argument('--n_reps', type=int, default=20)
    args = parser.parse_args()

    rows = build_table(args.N_list, args.kappa_list, args.n_plaq_list,
                       args.n_reps)
    print_table(rows)


if __name__ == '__main__':
    main()
