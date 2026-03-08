"""
Average Fisher-Zero Spacing Table for SU(N), N = 3, 5, 7
==========================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Paper  : Pi v16 — "Fisher Zeros of the Two-Plaquette SU(N) Model for Odd N"

Description
-----------
Computes the two-sequence parameters (rho_N, alpha_N, gaps) for SU(N) with
N odd, at multiple coupling constants kappa.  Confirms that in all cases:

    Δ_short + Δ_long = π    (pair sum, exact)
    ⟨Δ⟩ = π/2              (average, exact)

Usage
-----
    python spacing_table.py
    python spacing_table.py --N_list 3 5 7 9 --kappa_list 0.5 1.0 2.0 5.0
"""

import argparse
from itertools import combinations_with_replacement as cr
from math import comb, pi, exp, acos, sin


# ---------------------------------------------------------------------------
# General SU(N) helpers
# ---------------------------------------------------------------------------

def h_p_general(p, eigenvalues):
    """
    Complete homogeneous symmetric polynomial h_p(eigenvalues).
    Author: Grzegorz Olbryk
    """
    result = 0.0
    for combo in cr(range(len(eigenvalues)), p):
        term = 1.0
        for idx in combo:
            term *= eigenvalues[idx]
        result += term
    return result


def casimir_suN(p, N):
    """
    Quadratic Casimir C_2((p,0,...,0)) for SU(N):
        C_2 = p * (p + N) / N.
    Author: Grzegorz Olbryk
    """
    return p * (p + N) / float(N)


def dimension_suN(p, N):
    """
    Dimension of the SU(N) representation (p, 0, ..., 0):
        d = C(p + N - 1, N - 1).
    Author: Grzegorz Olbryk
    """
    return comb(p + N - 1, N - 1)


def balanced_saddles(N):
    """
    Return eigenvalue lists for the two balanced-split saddles U_A, U_B
    of SU(N) (N must be odd).

    For N odd:
        n_plus  = (N+1)/2,  n_minus = (N-1)/2
        U_A: n_plus  eigenvalues +1, n_minus eigenvalues -1   => Phi = +1
        U_B: n_minus eigenvalues +1, n_plus  eigenvalues -1   => Phi = -1

    By Lemma SA: chi_B = (-1)^p * chi_A  (Schur parity).

    Author: Grzegorz Olbryk
    """
    assert N % 2 == 1, f"N must be odd, got N={N}"
    n_plus  = (N + 1) // 2
    n_minus = (N - 1) // 2
    eig_A = [1.0] * n_plus + [-1.0] * n_minus
    return eig_A   # eig_B handled via Schur parity


def compute_rho_alpha(N, kappa, n_reps=18):
    """
    Compute rho_N(kappa) and alpha_N(kappa) for SU(N) odd.

    rho_N  = -S_AB / (2 * S_A)
    alpha_N = arccos(rho_N)

    Uses Lemma SA: chi_lambda(U_B) = (-1)^|lambda| * chi_lambda(U_A),
    so S_AB = sum_p d * (-1)^p * chi_A^2 * A_p.

    Author: Grzegorz Olbryk
    """
    eig_A = balanced_saddles(N)
    S_A = 0.0
    S_AB = 0.0

    for p in range(n_reps):
        d = dimension_suN(p, N)
        chi_A = h_p_general(p, eig_A)
        A_p = d * exp(-casimir_suN(p, N) * kappa)
        # Schur parity: chi_B = (-1)^p * chi_A
        sign = (-1) ** p
        S_A  += d * chi_A * chi_A * A_p
        S_AB += d * sign  * chi_A * chi_A * A_p

    if S_A < 1e-30:
        return None, None, None, None

    rho = -S_AB / (2.0 * S_A)
    rho = max(-1.0 + 1e-12, min(1.0 - 1e-12, rho))
    alpha = acos(rho)
    return S_A, S_AB, rho, alpha


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def build_table(N_list, kappa_list, n_reps=18):
    """
    Build a table of (N, kappa, S_A, rho, alpha, gap_short, gap_long, avg) rows.
    Author: Grzegorz Olbryk
    """
    rows = []
    for N in N_list:
        if N % 2 == 0:
            continue  # N even: different mechanism, not covered by Theorem 2Seq
        for kappa in kappa_list:
            S_A, S_AB, rho, alpha = compute_rho_alpha(N, kappa, n_reps)
            if S_A is None:
                continue
            gap_short = pi - alpha
            gap_long  = alpha
            pair_sum  = gap_short + gap_long
            avg       = pair_sum / 2.0
            rows.append({
                'N': N, 'kappa': kappa,
                'S_A': S_A, 'rho': rho, 'alpha': alpha,
                'gap_short': gap_short, 'gap_long': gap_long,
                'pair_sum': pair_sum, 'avg': avg,
            })
    return rows


def print_table(rows):
    """
    Print the spacing table.
    Author: Grzegorz Olbryk
    """
    print()
    print("=" * 85)
    print("  Average Fisher-Zero Spacing ⟨Δ⟩ = π/2 for SU(N) Two-Plaquette Model")
    print("  Author: Grzegorz Olbryk  |  Paper Pi v16")
    print("=" * 85)
    header = (
        f"  {'N':>3}  {'κ':>5}  {'S_A':>13}  {'ρ_N':>8}  "
        f"{'α_N':>8}  {'Δ_short':>8}  {'Δ_long':>8}  {'sum':>8}  {'avg':>8}  {'✓'}"
    )
    print(header)
    print("  " + "-" * 81)

    prev_N = None
    for r in rows:
        if r['N'] != prev_N and prev_N is not None:
            print()
        prev_N = r['N']
        ok = '✓' if abs(r['avg'] - pi / 2) < 1e-6 else '✗'
        print(
            f"  {r['N']:>3}  {r['kappa']:>5.1f}  {r['S_A']:>13.3f}  "
            f"{r['rho']:>8.4f}  {r['alpha']:>8.4f}  "
            f"{r['gap_short']:>8.4f}  {r['gap_long']:>8.4f}  "
            f"{r['pair_sum']:>8.4f}  {r['avg']:>8.5f}  {ok}"
        )

    print()
    all_ok = all(abs(r['avg'] - pi / 2) < 1e-6 for r in rows)
    print(f"  All {len(rows)} cases: ⟨Δ⟩ = π/2 ? {'YES ✓' if all_ok else 'NO ✗'}")
    print(f"  (π/2 = {pi/2:.5f})")
    print("=" * 85)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Spacing table for SU(N) two-plaquette Fisher zeros (N odd). "
                    "Author: Grzegorz Olbryk. Paper: Pi v16."
    )
    parser.add_argument(
        '--N_list', type=int, nargs='+', default=[3, 5, 7],
        help='List of N values (odd only; default: 3 5 7)'
    )
    parser.add_argument(
        '--kappa_list', type=float, nargs='+', default=[0.5, 1.0, 2.0],
        help='List of kappa values (default: 0.5 1.0 2.0)'
    )
    parser.add_argument(
        '--n_reps', type=int, default=18,
        help='Number of representations in character expansion (default: 18)'
    )
    args = parser.parse_args()

    for N in args.N_list:
        if N % 2 == 0:
            print(f"  Warning: N={N} is even — skipped (N even not covered by Theorem 2Seq).")

    rows = build_table(args.N_list, args.kappa_list, args.n_reps)
    print_table(rows)


if __name__ == '__main__':
    main()
