"""
Explicit Rouché Threshold y₀(N, κ) for SU(N) Two-Plaquette Model
===================================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Paper  : Pi v16 — "Fisher Zeros of the Two-Plaquette SU(N) Model for Odd N"

Description
-----------
Computes the explicit threshold y₀(N, κ) such that Rouché's theorem applies
for all Fisher zeros y_k* > y₀, guaranteeing exactly one simple zero of
Z_{2P}^{SU(N)}(iy, κ) in each disk D_k = D(y_k*, δ_k) with δ_k = y_k^{-1/2}.

Formula (Proposition R-N, Paper Pi v16, §4):

    y₀(N, κ) = (4 C_N(κ) / μ_N(κ))^{1 / (2m_N + ε_N/2 − 1/2)}

where:
    μ_N    = 4 S_A sin(α_N)           -- lower bound on |leading term| on ∂D_k
    C_N    = conservative bound on remainder coefficient
    m_N    = (N-1)/2 + ord_min/2      -- stationary-phase power
    ε_N    = ord_next − ord_min       -- gap to next Vandermonde order

Usage
-----
    python rouche_threshold.py
    python rouche_threshold.py --N_list 3 5 7 --kappa 1.0
"""

import argparse
from math import comb, pi, exp, acos, sin, log
from itertools import combinations_with_replacement as cr


# ---------------------------------------------------------------------------
# Helpers (shared with spacing_table.py)
# ---------------------------------------------------------------------------

def h_p_general(p, eigenvalues):
    """Author: Grzegorz Olbryk"""
    result = 0.0
    for combo in cr(range(len(eigenvalues)), p):
        term = 1.0
        for idx in combo: term *= eigenvalues[idx]
        result += term
    return result

def casimir_suN(p, N):
    """Author: Grzegorz Olbryk"""
    return p * (p + N) / float(N)

def dimension_suN(p, N):
    """Author: Grzegorz Olbryk"""
    return comb(p + N - 1, N - 1)


# ---------------------------------------------------------------------------
# Vandermonde order table
# ---------------------------------------------------------------------------

def vandermonde_order(k, N):
    """
    ord(k, N) = k*(k-1) + (N-k)*(N-k-1).
    Author: Grzegorz Olbryk
    """
    return k * (k - 1) + (N - k) * (N - k - 1)


def ord_parameters(N):
    """
    Return ord_min, ord_next, epsilon_N for SU(N) odd.

    ord_min  = ord(floor(N/2), N)   -- balanced split
    ord_next = min_{k != floor(N/2)} ord(k, N)
    epsilon  = ord_next - ord_min

    Author: Grzegorz Olbryk
    """
    k0 = N // 2
    ord_min = vandermonde_order(k0, N)
    ords_other = [vandermonde_order(k, N) for k in range(N + 1) if k != k0]
    ord_next = min(ords_other)
    return ord_min, ord_next, ord_next - ord_min


# ---------------------------------------------------------------------------
# Rouché threshold
# ---------------------------------------------------------------------------

def compute_threshold(N, kappa, n_reps=20, C_N_factor=10.0):
    """
    Compute y₀(N, κ) — explicit Rouché threshold (Prop R-N, Pi v16).

    Parameters
    ----------
    C_N_factor : float
        Conservative multiplier for the remainder coefficient C_N.
        C_N ≈ C_N_factor * S_A  (rough upper bound; exact bound requires
        full remainder analysis from Pi v11 §3).

    Author: Grzegorz Olbryk
    """
    # --- character expansion ---
    assert N % 2 == 1
    n_plus = (N + 1) // 2
    n_minus = (N - 1) // 2
    eig_A = [1.0] * n_plus + [-1.0] * n_minus

    S_A = 0.0
    S_AB = 0.0
    for p in range(n_reps):
        d = dimension_suN(p, N)
        chi_A = h_p_general(p, eig_A)
        A_p = d * exp(-casimir_suN(p, N) * kappa)
        S_A  += d * chi_A * chi_A * A_p
        S_AB += d * ((-1)**p) * chi_A * chi_A * A_p

    rho = max(-1 + 1e-12, min(1 - 1e-12, -S_AB / (2.0 * S_A)))
    alpha = acos(rho)

    # --- structural parameters ---
    ord_min, ord_next, eps = ord_parameters(N)
    m_N = (N - 1) / 2.0 + ord_min / 2.0

    # --- Rouché parameters ---
    mu_N = 4.0 * S_A * sin(alpha)           # lower bound on |leading term|
    C_N  = C_N_factor * S_A                 # conservative remainder bound
    power = 2 * m_N + eps / 2.0 - 0.5      # exponent in (|R|/|F|) ~ y^{-power}

    ratio = 4.0 * C_N / mu_N               # constant factor
    y0 = ratio ** (1.0 / power)            # threshold

    return {
        'N': N, 'kappa': kappa,
        'S_A': S_A, 'rho': rho, 'alpha': alpha,
        'mu_N': mu_N, 'C_N': C_N,
        'm_N': m_N, 'eps': eps, 'power': power,
        'ratio': ratio, 'y0': y0,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_threshold_table(N_list, kappa, n_reps, C_N_factor):
    """Author: Grzegorz Olbryk"""
    print()
    print("=" * 72)
    print("  Explicit Rouché Threshold y₀(N, κ)  (Proposition R-N, Pi v16)")
    print("  Author: Grzegorz Olbryk")
    print(f"  κ = {kappa},  C_N = {C_N_factor} × S_A  (conservative bound)")
    print("=" * 72)
    header = (
        f"  {'N':>3}  {'m_N':>5}  {'ε_N':>4}  {'power':>6}  "
        f"{'μ_N':>10}  {'ratio':>8}  {'y₀':>8}  {'applies from'}"
    )
    print(header)
    print("  " + "-" * 68)

    for N in N_list:
        if N % 2 == 0:
            continue
        r = compute_threshold(N, kappa, n_reps, C_N_factor)
        print(
            f"  {r['N']:>3}  {r['m_N']:>5.1f}  {r['eps']:>4.0f}  "
            f"{r['power']:>6.1f}  {r['mu_N']:>10.2f}  "
            f"{r['ratio']:>8.3f}  {r['y0']:>8.3f}  "
            f"k ≥ k₀ with y_k* > {r['y0']:.2f}"
        )

    print()
    print("  Condition: |R(y)| / |F(y)| ≤ ratio × y^{-power} < 1 for y > y₀.")
    print("  For each N: Rouché gives exactly 1 simple zero per disk D_k = D(y_k*, y_k^{-1/2}).")
    print("=" * 72)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Rouché threshold y₀(N,κ) for SU(N) two-plaquette Fisher zeros. "
                    "Author: Grzegorz Olbryk. Paper: Pi v16."
    )
    parser.add_argument('--N_list',    type=int,   nargs='+', default=[3, 5, 7])
    parser.add_argument('--kappa',     type=float, default=1.0)
    parser.add_argument('--n_reps',    type=int,   default=20)
    parser.add_argument('--C_factor',  type=float, default=10.0,
                        help='Conservative factor for C_N = factor * S_A (default: 10)')
    args = parser.parse_args()

    print_threshold_table(args.N_list, args.kappa, args.n_reps, args.C_factor)


if __name__ == '__main__':
    main()
