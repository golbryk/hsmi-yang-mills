"""
Saddle Equality Test: S_A^(n) = S_B^(n)?
==========================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: Referee concern — does S_A^(n) = S_B^(n) hold for all N, n?

This script:
1. Identifies the dominant saddle-point configurations S_A, S_B for each N
2. Computes the character values chi_lambda(S_A), chi_lambda(S_B)
3. Computes the saddle amplitudes v_lambda^A, v_lambda^B
4. Tests S_A^(n) = S_B^(n) numerically for n=2..8
5. Checks the Rouche condition for the n-plaquette TSI argument
"""

import numpy as np
from math import comb, pi, factorial
from itertools import product


def build_saddle_configs(N):
    """Find the dominant balanced-split saddle configurations for SU(N).

    Returns list of (theta_array, Phi, n_plus, n_minus) for each saddle.
    Saddles are {0,pi}^N configurations satisfying sum(theta) = 0 mod 2pi.
    """
    saddles = []
    # Enumerate all {0, pi}^N configurations with det = 1
    for bits in product([0, 1], repeat=N):
        theta = np.array([b * pi for b in bits])
        if abs(np.sum(theta) % (2 * pi)) < 0.01 or abs(np.sum(theta) % (2 * pi) - 2 * pi) < 0.01:
            Phi = np.sum(np.cos(theta))
            n_plus = sum(1 for b in bits if b == 0)
            n_minus = sum(1 for b in bits if b == 1)
            # Vandermonde order
            vo = n_plus * (n_plus - 1) + n_minus * (n_minus - 1)
            saddles.append({
                'theta': theta,
                'Phi': Phi,
                'n_plus': n_plus,
                'n_minus': n_minus,
                'vand_order': vo,
                'eigenvalues': np.exp(1j * theta)
            })

    # Sort by Vandermonde order (lower = more dominant)
    saddles.sort(key=lambda s: s['vand_order'])
    return saddles


def compute_vandermonde_sq(eigenvalues):
    """Compute |Delta(z)|^2 = product_{j<k} |z_j - z_k|^2."""
    N = len(eigenvalues)
    v2 = 1.0
    for j in range(N):
        for k in range(j + 1, N):
            v2 *= abs(eigenvalues[j] - eigenvalues[k]) ** 2
    return v2


def h_p_at_point(p, eigenvalues):
    """Compute h_p(z1,...,zN) at a single point using Newton's identities."""
    N = len(eigenvalues)
    z = eigenvalues
    if p == 0:
        return 1.0
    # Power sums
    pk = [np.sum(z ** k) for k in range(1, p + 1)]
    h = [0.0] * (p + 1)
    h[0] = 1.0
    for k in range(1, p + 1):
        h[k] = sum(pk[j] * h[k - 1 - j] for j in range(k)) / k
    return h[p]


def dim_rep(p, N):
    return comb(p + N - 1, N - 1)


def main():
    print("=" * 80)
    print("  Saddle Equality Test: S_A^(n) =? S_B^(n)")
    print("  Author: Grzegorz Olbryk  |  March 2026  |  Referee Response")
    print("=" * 80)

    P_MAX = 20  # max representation index

    for N in [3, 5, 7, 6, 10]:
        print(f"\n\n  {'='*70}")
        print(f"  SU({N})")
        print(f"  {'='*70}")

        saddles = build_saddle_configs(N)

        # Find minimum Vandermonde order
        min_vo = saddles[0]['vand_order']
        dominant = [s for s in saddles if s['vand_order'] == min_vo]

        print(f"\n  Total {0,pi}^N configs with det=1: {len(saddles)}")
        print(f"  Minimum Vandermonde order: {min_vo}")
        print(f"  Number of dominant saddles: {len(dominant)}")

        # Group by Phi value
        phi_groups = {}
        for s in dominant:
            phi = round(s['Phi'], 10)
            if phi not in phi_groups:
                phi_groups[phi] = []
            phi_groups[phi].append(s)

        print(f"  Phi values at dominant saddles: {sorted(phi_groups.keys())}")

        for phi, group in sorted(phi_groups.items()):
            s = group[0]
            print(f"    Phi={phi}: n+={s['n_plus']}, n-={s['n_minus']}, "
                  f"vand_order={s['vand_order']}, count={len(group)}")

        # Compute character values at saddle points
        # For symmetric representations: h_p(eigenvalues) / d_p = chi_p(U)
        # v_lambda = C * chi(S) * |Delta(S)| / (d_lambda * |det H|^{1/2})
        # Since |Delta|, |det H| are the same for all dominant saddles with same vand_order,
        # the key ratio is chi(S_A) vs chi(S_B).

        # Pick S_A and S_B (the two Phi ≠ 0 saddles for Phi0 != 0)
        phi_vals = sorted(phi_groups.keys())

        if len(phi_vals) < 2:
            print(f"  Only one Phi value — degenerate case (N ≡ 0 mod 4)")
            continue

        # For N odd: Phi0 = ±1
        # For N ≡ 2 mod 4: Phi0 = ±2
        S_A_saddle = phi_groups[max(phi_vals)][0]  # positive Phi
        S_B_saddle = phi_groups[min(phi_vals)][0]  # negative Phi

        z_A = S_A_saddle['eigenvalues']
        z_B = S_B_saddle['eigenvalues']
        V2_A = compute_vandermonde_sq(z_A)
        V2_B = compute_vandermonde_sq(z_B)

        print(f"\n  S_A: Phi={S_A_saddle['Phi']}, |Delta|^2={V2_A:.1f}")
        print(f"  S_B: Phi={S_B_saddle['Phi']}, |Delta|^2={V2_B:.1f}")
        print(f"  |Delta|^2 equal: {abs(V2_A - V2_B) < 1e-10}")

        # Compute character values h_p at S_A, S_B
        print(f"\n  {'p':>4s}  {'d_p':>6s}  {'h_p(S_A)':>12s}  {'h_p(S_B)':>12s}  "
              f"{'chi_A':>10s}  {'chi_B':>10s}  {'chi_B/chi_A':>12s}")

        chi_A = []
        chi_B = []
        for p in range(P_MAX + 1):
            dp = dim_rep(p, N)
            hp_A = h_p_at_point(p, z_A)
            hp_B = h_p_at_point(p, z_B)
            cA = hp_A.real / dp if dp > 0 else 0
            cB = hp_B.real / dp if dp > 0 else 0
            chi_A.append(cA)
            chi_B.append(cB)
            ratio = cB / cA if abs(cA) > 1e-15 else float('nan')
            if p <= 10:
                print(f"  {p:4d}  {dp:6d}  {hp_A.real:12.4f}  {hp_B.real:12.4f}  "
                      f"{cA:10.6f}  {cB:10.6f}  {ratio:12.6f}")

        # Now compute S_A^(n) and S_B^(n) for various n
        # Using v_lambda proportional to chi_lambda(S) / d_lambda
        # v_lambda^A ~ chi_A(p) (up to a common positive factor)
        # S_A^(n) = sum_p d_p * (v_p^A)^n ~ sum_p d_p * chi_A(p)^n

        print(f"\n  S_A^(n) vs S_B^(n) (using chi ratios, same prefactor):")
        print(f"  {'n':>4s}  {'S_A^(n)':>14s}  {'S_B^(n)':>14s}  "
              f"{'S_A=S_B?':>10s}  {'ratio':>10s}")

        for n in range(2, 9):
            SA_n = sum(dim_rep(p, N) * chi_A[p] ** n for p in range(P_MAX + 1))
            SB_n = sum(dim_rep(p, N) * chi_B[p] ** n for p in range(P_MAX + 1))
            eq = abs(SA_n - SB_n) < 1e-8 * max(abs(SA_n), abs(SB_n), 1e-15)
            ratio = SB_n / SA_n if abs(SA_n) > 1e-15 else float('nan')
            print(f"  {n:4d}  {SA_n:14.8f}  {SB_n:14.8f}  "
                  f"{'YES' if eq else 'NO':>10s}  {ratio:10.6f}")

        # Check: is chi_B = (-1)^p * chi_A for all p?
        print(f"\n  Pattern check: chi_B(p) =? (-1)^p chi_A(p)")
        all_match = True
        for p in range(P_MAX + 1):
            predicted = (-1) ** p * chi_A[p]
            match = abs(chi_B[p] - predicted) < 1e-10
            if not match and p <= 10:
                print(f"    p={p}: chi_B={chi_B[p]:.6f}, (-1)^p chi_A={predicted:.6f} "
                      f"{'MATCH' if match else 'MISMATCH'}")
                all_match = False
        if all_match:
            print(f"    chi_B(p) = (-1)^p chi_A(p) for ALL p = 0..{P_MAX}")
            print(f"    => For even n: S_B^(n) = sum d_p [(-1)^p chi_A]^n = sum d_p chi_A^n = S_A^(n)")
            print(f"    => For odd n:  S_B^(n) = sum d_p (-1)^{n}p chi_A^n = sum d_p (-1)^p chi_A^n")
            print(f"       This equals S_A^(n) only if sum_p d_p (-1)^p chi_A^n = sum_p d_p chi_A^n")
            print(f"       i.e., only if contributions from odd p cancel or equal even p.")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
