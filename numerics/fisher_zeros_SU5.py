from math import comb, pi, cos, sin, sqrt, acos, exp
"""
Fisher Zeros of the Two-Plaquette SU(5) Model
==============================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Paper  : Pi v16 — "Fisher Zeros of the Two-Plaquette SU(N) Model for Odd N"

Description
-----------
Computes the Fisher zeros of Z_{2P}^{SU(5)}(iy, kappa) for imaginary coupling
y > 0, using the two-sequence structure established in Theorem 2Seq-SU(N).

Method
------
1. Compute S_A and S_AB via character expansion up to degree n_reps.
2. Compute rho_N = -S_AB / (2 S_A) and alpha_N = arccos(rho_N).
3. Generate the two zero families:
       y_k^(+) = (alpha_N + 2*pi*k) / 2
       y_k^(-) = (2*pi - alpha_N + 2*pi*k) / 2
4. Verify: each adjacent pair sums to pi; mean gap = pi/2.

Reference: Paper Pi v16, §4 (Theorem 2Seq-SU(N)).

Usage
-----
    python fisher_zeros_SU5.py
    python fisher_zeros_SU5.py --kappa 2.0 --n_zeros 40 --n_reps 25
"""

import argparse
import sys
from itertools import combinations_with_replacement as cr

# ---------------------------------------------------------------------------
# Character expansion helpers
# ---------------------------------------------------------------------------

def h_p(p, eigenvalues):
    """
    Complete homogeneous symmetric polynomial h_p evaluated at `eigenvalues`.

    h_p(x_1,...,x_n) = sum_{|alpha|=p} x_1^{alpha_1} ... x_n^{alpha_n}

    Author: Grzegorz Olbryk
    """
    result = 0.0
    for combo in cr(range(len(eigenvalues)), p):
        term = 1.0
        for idx in combo:
            term *= eigenvalues[idx]
        result += term
    return result


def casimir_su5(p):
    """
    Quadratic Casimir C_2((p,0,0,0)) for SU(5) in the representation
    labelled by a single non-negative integer p (symmetric representations).

        C_2 = p*(p + N) / N  with N=5.

    Author: Grzegorz Olbryk
    """
    return p * (p + 5) / 5.0


def dimension_su5(p):
    """
    Dimension of the SU(5) representation (p,0,0,0):
        d = C(p+4, 4).

    Author: Grzegorz Olbryk
    """
    return comb(p + 4, 4)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_amplitudes(kappa, n_reps=20):
    """
    Compute S_A, S_AB for SU(5) at coupling kappa.

    Dominant saddles (N=5 odd, balanced split):
        U_A = diag(1, 1, 1, -1, -1)   =>  chi_A = h_p([1,1,1,-1,-1])
        U_B = diag(1, 1, -1, -1, -1)  =>  chi_B = h_p([1,1,-1,-1,-1])

    By Lemma SA (Schur parity): chi_B = (-1)^p * chi_A.

    Returns
    -------
    S_A  : float  -- Sigma_p d_p * chi_A^2 * A_p(kappa)
    S_AB : float  -- Sigma_p d_p * chi_A * chi_B * A_p(kappa)
                     = Sigma_p d_p * (-1)^p * chi_A^2 * A_p(kappa)

    Author: Grzegorz Olbryk
    """
    eig_A = [1.0, 1.0, 1.0, -1.0, -1.0]   # U_A eigenvalues, Phi = +1
    S_A = 0.0
    S_AB = 0.0

    for p in range(n_reps):
        d = dimension_su5(p)
        chi_A = h_p(p, eig_A)


        import math
        A_p = d * math.exp(-casimir_su5(p) * kappa)
        # Lemma SA: chi_B = (-1)^p * chi_A
        chi_B = ((-1) ** p) * chi_A

        S_A  += d * chi_A * chi_A * A_p
        S_AB += d * chi_A * chi_B * A_p

    return S_A, S_AB


def compute_alpha(kappa, n_reps=20):
    """
    Compute rho_N and alpha_N for SU(5) at given kappa.

    rho_N  = -S_AB / (2 * S_A)  in (-1, 1)
    alpha_N = arccos(rho_N)      in (0, pi)

    Author: Grzegorz Olbryk
    """
    S_A, S_AB = compute_amplitudes(kappa, n_reps)
    rho = -S_AB / (2.0 * S_A)
    rho = max(-1.0 + 1e-12, min(1.0 - 1e-12, rho))
    alpha = acos(rho)
    return S_A, S_AB, rho, alpha


def fisher_zeros(kappa, n_zeros=20, n_reps=20):
    """
    Compute the first n_zeros Fisher zeros of Z_{2P}^{SU(5)}(iy, kappa).

    Zero families (Theorem 2Seq-SU(N), Paper Pi v16):
        y_k^(+) = (alpha + 2*pi*k) / 2    [family (+)]
        y_k^(-) = (2*pi - alpha + 2*pi*k) / 2    [family (-)]

    Returns
    -------
    zeros  : list of (y_k, family, gap_to_previous)
    S_A, S_AB, rho, alpha : float -- expansion parameters

    Author: Grzegorz Olbryk
    """
    S_A, S_AB, rho, alpha = compute_alpha(kappa, n_reps)

    # Generate enough terms from each family
    k_max = n_zeros + 5
    fam_plus  = [(alpha / 2.0 + pi * k, '+') for k in range(k_max)]
    fam_minus = [((2.0 * pi - alpha) / 2.0 + pi * k, '-') for k in range(k_max)]

    all_zeros = sorted(fam_plus + fam_minus, key=lambda x: x[0])
    all_zeros = [z for z in all_zeros if z[0] > 0][:n_zeros + 1]

    result = []
    for i, (y, fam) in enumerate(all_zeros):
        if i == 0:
            result.append({'k': i, 'y': y, 'family': fam, 'gap': None, 'gap_type': None})
        else:
            gap = y - all_zeros[i - 1][0]
            gap_type = 'pi-alpha' if abs(gap - (pi - alpha)) < 1e-6 else 'alpha'
            result.append({'k': i, 'y': y, 'family': fam, 'gap': gap, 'gap_type': gap_type})

    return result[:n_zeros], S_A, S_AB, rho, alpha


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(zeros, alpha):
    """
    Check that:
      (a) every adjacent pair sums to pi,
      (b) mean gap = pi/2.

    Author: Grzegorz Olbryk
    """
    gaps = [z['gap'] for z in zeros if z['gap'] is not None]
    pair_sums = [gaps[i] + gaps[i + 1] for i in range(0, len(gaps) - 1, 2)]
    all_pi = all(abs(s - pi) < 1e-6 for s in pair_sums)
    mean_gap = sum(gaps) / len(gaps)
    return pair_sums, all_pi, mean_gap


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(zeros, kappa, S_A, S_AB, rho, alpha):
    """
    Print a formatted table of Fisher zeros.
    Author: Grzegorz Olbryk
    """
    print()
    print("=" * 65)
    print("  Fisher Zeros of Z_{2P}^{SU(5)}(iy, kappa)")
    print("  Author: Grzegorz Olbryk  |  Paper Pi v16")
    print("=" * 65)
    print(f"  kappa  = {kappa}")
    print(f"  S_A    = {S_A:.6f}")
    print(f"  S_AB   = {S_AB:.6f}")
    print(f"  rho    = {rho:.6f}")
    print(f"  alpha  = {alpha:.6f} rad = {alpha/pi:.4f} * pi")
    print(f"  gaps   : {pi - alpha:.5f}  (pi-alpha)  and  {alpha:.5f}  (alpha)")
    print("=" * 65)
    print(f"  {'k':>3}  {'family':>6}  {'y_k':>10}  {'gap':>10}  {'type':>8}")
    print("  " + "-" * 43)
    for z in zeros:
        gap_str = f"{z['gap']:.5f}" if z['gap'] is not None else "      —"
        type_str = z['gap_type'] if z['gap_type'] else "—"
        print(f"  {z['k']:>3}  {z['family']:>6}  {z['y']:>10.5f}  {gap_str:>10}  {type_str:>8}")
    print()

    gaps = [z['gap'] for z in zeros if z['gap'] is not None]
    pair_sums, all_pi, mean_gap = verify(zeros, alpha)
    print(f"  Pair sums (should all = pi = {pi:.5f}):")
    for i, s in enumerate(pair_sums[:5]):
        print(f"    pair {i}: {s:.5f}  {'✓' if abs(s - pi) < 1e-6 else '✗'}")
    if len(pair_sums) > 5:
        print(f"    ... ({len(pair_sums)} pairs total, all {'✓' if all_pi else '✗'})")
    print()
    print(f"  Mean of {len(gaps)} gaps = {mean_gap:.5f}")
    print(f"  pi/2                  = {pi/2:.5f}  {'✓' if abs(mean_gap - pi/2) < 1e-5 else '✗'}")
    print("=" * 65)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fisher zeros of the two-plaquette SU(5) model. "
                    "Author: Grzegorz Olbryk. Paper: Pi v16."
    )
    parser.add_argument('--kappa',   type=float, default=1.0,
                        help='Coupling constant kappa > 0 (default: 1.0)')
    parser.add_argument('--n_zeros', type=int,   default=20,
                        help='Number of zeros to compute (default: 20)')
    parser.add_argument('--n_reps',  type=int,   default=20,
                        help='Number of representations in expansion (default: 20)')
    args = parser.parse_args()

    if args.kappa <= 0:
        print("Error: kappa must be positive.", file=sys.stderr)
        sys.exit(1)

    zeros, S_A, S_AB, rho, alpha = fisher_zeros(
        kappa=args.kappa, n_zeros=args.n_zeros, n_reps=args.n_reps
    )
    print_table(zeros, args.kappa, S_A, S_AB, rho, alpha)


if __name__ == '__main__':
    main()
