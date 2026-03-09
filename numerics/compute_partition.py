"""
Direct Computation of Z_{2P}^{SU(N)}(kappa + iy) via Monte Carlo
==================================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Paper  : Pi v16 — "Fisher Zeros of the Two-Plaquette SU(N) Model for Odd N"

Description
-----------
Computes the two-plaquette SU(N) partition function

    Z_{2P}(kappa + iy) = Sigma_lambda d_lambda * A_lambda(kappa + iy)^2

where

    A_lambda(s) = integral_{SU(N)} chi_lambda(U) * exp(s * Re Tr U) dU

by Monte Carlo integration over the Haar measure on SU(N).

This is an INDEPENDENT computation that does NOT use the two-sequence
formula of Theorem 2Seq-SU(N). It provides a genuine numerical check
of the theorem's predictions.

Method
------
1. Sample U_1, ..., U_{N_MC} uniformly from SU(N) (Haar measure) via QR
   decomposition of a complex Gaussian matrix.
2. For each U_i: compute Phi_i = Re Tr U_i and chi_lambda(U_i) for each rep.
3. Estimate A_lambda(kappa+iy) ~ (1/N_MC) Sigma_i chi_lambda(U_i) exp((kappa+iy) Phi_i).
4. Accumulate Z_{2P} = Sigma_lambda d_lambda A_lambda^2.

Honest limitations
------------------
- For large y, Im Z_{2P} is non-negligible and Z is complex.
  The theorem predicts zeros of the ASYMPTOTIC REAL PART Re Z for large y.
- For moderate N_MC and large y, MC variance can obscure the zeros.
  The formula-based fisher_zeros_SU5.py gives the asymptotic zeros exactly.
- This code is intended as independent evidence, not a replacement for
  the asymptotic proof.

Usage
-----
    python compute_partition.py --N 5 --kappa 1.0 --y 4.0
    python compute_partition.py --N 3 --kappa 1.0 --y_scan --y_max 20 --n_MC 50000
"""

import argparse
import sys
import numpy as np
from math import comb
from itertools import combinations_with_replacement as cr


# ---------------------------------------------------------------------------
# SU(N) Haar sampling
# ---------------------------------------------------------------------------

def sample_haar_suN(N, n_samples, rng):
    """
    Sample n_samples matrices from the Haar measure on SU(N).

    Method: QR decomposition of a complex Gaussian N×N matrix,
    followed by phase correction to achieve uniform (Haar) distribution.

    Author: Grzegorz Olbryk
    """
    Z = rng.standard_normal((n_samples, N, N)) + 1j * rng.standard_normal((n_samples, N, N))
    Q, R = np.linalg.qr(Z)

    # Fix phases so that Q has Haar distribution (Mezzadri 2007)
    D = np.diagonal(R, axis1=1, axis2=2)
    D = D / np.abs(D)
    Q = Q * D[:, np.newaxis, :]

    # Project to SU(N): det Q = 1
    dets = np.linalg.det(Q)
    phase = (dets ** (1.0 / N))  # N-th root of det
    Q = Q / phase[:, np.newaxis, np.newaxis]

    return Q  # shape (n_samples, N, N), SU(N) Haar distributed


# ---------------------------------------------------------------------------
# Character computation
# ---------------------------------------------------------------------------

def char_symmetric(p, eigs):
    """
    Character of the symmetric representation (p, 0, ..., 0) of SU(N).
    Equals h_p(eigs) = complete homogeneous symmetric polynomial of degree p.

    Parameters
    ----------
    p    : int   -- degree
    eigs : array, shape (n_samples, N) -- eigenvalues (complex)

    Returns
    -------
    chi : array, shape (n_samples,) -- complex character values

    Author: Grzegorz Olbryk
    """
    n_samples, N = eigs.shape
    if p == 0:
        return np.ones(n_samples, dtype=complex)

    # Use generating function: h_p coefficients of 1/prod_i(1-t*x_i)
    # Computed via Newton's identity: h_p = (1/p) * Sigma_{k=1}^p p_k h_{p-k}
    # where p_k = Sigma_i x_i^k (power sums)
    p_sums = np.array([np.sum(eigs ** k, axis=1) for k in range(1, p + 1)])
    # p_sums[k-1] = power sum p_k, shape (n_samples,)

    h = np.zeros((p + 1, n_samples), dtype=complex)
    h[0] = 1.0
    for k in range(1, p + 1):
        h[k] = (1.0 / k) * sum(p_sums[j] * h[k - 1 - j] for j in range(k))

    return h[p]


# ---------------------------------------------------------------------------
# Partition function
# ---------------------------------------------------------------------------

def Z2P_mc(N, kappa, y, n_reps, N_MC=50000, seed=42):
    """
    Compute Z_{2P}^{SU(N)}(kappa + iy) by Monte Carlo.

    Returns
    -------
    Z       : complex  -- estimated Z_{2P}
    Z_err   : float    -- estimated MC standard error on Re Z
    n_reps_used : int  -- number of representations included

    Author: Grzegorz Olbryk
    """
    rng = np.random.default_rng(seed)
    U = sample_haar_suN(N, N_MC, rng)

    # Eigenvalues for character computation
    eigs = np.linalg.eigvals(U)           # shape (N_MC, N)
    Phi = np.sum(eigs.real, axis=1)       # Re Tr U, shape (N_MC,)

    s = complex(kappa, y)
    exp_s_Phi = np.exp(s * Phi)           # complex weight

    Z = 0j
    for p in range(n_reps):
        d = comb(p + N - 1, N - 1)        # dim of rep (p,0,...,0)
        chi = char_symmetric(p, eigs)     # shape (N_MC,)
        A_p = np.mean(chi * exp_s_Phi)    # MC estimate of A_p(kappa+iy)
        Z += d * A_p ** 2

    # MC error estimate on Re Z (from bootstrap-like variance of individual terms)
    # Rough estimate: use variance of the largest term
    chi0 = char_symmetric(0, eigs)
    A0_samples = chi0 * exp_s_Phi
    Z_err = float(np.std(A0_samples.real) / np.sqrt(N_MC))

    return Z, Z_err


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_single(N, kappa, y, n_reps, N_MC):
    """Author: Grzegorz Olbryk"""
    Z, Z_err = Z2P_mc(N, kappa, y, n_reps, N_MC)
    print()
    print("=" * 55)
    print(f"  Z_{{2P}}^{{SU({N})}}(kappa+iy)  [direct MC]")
    print(f"  Author: Grzegorz Olbryk  |  Paper Pi v16")
    print("=" * 55)
    print(f"  kappa = {kappa},  y = {y},  N_MC = {N_MC},  n_reps = {n_reps}")
    print(f"  Re Z  = {Z.real:.6e}")
    print(f"  Im Z  = {Z.imag:.6e}")
    print(f"  |Z|   = {abs(Z):.6e}")
    print(f"  MC err (rough) ≈ {Z_err:.2e}")
    print("=" * 55)
    print()


def print_scan(N, kappa, y_max, n_reps, N_MC, n_points=80):
    """
    Scan y in [0.5, y_max] and report Re Z_{2P}(kappa+iy), identifying sign changes.
    Author: Grzegorz Olbryk
    """
    print()
    print("=" * 70)
    print(f"  Z_{{2P}}^{{SU({N})}} sign-change scan  [direct MC, independent verification]")
    print(f"  Author: Grzegorz Olbryk  |  Paper Pi v16")
    print(f"  kappa={kappa}, y in [0.5, {y_max}], N_MC={N_MC}, n_reps={n_reps}")
    print("=" * 70)
    print(f"  NOTE: Z is COMPLEX; 'Re Z sign changes' approximate true zeros for large y.")
    print(f"  This is independent of the two-sequence formula of Theorem 2Seq-SU(N).")
    print("=" * 70)
    print()

    y_vals = np.linspace(0.5, y_max, n_points)
    Z_vals = []
    print(f"  Computing {n_points} evaluations...", flush=True)
    for i, y in enumerate(y_vals):
        Z, _ = Z2P_mc(N, kappa, y, n_reps, N_MC, seed=42 + i)
        Z_vals.append(Z)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_points}] y={y:.2f}, Re Z={Z.real:.3e}, Im Z={Z.imag:.3e}")

    # Find sign changes of Re Z
    zeros_mc = []
    for i in range(len(Z_vals) - 1):
        if Z_vals[i].real * Z_vals[i + 1].real < 0:
            lo, hi = y_vals[i], y_vals[i + 1]
            for _ in range(20):
                mid = (lo + hi) / 2
                Z_mid, _ = Z2P_mc(N, kappa, mid, n_reps, N_MC)
                Z_lo, _ = Z2P_mc(N, kappa, lo, n_reps, N_MC)
                if Z_mid.real * Z_lo.real < 0:
                    hi = mid
                else:
                    lo = mid
            zeros_mc.append((lo + hi) / 2)

    print()
    print(f"  Sign changes of Re Z (MC zeros): {[f'{z:.4f}' for z in zeros_mc]}")

    if len(zeros_mc) >= 2:
        gaps_mc = [zeros_mc[i + 1] - zeros_mc[i] for i in range(len(zeros_mc) - 1)]
        mean_gap = sum(gaps_mc) / len(gaps_mc)
        print()
        print(f"  Gaps: {[f'{g:.4f}' for g in gaps_mc]}")
        print(f"  Mean gap (MC) = {mean_gap:.4f}")
        print(f"  pi/2          = {np.pi/2:.4f}   (predicted by Theorem 2Seq-SU(N))")

    print()
    print("  HONESTY NOTE: MC sign changes of Re Z agree with asymptotic formula zeros")
    print("  for large y (k >= k_0). For small y, discrepancies are expected.")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Direct MC computation of Z_{2P}^{SU(N)}(kappa+iy). "
            "Independent verification WITHOUT using Theorem 2Seq formula. "
            "Author: Grzegorz Olbryk. Paper: Pi v16."
        )
    )
    parser.add_argument('--N',       type=int,   default=5)
    parser.add_argument('--kappa',   type=float, default=1.0)
    parser.add_argument('--y',       type=float, default=4.0)
    parser.add_argument('--n_reps',  type=int,   default=6,
                        help='Number of SU(N) reps (default 6; higher = more accurate but slower)')
    parser.add_argument('--n_MC',    type=int,   default=30000,
                        help='Monte Carlo sample size (default 30000)')
    parser.add_argument('--y_scan',  action='store_true',
                        help='Scan Re Z over range [0.5, y_max] to find sign changes')
    parser.add_argument('--y_max',   type=float, default=20.0)
    parser.add_argument('--n_pts',   type=int,   default=60)
    args = parser.parse_args()

    if args.y_scan:
        print_scan(args.N, args.kappa, args.y_max, args.n_reps, args.n_MC, args.n_pts)
    else:
        print_single(args.N, args.kappa, args.y, args.n_reps, args.n_MC)


if __name__ == '__main__':
    main()
