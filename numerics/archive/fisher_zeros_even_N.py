"""
Fisher Zeros of Z_{2P}^{SU(N)}(kappa+iy) for N Even — Numerical Investigation
================================================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: Extension of Pi v27 to N even

Description
-----------
Computes Fisher zeros of the two-plaquette SU(N) partition function
for N even via two independent methods:

Method 1: Exact Weyl integration (N=4 only; 3D quadrature)
Method 2: High-statistics Monte Carlo (any N)

Then analyzes the spacing pattern to identify the asymptotic mechanism.

Key structural observation for N even
--------------------------------------
The saddle points of Re Tr U on {0,pi}^N cap SU(N) are classified by
the number k of eigenvalues +1 (the rest are -1).

For det U = 1: we need (-1)^{N-k} = 1, i.e., N-k even, i.e., k even.

So the allowed Phi = 2k - N for k in {0, 2, 4, ..., N}.

The Vandermonde order is: ord(k,N) = k(k-1) + (N-k)(N-k-1).
The dominant saddle (minimal ord) is the balanced split k = N/2.

Two sub-cases:
  N = 0 mod 4: k = N/2 is even, so it IS in SU(N). Phi = 0.
      => Dominant saddle is non-oscillating.
  N = 2 mod 4: k = N/2 is odd, so it is NOT in SU(N).
      => Closest-to-zero saddles have k = N/2 +/- 1 (both even), Phi = +/- 2.
      => Two oscillating dominant saddles, like N-odd case but with Phi_0 = 2.

Prediction:
  N = 0 mod 4: zeros come from sub-dominant interference, spacing NOT pi/2
  N = 2 mod 4: two-saddle interference with omega = 2*Phi_0 = 4, giving <Delta> = pi/4

Usage
-----
    python fisher_zeros_even_N.py --N 4 --kappa 1.0
    python fisher_zeros_even_N.py --N 6 --kappa 1.0
    python fisher_zeros_even_N.py --N 4 --kappa 1.0 --method weyl --n_quad 50
"""

import argparse
import numpy as np
from math import comb, pi, exp, acos, sin, cos, factorial
from itertools import combinations_with_replacement as cr


# ---------------------------------------------------------------------------
# Saddle-point analysis for N even
# ---------------------------------------------------------------------------

def saddle_point_structure(N):
    """
    Classify all saddle points on {0,pi}^N cap SU(N) for general N.

    Returns list of dicts: {k, Phi, ord, in_SUN, role}
    """
    saddles = []
    for k in range(N + 1):
        Phi = 2 * k - N
        ord_val = k * (k - 1) + (N - k) * (N - k - 1)
        in_SUN = (N - k) % 2 == 0
        saddles.append({
            'k': k, 'Phi': Phi, 'ord': ord_val, 'in_SUN': in_SUN
        })

    # Find dominant saddle(s)
    allowed = [s for s in saddles if s['in_SUN']]
    if not allowed:
        return saddles
    min_ord = min(s['ord'] for s in allowed)
    for s in allowed:
        if s['ord'] == min_ord:
            s['role'] = 'dominant'
        else:
            s['role'] = 'sub-dominant'
    return saddles


def print_saddle_analysis(N):
    """Print saddle-point analysis for SU(N)."""
    saddles = saddle_point_structure(N)
    print(f"\n  Saddle-Point Structure for SU({N})  [N {'even' if N%2==0 else 'odd'}]")
    print("  " + "-" * 55)
    print(f"  {'k':>3}  {'Phi':>5}  {'ord':>5}  {'in SU(N)':>8}  {'role':>12}")
    print("  " + "-" * 55)

    for s in saddles:
        if s['in_SUN']:
            role = s.get('role', '')
            mark = ' <--' if role == 'dominant' else ''
            print(f"  {s['k']:>3}  {s['Phi']:>5}  {s['ord']:>5}  {'YES':>8}  {role:>12}{mark}")
        else:
            print(f"  {s['k']:>3}  {s['Phi']:>5}  {s['ord']:>5}  {'no':>8}")

    dominant = [s for s in saddles if s.get('role') == 'dominant']
    Phi_vals = sorted(set(abs(s['Phi']) for s in dominant))
    if len(dominant) == 1 and dominant[0]['Phi'] == 0:
        print(f"\n  ==> Single non-oscillating dominant saddle (Phi=0).")
        print(f"      Zeros come from sub-dominant interference.")
    elif len(dominant) == 2 and Phi_vals == [abs(dominant[0]['Phi'])]:
        Phi0 = abs(dominant[0]['Phi'])
        omega = 2 * Phi0
        pred_spacing = pi / omega
        print(f"\n  ==> Two oscillating dominant saddles with Phi = +/-{Phi0}.")
        print(f"      Predicted: omega = 2*{Phi0} = {omega}, <Delta> = pi/{omega} = {pred_spacing:.5f}")
    print()


# ---------------------------------------------------------------------------
# SU(N) helpers (from spacing_table.py)
# ---------------------------------------------------------------------------

def h_p_exact(p, eigenvalues):
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


def h_p_newton(p, eigs):
    """h_p via Newton's identity. eigs: 1D array of complex eigenvalues."""
    n = len(eigs)
    if p == 0:
        return complex(1.0)
    psums = [sum(e**k for e in eigs) for k in range(1, p + 1)]
    h = [complex(0)] * (p + 1)
    h[0] = 1.0
    for k in range(1, p + 1):
        h[k] = sum(psums[j] * h[k - 1 - j] for j in range(k)) / k
    return h[p]


def casimir(p, N):
    return p * (p + N) / float(N)


def dim_rep(p, N):
    return comb(p + N - 1, N - 1)


# ---------------------------------------------------------------------------
# Method 1: Exact Weyl integration for SU(4)
# ---------------------------------------------------------------------------

def compute_Z2P_weyl_su4(kappa, y, n_reps=15, n_quad=50):
    """
    Compute Z_{2P}^{SU(4)}(kappa+iy) via exact Weyl integration.

    Uses Gauss-Legendre quadrature on the 3D torus [0,2pi]^3
    with the SU(4) constraint theta_4 = -(theta_1+theta_2+theta_3).

    Returns Z (complex).
    """
    N = 4
    s = complex(kappa, y)

    # Gauss-Legendre nodes on [0, 2pi]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)
    nodes = np.pi * (gl_nodes + 1)  # [0, 2pi]
    weights = np.pi * gl_weights

    # 3D tensor product grid
    t1, t2, t3 = np.meshgrid(nodes, nodes, nodes, indexing='ij')
    w1, w2, w3 = np.meshgrid(weights, weights, weights, indexing='ij')

    t1 = t1.ravel()
    t2 = t2.ravel()
    t3 = t3.ravel()
    W = (w1 * w2 * w3).ravel()
    n_pts = len(t1)

    t4 = -(t1 + t2 + t3)  # SU(4) constraint (no mod needed for trig)

    # Eigenvalues z_j = exp(i theta_j)
    z = np.stack([np.exp(1j * t1), np.exp(1j * t2),
                  np.exp(1j * t3), np.exp(1j * t4)], axis=1)  # (n_pts, 4)

    # Vandermonde squared: prod_{j<k} |z_j - z_k|^2
    V2 = np.ones(n_pts)
    for j in range(N):
        for k in range(j + 1, N):
            V2 *= np.abs(z[:, j] - z[:, k]) ** 2

    # Phase: Phi = Re Tr U = sum cos(theta_j)
    Phi = np.cos(t1) + np.cos(t2) + np.cos(t3) + np.cos(t4)
    exp_sPhi = np.exp(s * Phi)

    # Measure (unnormalized)
    measure = W * V2 / (2 * np.pi) ** 3

    # Normalization: integral of 1 over SU(4) must give N! = 24
    # (the Weyl formula overcounts by N!)
    norm = np.sum(measure).real

    # Compute A_p for each p, then Z_{2P} = sum d_p * A_p^2
    # Power sums for Newton's identity (vectorized)
    Z = 0j
    for p in range(n_reps):
        d = dim_rep(p, N)
        # h_p via Newton's identity (vectorized)
        if p == 0:
            hp = np.ones(n_pts, dtype=complex)
        else:
            psums = np.array([np.sum(z ** k, axis=1) for k in range(1, p + 1)])
            h = np.zeros((p + 1, n_pts), dtype=complex)
            h[0] = 1.0
            for kk in range(1, p + 1):
                h[kk] = sum(psums[j] * h[kk - 1 - j] for j in range(kk)) / kk
            hp = h[p]

        Ap = np.sum(hp * exp_sPhi * measure) / norm
        Z += d * Ap ** 2

    return Z


# ---------------------------------------------------------------------------
# Method 2: Monte Carlo (any N)
# ---------------------------------------------------------------------------

def sample_haar(N, n_samples, rng):
    """Sample from Haar measure on SU(N) via QR decomposition."""
    Z = rng.standard_normal((n_samples, N, N)) + 1j * rng.standard_normal((n_samples, N, N))
    Q, R = np.linalg.qr(Z)
    D = np.diagonal(R, axis1=1, axis2=2)
    D = D / np.abs(D)
    Q = Q * D[:, np.newaxis, :]
    dets = np.linalg.det(Q)
    Q = Q / (dets ** (1.0 / N))[:, np.newaxis, np.newaxis]
    return Q


def h_p_vec(p, eigs):
    """h_p via Newton's identity, vectorized. eigs: (n_samples, N) complex."""
    n_samples = eigs.shape[0]
    if p == 0:
        return np.ones(n_samples, dtype=complex)
    psums = np.array([np.sum(eigs ** k, axis=1) for k in range(1, p + 1)])
    h = np.zeros((p + 1, n_samples), dtype=complex)
    h[0] = 1.0
    for k in range(1, p + 1):
        h[k] = sum(psums[j] * h[k - 1 - j] for j in range(k)) / k
    return h[p]


def compute_Z2P_mc(N, kappa, y, n_reps=10, n_MC=100000, seed=42):
    """Compute Z_{2P}^{SU(N)}(kappa+iy) via Monte Carlo."""
    rng = np.random.default_rng(seed)
    U = sample_haar(N, n_MC, rng)
    eigs = np.linalg.eigvals(U)
    Phi = np.sum(eigs.real, axis=1)
    exp_sPhi = np.exp(complex(kappa, y) * Phi)

    Z = 0j
    for p in range(n_reps):
        d = dim_rep(p, N)
        chi = h_p_vec(p, eigs)
        Ap = np.mean(chi * exp_sPhi)
        Z += d * Ap ** 2
    return Z


def precompute_mc_data(N, n_reps, n_MC, seed=42):
    """
    Pre-compute Haar samples, eigenvalues, and characters.
    Reuse for all y values (much faster than re-sampling each time).
    """
    rng = np.random.default_rng(seed)
    U = sample_haar(N, n_MC, rng)
    eigs = np.linalg.eigvals(U)
    Phi = np.sum(eigs.real, axis=1)  # Re Tr U

    # Pre-compute characters for all reps
    chars = []
    dims = []
    for p in range(n_reps):
        d = dim_rep(p, N)
        chi = h_p_vec(p, eigs)
        chars.append(chi)
        dims.append(d)

    return {'eigs': eigs, 'Phi': Phi, 'chars': chars, 'dims': dims}


def Z2P_from_precomputed(data, kappa, y):
    """Compute Z_{2P}(kappa+iy) using pre-computed MC data."""
    s = complex(kappa, y)
    exp_sPhi = np.exp(s * data['Phi'])

    Z = 0j
    for p in range(len(data['chars'])):
        Ap = np.mean(data['chars'][p] * exp_sPhi)
        Z += data['dims'][p] * Ap ** 2
    return Z


# ---------------------------------------------------------------------------
# Zero finder
# ---------------------------------------------------------------------------

def find_zeros_scan(N, kappa, y_min, y_max, n_scan, n_reps, method='mc',
                    n_MC=100000, n_quad=50):
    """
    Find Fisher zeros by scanning Re Z for sign changes, then bisecting.
    Returns list of (y_zero, gap).
    """
    y_vals = np.linspace(y_min, y_max, n_scan)

    print(f"  Scanning y in [{y_min:.1f}, {y_max:.1f}] with {n_scan} points...", flush=True)

    if method == 'weyl' and N == 4:
        ReZ = np.zeros(n_scan)
        for i, y in enumerate(y_vals):
            Z = compute_Z2P_weyl_su4(kappa, y, n_reps, n_quad)
            ReZ[i] = Z.real
            if (i + 1) % 20 == 0:
                print(f"    [{i+1}/{n_scan}] y={y:.3f}, Re Z = {Z.real:.6e}", flush=True)

        # For bisection, also use Weyl
        def eval_Z(y_val):
            return compute_Z2P_weyl_su4(kappa, y_val, n_reps, n_quad)
    else:
        # Pre-compute MC data (shared across all y values)
        print(f"  Pre-computing MC data: N_MC={n_MC}, n_reps={n_reps}...", flush=True)
        mc_data = precompute_mc_data(N, n_reps, n_MC, seed=42)
        print(f"  Done. Scanning...", flush=True)

        ReZ = np.zeros(n_scan)
        for i, y in enumerate(y_vals):
            Z = Z2P_from_precomputed(mc_data, kappa, y)
            ReZ[i] = Z.real
            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{n_scan}] y={y:.3f}, Re Z = {Z.real:.6e}", flush=True)

        def eval_Z(y_val):
            return Z2P_from_precomputed(mc_data, kappa, y_val)

    # Find sign changes
    brackets = []
    for i in range(len(ReZ) - 1):
        if ReZ[i] * ReZ[i + 1] < 0:
            brackets.append((y_vals[i], y_vals[i + 1]))

    # Bisection
    zeros = []
    for lo, hi in brackets:
        for _ in range(25):
            mid = (lo + hi) / 2
            Z_mid = eval_Z(mid)
            Z_lo = eval_Z(lo)
            if Z_mid.real * Z_lo.real < 0:
                hi = mid
            else:
                lo = mid
        zeros.append((lo + hi) / 2)

    # Compute gaps
    results = []
    for i, y in enumerate(zeros):
        gap = y - zeros[i - 1] if i > 0 else None
        results.append({'k': i, 'y': y, 'gap': gap})

    return results


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_zeros(N, kappa, zeros):
    """Analyze the spacing pattern of Fisher zeros."""
    print(f"\n  Fisher Zeros of Z_{{2P}}^{{SU({N})}}(kappa+iy), kappa={kappa}")
    print("  " + "=" * 60)
    print(f"  {'k':>3}  {'y_k':>12}  {'Delta_k':>10}  {'type':>8}")
    print("  " + "-" * 40)

    gaps = []
    for z in zeros:
        gap_str = f"{z['gap']:.5f}" if z['gap'] is not None else "—"
        print(f"  {z['k']:>3}  {z['y']:>12.5f}  {gap_str:>10}")
        if z['gap'] is not None:
            gaps.append(z['gap'])

    if len(gaps) < 2:
        print("\n  Too few zeros found for spacing analysis.")
        return

    mean_gap = sum(gaps) / len(gaps)
    print(f"\n  Number of zeros found: {len(zeros)}")
    print(f"  Number of gaps: {len(gaps)}")
    print(f"  Mean gap = {mean_gap:.5f}")
    print(f"  pi/2     = {pi/2:.5f}")
    print(f"  pi/4     = {pi/4:.5f}")
    print(f"  pi/3     = {pi/3:.5f}")

    # Check pair sums (alternating pattern)
    if len(gaps) >= 2:
        pair_sums = [gaps[i] + gaps[i + 1] for i in range(0, len(gaps) - 1, 2)]
        if pair_sums:
            mean_pair = sum(pair_sums) / len(pair_sums)
            print(f"\n  Consecutive pair sums: {[f'{s:.5f}' for s in pair_sums]}")
            print(f"  Mean pair sum = {mean_pair:.5f}")
            print(f"  pi     = {pi:.5f}")
            print(f"  pi/2   = {pi/2:.5f}")

    # Saddle-point prediction
    saddles = saddle_point_structure(N)
    dominant = [s for s in saddles if s.get('role') == 'dominant']
    if len(dominant) == 2 and dominant[0]['Phi'] != 0:
        Phi0 = abs(dominant[0]['Phi'])
        omega = 2 * Phi0
        pred = pi / omega
        print(f"\n  Saddle-point prediction: Phi_0 = {Phi0}, omega = {omega}, <Delta> = pi/{omega} = {pred:.5f}")
        print(f"  Deviation from prediction: |mean_gap - pred| = {abs(mean_gap - pred):.5f}")
    elif len(dominant) == 1 and dominant[0]['Phi'] == 0:
        print(f"\n  Dominant saddle has Phi=0 (non-oscillating).")
        print(f"  Zeros come from sub-dominant interference — pattern to be determined.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fisher zeros of Z_{2P}^{SU(N)} for N even. "
                    "Author: Grzegorz Olbryk."
    )
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--kappa', type=float, default=1.0)
    parser.add_argument('--y_min', type=float, default=0.3)
    parser.add_argument('--y_max', type=float, default=30.0)
    parser.add_argument('--n_scan', type=int, default=200)
    parser.add_argument('--n_reps', type=int, default=12)
    parser.add_argument('--n_MC', type=int, default=100000)
    parser.add_argument('--n_quad', type=int, default=50,
                        help='Quadrature points per dim for Weyl method (N=4 only)')
    parser.add_argument('--method', choices=['mc', 'weyl'], default='mc',
                        help='Computation method: mc or weyl (weyl only for N=4)')
    args = parser.parse_args()

    print()
    print("=" * 70)
    print(f"  Fisher Zeros for N Even: SU({args.N}), kappa={args.kappa}")
    print(f"  Author: Grzegorz Olbryk")
    print("=" * 70)

    # Saddle-point analysis
    print_saddle_analysis(args.N)

    # Compute zeros
    zeros = find_zeros_scan(
        args.N, args.kappa, args.y_min, args.y_max, args.n_scan,
        args.n_reps, args.method, args.n_MC, args.n_quad
    )

    # Analyze
    analyze_zeros(args.N, args.kappa, zeros)

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
