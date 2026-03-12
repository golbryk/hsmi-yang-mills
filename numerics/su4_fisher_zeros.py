"""
SU(4) Exact Fisher Zero Search via Weyl Quadrature
====================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 4b — N ≡ 0 mod 4 exact numerics

Computes the n-plaquette SU(N) partition function EXACTLY:

    Z_{nP}(s) = Σ_p d_p [A_p(s)]^n

where A_p(s) = ∫_{SU(N)} h_p(U) exp(s Re Tr U) dU  (Wilson action).

Uses (N-1)-dimensional Weyl quadrature (trapezoidal rule on the torus),
which converges exponentially for smooth periodic integrands.

Key question: Does Z_{2P}^{SU(4)}(κ + iy) have Fisher zeros?
RESULT_006 predicts: at most finitely many, confined to y < y_crit ≈ 3.
"""

import numpy as np
from math import comb, pi, acos


# ---------------------------------------------------------------------------
# SU(N) representation theory
# ---------------------------------------------------------------------------

def dim_rep(p, N):
    """Dimension of SU(N) symmetric rep (p,0,...,0)."""
    return comb(p + N - 1, N - 1)


# ---------------------------------------------------------------------------
# Weyl quadrature for A_p(s)
# ---------------------------------------------------------------------------

def h_p_vec(p, z):
    """Complete homogeneous symmetric polynomial h_p(z_1,...,z_N).

    Uses Newton's identity: h_k = (1/k) Σ_{j=1}^k p_j h_{k-j}
    where p_j = Σ z_i^j (power sums).

    Parameters
    ----------
    p : int
    z : array, shape (n_pts, N) — complex eigenvalues

    Returns
    -------
    h_p : array, shape (n_pts,) — complex values
    """
    n_pts = z.shape[0]
    if p == 0:
        return np.ones(n_pts, dtype=complex)
    psums = np.array([np.sum(z ** k, axis=1) for k in range(1, p + 1)])
    h = np.zeros((p + 1, n_pts), dtype=complex)
    h[0] = 1.0
    for k in range(1, p + 1):
        h[k] = sum(psums[j] * h[k - 1 - j] for j in range(k)) / k
    return h[p]


def build_weyl_grid(N, n_quad):
    """Build the Weyl integration grid for SU(N).

    Returns
    -------
    z     : (n_pts, N) complex — eigenvalues e^{iθ_j}
    Phi   : (n_pts,) float — Re Tr U = Σ cos θ_j
    measure : (n_pts,) float — Vandermonde × volume element / normalization
    """
    dim = N - 1
    nodes = np.linspace(0, 2 * np.pi, n_quad, endpoint=False)
    w = (2 * np.pi / n_quad) ** dim

    grids = np.meshgrid(*([nodes] * dim), indexing='ij')
    theta = np.stack([g.ravel() for g in grids], axis=1)  # (n_pts, dim)
    n_pts = theta.shape[0]

    # SU(N) constraint: θ_N = -Σ θ_j
    theta_N = -np.sum(theta, axis=1, keepdims=True)
    theta_all = np.concatenate([theta, theta_N], axis=1)

    z = np.exp(1j * theta_all)

    # Vandermonde squared: Π_{j<k} |e^{iθ_j} - e^{iθ_k}|²
    V2 = np.ones(n_pts)
    for j in range(N):
        for k in range(j + 1, N):
            V2 *= np.abs(z[:, j] - z[:, k]) ** 2

    Phi = np.sum(np.cos(theta_all), axis=1)
    measure = w * V2 / (2 * np.pi) ** dim
    norm = np.sum(measure).real
    measure = measure / norm

    return z, Phi, measure


def precompute_Ap(z, Phi, measure, kappa, y_vals, n_reps):
    """Compute A_p(κ + iy) for all y values and all reps p.

    A_p(s) = ∫ h_p(U) exp(s Re Tr U) dU

    Returns
    -------
    Ap : (n_y, n_reps) complex
    """
    n_y = len(y_vals)
    hp_list = [h_p_vec(p, z) for p in range(n_reps)]

    exp_kappaPhi = np.exp(kappa * Phi)  # real, shape (n_pts,)

    Ap = np.zeros((n_y, n_reps), dtype=complex)
    for iy, y in enumerate(y_vals):
        exp_sPhi = exp_kappaPhi * np.exp(1j * y * Phi)
        weighted = exp_sPhi * measure
        for p in range(n_reps):
            Ap[iy, p] = np.sum(hp_list[p] * weighted)

    return Ap


def compute_Z(Ap, n_plaq, N, n_reps):
    """Compute Z_{nP} = Σ_p d_p A_p^{n_plaq}."""
    n_y = Ap.shape[0]
    Z = np.zeros(n_y, dtype=complex)
    for p in range(n_reps):
        d = dim_rep(p, N)
        Z += d * Ap[:, p] ** n_plaq
    return Z


# ---------------------------------------------------------------------------
# Zero finding
# ---------------------------------------------------------------------------

def find_sign_changes(Z_vals, y_vals):
    """Find y values where Re Z changes sign (bisection via interpolation)."""
    zeros = []
    for i in range(len(Z_vals) - 1):
        if Z_vals[i].real * Z_vals[i + 1].real < 0:
            # Linear interpolation for the zero
            r0 = Z_vals[i].real
            r1 = Z_vals[i + 1].real
            frac = r0 / (r0 - r1)
            y_zero = y_vals[i] + frac * (y_vals[i + 1] - y_vals[i])
            # Interpolated |Z| at zero
            Z_at_zero = Z_vals[i] * (1 - frac) + Z_vals[i + 1] * frac
            zeros.append({'y': y_zero, 'absZ': abs(Z_at_zero),
                          'ImZ': Z_at_zero.imag})
    return zeros


def find_absZ_minima(Z_vals, y_vals, threshold_frac=0.1):
    """Find local minima of |Z| below threshold_frac × Z(0)."""
    absZ = np.abs(Z_vals)
    threshold = absZ[0] * threshold_frac

    minima = []
    for i in range(1, len(absZ) - 1):
        if absZ[i] < absZ[i - 1] and absZ[i] < absZ[i + 1]:
            if absZ[i] < threshold:
                minima.append({'y': y_vals[i], 'absZ': absZ[i],
                               'ratio': absZ[i] / absZ[0]})

    minima.sort(key=lambda x: x['absZ'])
    return minima


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

def convergence_check(N, kappa, y_test, n_reps):
    """Check convergence of Weyl quadrature."""
    print(f"\n  Convergence check: SU({N}), κ={kappa}, y={y_test}")
    for nq in [20, 30, 40, 50]:
        z, Phi, measure = build_weyl_grid(N, nq)
        Ap = precompute_Ap(z, Phi, measure, kappa, [y_test], n_reps)
        Z = compute_Z(Ap, 2, N, n_reps)
        print(f"    n_quad={nq:3d} ({nq**(N-1):>8d} pts): "
              f"Re Z = {Z[0].real:+.8e}  Im Z = {Z[0].imag:+.8e}  "
              f"|Z| = {abs(Z[0]):.8e}")


# ---------------------------------------------------------------------------
# Full analysis for one (N, n_plaq, κ) combination
# ---------------------------------------------------------------------------

def analyze(N, n_plaq, kappa, y_vals, Ap, label=""):
    """Analyze Z_{nP}(κ+iy) for zeros."""
    n_reps = Ap.shape[1]
    Z = compute_Z(Ap, n_plaq, N, n_reps)
    Z0 = abs(Z[0])

    print(f"\n  {label}SU({N}), n={n_plaq}, κ={kappa}:")
    print(f"    Z(κ) = {Z[0].real:.8e} + {Z[0].imag:.2e}i  "
          f"|Z(κ)| = {Z0:.8e}")

    # Sign changes of Re Z
    sign_zeros = find_sign_changes(Z, y_vals)

    # |Z| minima
    minima = find_absZ_minima(Z, y_vals, threshold_frac=0.5)

    if sign_zeros:
        print(f"    Re Z sign changes: {len(sign_zeros)}")
        for i, sz in enumerate(sign_zeros[:20]):
            print(f"      y = {sz['y']:10.6f}  |Z| = {sz['absZ']:.4e}  "
                  f"Im Z = {sz['ImZ']:.4e}  "
                  f"|Z|/Z₀ = {sz['absZ']/Z0:.4e}")

        # Measure spacings
        if len(sign_zeros) >= 2:
            ys = [sz['y'] for sz in sign_zeros]
            gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
            avg = sum(gaps) / len(gaps)
            print(f"\n    Average Re Z crossing gap: {avg:.6f}")
    else:
        print(f"    No Re Z sign changes found.")

    if minima:
        n_show = min(10, len(minima))
        print(f"\n    Deepest |Z| minima ({n_show} of {len(minima)}):")
        for m in minima[:n_show]:
            print(f"      y = {m['y']:10.6f}  |Z| = {m['absZ']:.4e}  "
                  f"|Z|/Z₀ = {m['ratio']:.6e}")
    else:
        # Global minimum
        absZ = np.abs(Z)
        idx = np.argmin(absZ[1:]) + 1
        print(f"    Global min |Z| = {absZ[idx]:.6e} at y = {y_vals[idx]:.4f}")
        print(f"    Min |Z|/Z₀ = {absZ[idx]/Z0:.6e}")

    return sign_zeros, minima, Z


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_reps = 10  # representations to include
    n_quad = 40  # quadrature points per dimension

    print()
    print("=" * 90)
    print("  Fisher Zero Exact Search via Weyl Quadrature (Wilson Action)")
    print("  Author: Grzegorz Olbryk  |  March 2026  |  Task 4b")
    print("=" * 90)

    # ==================================================================
    # Part 0: Convergence
    # ==================================================================
    print("\n  PART 0: Convergence Check")
    print("  " + "-" * 70)
    convergence_check(3, 1.0, 2.0, n_reps)
    convergence_check(4, 1.0, 2.0, n_reps)

    # ==================================================================
    # Part 1: SU(3) verification
    # ==================================================================
    print("\n\n  PART 1: SU(3) Verification (N odd, |Φ₀| = 1)")
    print("  " + "-" * 70)

    N = 3
    y_vals = np.linspace(0.1, 15.0, 500)

    print(f"  Building Weyl grid: SU(3), n_quad={n_quad} "
          f"({n_quad**(N-1)} pts)...", flush=True)
    z3, Phi3, meas3 = build_weyl_grid(N, n_quad)

    for kappa in [0.5, 1.0]:
        print(f"\n  Precomputing A_p for κ={kappa}...", flush=True)
        Ap3 = precompute_Ap(z3, Phi3, meas3, kappa, y_vals, n_reps)

        sz, _, _ = analyze(N, 2, kappa, y_vals, Ap3, label="[verify] ")

        if len(sz) >= 4:
            ys = [s['y'] for s in sz]
            gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
            avg = sum(gaps) / len(gaps)
            expected = pi / 2
            print(f"    Expected ⟨Δ⟩ = π/2 = {expected:.6f}")
            print(f"    Ratio: {avg/expected:.6f}")

    # ==================================================================
    # Part 2: SU(4) (N ≡ 0 mod 4)
    # ==================================================================
    print("\n\n  PART 2: SU(4) — N ≡ 0 mod 4 (|Φ₀| = 0)")
    print("  " + "-" * 70)

    N = 4
    y_vals_4 = np.linspace(0.05, 12.0, 600)

    print(f"  Building Weyl grid: SU(4), n_quad={n_quad} "
          f"({n_quad**(N-1)} pts)...", flush=True)
    z4, Phi4, meas4 = build_weyl_grid(N, n_quad)

    for kappa in [0.5, 1.0, 2.0]:
        print(f"\n  Precomputing A_p for κ={kappa}...", flush=True)
        Ap4 = precompute_Ap(z4, Phi4, meas4, kappa, y_vals_4, n_reps)

        for n_plaq in [2, 3]:
            analyze(N, n_plaq, kappa, y_vals_4, Ap4)

    # ==================================================================
    # Part 3: SU(6) for comparison (N ≡ 2 mod 4, |Φ₀| = 2)
    # ==================================================================
    print("\n\n  PART 3: SU(6) Comparison — N ≡ 2 mod 4 (|Φ₀| = 2)")
    print("  " + "-" * 70)

    N = 6
    n_quad_6 = 16  # 16^5 = 1,048,576 — feasible but larger
    y_vals_6 = np.linspace(0.1, 8.0, 300)

    print(f"  Building Weyl grid: SU(6), n_quad={n_quad_6} "
          f"({n_quad_6**(N-1)} pts)...", flush=True)
    z6, Phi6, meas6 = build_weyl_grid(N, n_quad_6)

    kappa = 1.0
    print(f"  Precomputing A_p for κ={kappa}...", flush=True)
    Ap6 = precompute_Ap(z6, Phi6, meas6, kappa, y_vals_6, n_reps)

    sz, _, _ = analyze(N, 2, kappa, y_vals_6, Ap6, label="[verify] ")
    if len(sz) >= 4:
        ys = [s['y'] for s in sz]
        gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        avg = sum(gaps) / len(gaps)
        expected = pi / 4
        print(f"    Expected ⟨Δ⟩ = π/4 = {expected:.6f}")
        print(f"    Ratio: {avg/expected:.6f}")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n\n  SUMMARY")
    print("  " + "-" * 70)
    print("  Method: Exact Weyl quadrature of A_p(s) = ∫ h_p(U) exp(s Re Tr U) dU")
    print("  Z_{nP} = Σ_p d_p A_p^n computed from exact A_p values.")
    print()
    print("  SU(3): Verification of known Fisher zeros with spacing π/2.")
    print("  SU(4): Search for Fisher zeros in N ≡ 0 mod 4 case.")
    print("  SU(6): Verification of known Fisher zeros with spacing π/4.")
    print("=" * 90)


if __name__ == '__main__':
    main()
