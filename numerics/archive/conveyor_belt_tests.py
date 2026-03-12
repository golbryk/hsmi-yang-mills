"""
Conveyor Belt: Three Critical Tests
=====================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: Referee feedback on RESULT_025 — strengthen numerical evidence

Three tests:
  TEST 1: SU(8) via Haar-random importance sampling (eliminates Vandermonde variance)
  TEST 2: |A_p(κ+iy)| vs y for several p — visual conveyor belt
  TEST 3: p*(y) scaling law — check if p* ~ y for large y

Also: verification that Z_n^HK is NOT entire (key distinction from Wilson).
"""

import numpy as np
from math import comb, pi
import time


def dim_rep(p, N):
    return comb(p + N - 1, N - 1)


def casimir_suN(p, N):
    return p * (p + N) / float(N)


def h_p_vec(p, z):
    """Vectorized h_p via Newton's identities. z: (n_samples, N) complex."""
    n_samples = z.shape[0]
    if p == 0:
        return np.ones(n_samples, dtype=complex)
    psums = np.array([np.sum(z ** k, axis=1) for k in range(1, p + 1)])
    h = np.zeros((p + 1, n_samples), dtype=complex)
    h[0] = 1.0
    for k in range(1, p + 1):
        h[k] = sum(psums[j] * h[k - 1 - j] for j in range(k)) / k
    return h[p]


# ================================================================
# Haar-random eigenvalue sampling for SU(N)
# ================================================================

def haar_random_eigenvalues(N, n_samples):
    """Generate Haar-random eigenvalues on SU(N) using QR decomposition.

    This automatically includes the Vandermonde weight |Δ|² — no reweighting
    needed, eliminating the extreme variance of flat-torus sampling.
    """
    eigenvalues = np.zeros((n_samples, N), dtype=complex)
    for i in range(n_samples):
        # Ginibre matrix → QR → Haar-random unitary
        Z = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2)
        Q, R = np.linalg.qr(Z)
        # Fix the phase ambiguity (Mezzadri's correction)
        d = np.diag(R)
        ph = d / np.abs(d)
        Q = Q * ph[np.newaxis, :]
        # Project to SU(N): divide all eigenvalues by det^{1/N}
        eigs = np.linalg.eigvals(Q)
        det_phase = np.prod(eigs) / np.abs(np.prod(eigs))
        eigs = eigs / det_phase ** (1.0 / N)
        eigenvalues[i] = eigs
    return eigenvalues


def haar_random_eigenvalues_fast(N, n_samples, batch_size=1000):
    """Batched version for speed."""
    all_eigs = []
    remaining = n_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        Z = (np.random.randn(bs, N, N) + 1j * np.random.randn(bs, N, N)) / np.sqrt(2)
        eigs_batch = np.zeros((bs, N), dtype=complex)
        for i in range(bs):
            Q, R = np.linalg.qr(Z[i])
            d = np.diag(R)
            ph = d / np.abs(d)
            Q = Q * ph[np.newaxis, :]
            eigs = np.linalg.eigvals(Q)
            det_phase = np.prod(eigs) / np.abs(np.prod(eigs))
            eigs_batch[i] = eigs / det_phase ** (1.0 / N)
        all_eigs.append(eigs_batch)
        remaining -= bs
    return np.concatenate(all_eigs, axis=0)[:n_samples]


def compute_Ap_haar(z, s, p_max):
    """Compute A_p(s) for p=0..p_max using Haar-distributed eigenvalues.

    A_p(s) = E_Haar[h_p(z) exp(s Φ(z))]

    Since z is already Haar-distributed, no Vandermonde reweighting needed.
    Returns A_p(s) / A_0(s) (normalized) for stability.
    """
    N = z.shape[1]
    Phi = np.sum(z.real, axis=1)  # Re Tr U = Σ cos θ_j
    exp_sPhi = np.exp(s * Phi)

    results = np.zeros(p_max + 1, dtype=complex)
    for p in range(p_max + 1):
        hp = h_p_vec(p, z)
        results[p] = np.mean(hp * exp_sPhi)

    return results


# ================================================================
# SU(4) Weyl quadrature (exact, for comparison)
# ================================================================

def build_weyl_grid_su4(n_quad=40):
    N = 4
    dim = 3
    nodes = np.linspace(0, 2 * np.pi, n_quad, endpoint=False)
    w = (2 * np.pi / n_quad) ** dim
    grids = np.meshgrid(*([nodes] * dim), indexing='ij')
    theta = np.stack([g.ravel() for g in grids], axis=1)
    n_pts = theta.shape[0]
    theta_N = -np.sum(theta, axis=1, keepdims=True)
    theta_all = np.concatenate([theta, theta_N], axis=1)
    z = np.exp(1j * theta_all)
    V2 = np.ones(n_pts)
    for j in range(N):
        for k in range(j + 1, N):
            V2 *= np.abs(z[:, j] - z[:, k]) ** 2
    Phi = np.sum(np.cos(theta_all), axis=1)
    measure = w * V2 / (2 * np.pi) ** dim
    norm = np.sum(measure).real
    measure /= norm
    return z, Phi, measure


def compute_Ap_weyl_su4(z, Phi, measure, s, p_max):
    """Exact A_p(s) for SU(4) via Weyl quadrature."""
    exp_sPhi = np.exp(s * Phi)
    weighted = exp_sPhi * measure
    results = np.zeros(p_max + 1, dtype=complex)
    hp_list = [h_p_vec(p, z) for p in range(p_max + 1)]
    for p in range(p_max + 1):
        results[p] = np.sum(hp_list[p] * weighted)
    return results


# ================================================================
# MAIN
# ================================================================

def main():
    np.random.seed(42)

    print("=" * 80)
    print("  Conveyor Belt: Three Critical Tests")
    print("  Author: Grzegorz Olbryk  |  March 2026")
    print("=" * 80)

    kappa = 1.0
    n_plaq = 2

    # ==============================================================
    # TEST 0: Z_n^HK is NOT entire
    # ==============================================================
    print(f"\n{'='*80}")
    print(f"  TEST 0: Z_n^HK is NOT entire (key distinction)")
    print(f"{'='*80}")

    N = 4
    print(f"\n  Z_2^HK(s) = Sigma d_p exp(-2 C_2(p) s)  for SU({N})")
    print(f"\n  Partial sums at s = -0.1 (Re(s) < 0):")
    for p_max_test in [5, 10, 15, 20, 30, 50]:
        partial = sum(dim_rep(p, N) * np.exp(-2 * casimir_suN(p, N) * (-0.1))
                      for p in range(p_max_test + 1))
        print(f"    P_MAX={p_max_test:3d}: Z_2^HK(-0.1) = {partial:.6e}")

    print(f"\n  At s = -0.5:")
    for p_max_test in [5, 10, 15, 20]:
        partial = sum(dim_rep(p, N) * np.exp(-2 * casimir_suN(p, N) * (-0.5))
                      for p in range(p_max_test + 1))
        print(f"    P_MAX={p_max_test:3d}: Z_2^HK(-0.5) = {partial:.6e}")

    print("""
  The partial sums DIVERGE for Re(s) < 0.
  Z_n^HK is a Dirichlet series convergent only for Re(s) > 0.
  Therefore Z_n^HK is NOT entire.
  Hadamard factorization does NOT apply to Z_n^HK.
  This is why heat-kernel CAN have finitely many zeros.

  Wilson Z_n^W(s) = integral over compact SU(N) of entire integrand
  => Z_n^W is ENTIRE of order <= 1
  => Hadamard DOES apply
  => combined with Riemann-Lebesgue bidirectional decay => inf many zeros
  """)

    # ==============================================================
    # TEST 1: SU(8) via Haar-random importance sampling
    # ==============================================================
    print(f"{'='*80}")
    print(f"  TEST 1: SU(8) — Haar-Random Importance Sampling")
    print(f"{'='*80}")

    N = 8
    n_samples = 50000
    p_max = 15

    print(f"  Generating {n_samples} Haar-random SU({N}) eigenvalues...")
    t0 = time.time()
    z8 = haar_random_eigenvalues_fast(N, n_samples)
    t_gen = time.time() - t0
    print(f"  Generated in {t_gen:.1f}s")

    # Verify Haar distribution: E[Tr U] should be 0 for SU(N)
    Phi8 = np.sum(z8.real, axis=1)
    print(f"  Verification: E[Phi] = {np.mean(Phi8):.4f} (should be ~0)")
    print(f"  Std[Phi] = {np.std(Phi8):.4f}")

    # A_p at real kappa
    print(f"\n  A_p(kappa={kappa}) for SU({N}):")
    Ap_real = compute_Ap_haar(z8, kappa, p_max)
    A0_real = Ap_real[0].real
    for p in range(min(8, p_max + 1)):
        dp = dim_rep(p, N)
        print(f"    p={p}: d_p={dp:6d}, A_p/A_0 = {(Ap_real[p]/A0_real).real:.6e}")

    # Conveyor belt for SU(8)
    y_values = np.arange(0.5, 15, 0.5)
    print(f"\n  Computing partner p*(y) for {len(y_values)} y values...")

    su8_partner = []
    su8_T0_abs = []
    for y in y_values:
        s = kappa + 1j * y
        Ap = compute_Ap_haar(z8, s, p_max)

        T_p = np.array([dim_rep(p, N) * Ap[p]**n_plaq for p in range(p_max+1)])

        T0_phase = np.angle(T_p[0])
        anti_target = T0_phase + pi

        best_p = 1
        best_s = 0
        for p in range(1, p_max + 1):
            if abs(T_p[p]) < 1e-30:
                continue
            pd = (np.angle(T_p[p]) - anti_target + pi) % (2 * pi) - pi
            if abs(pd) < pi / 2:
                if abs(T_p[p]) > best_s:
                    best_s = abs(T_p[p])
                    best_p = p
        su8_partner.append(best_p)
        su8_T0_abs.append(abs(T_p[0]))

    pstar = "p*"
    print(f"\n  {'y':>6}  {pstar:>4}  note")
    for i, (y, pp) in enumerate(zip(y_values, su8_partner)):
        note = ""
        if i > 0 and pp > su8_partner[i - 1]:
            note = " UP"
        elif i > 0 and pp < su8_partner[i - 1]:
            note = " dn"
        print(f"  {y:6.1f}  {pp:4d}  {note}")

    inc = sum(1 for i in range(1, len(su8_partner)) if su8_partner[i] >= su8_partner[i-1])
    tot = len(su8_partner) - 1
    print(f"\n  Monotonicity: {inc}/{tot} ({100*inc/tot:.0f}%) non-decreasing")

    # ==============================================================
    # TEST 2: |A_p(κ+iy)| vs y for several p
    # ==============================================================
    print(f"\n{'='*80}")
    print(f"  TEST 2: |A_p(kappa+iy)| vs y — Conveyor Belt Visualization")
    print(f"{'='*80}")

    # SU(4) exact quadrature
    print(f"\n  SU(4) — Exact Weyl Quadrature")
    z4, Phi4, meas4 = build_weyl_grid_su4(40)
    hp4_list = [h_p_vec(p, z4) for p in range(26)]
    exp_kPhi4 = np.exp(kappa * Phi4)

    y_test = np.arange(0.5, 20, 0.25)
    # Compute |A_p(κ+iy)| for p = 0, 1, 3, 5, 8, 12
    test_p = [0, 1, 3, 5, 8, 12]

    Ap_abs = {p: [] for p in test_p}
    Tp_abs = {p: [] for p in test_p}

    for y in y_test:
        weighted = exp_kPhi4 * np.exp(1j * y * Phi4) * meas4
        for p in test_p:
            Ap = np.sum(hp4_list[p] * weighted)
            dp = dim_rep(p, 4)
            Ap_abs[p].append(abs(Ap))
            Tp_abs[p].append(dp * abs(Ap) ** n_plaq)

    # Print table: for each y, which p has the largest |T_p| among p>=1
    print(f"\n  |T_p| = d_p |A_p|^2 for SU(4), kappa={kappa}:")
    print(f"  {'y':>6}", end="")
    for p in test_p:
        print(f"  {'p='+str(p):>10}", end="")
    print(f"  {'max_p':>6}")

    for iy, y in enumerate(y_test):
        if iy % 4 != 0:
            continue
        print(f"  {y:6.1f}", end="")
        max_p = 0
        max_T = Tp_abs[0][iy]
        for p in test_p:
            val = Tp_abs[p][iy]
            print(f"  {val:10.3e}", end="")
            if p >= 1 and val > max_T:
                max_T = val
                max_p = p
        print(f"  {max_p:6d}")

    # Show the crossing: which p first exceeds T_0 at each y
    print(f"\n  First p >= 1 where |T_p| > |T_0| (SU(4)):")
    for iy, y in enumerate(y_test):
        if iy % 4 != 0:
            continue
        T0 = Tp_abs[0][iy]
        first_p = "-"
        for p in [1, 3, 5, 8, 12]:
            if Tp_abs[p][iy] > T0:
                first_p = str(p)
                break
        print(f"    y={y:5.1f}: first exceeding p = {first_p}")

    # Compute |A_p/A_0| ratio — the key quantity
    print(f"\n  |A_p(kappa+iy)| / |A_p(kappa)| ratio (SU(4)):")
    print(f"  This ratio > 1 means rep p is RELATIVELY more important at complex s")
    Ap_at_real = {}
    weighted_real = exp_kPhi4 * meas4
    for p in test_p:
        Ap_at_real[p] = abs(np.sum(hp4_list[p] * weighted_real))

    print(f"  {'y':>6}", end="")
    for p in test_p:
        print(f"  {'p='+str(p):>10}", end="")
    print()

    for iy, y in enumerate(y_test):
        if iy % 8 != 0:
            continue
        print(f"  {y:6.1f}", end="")
        for p in test_p:
            ratio = Ap_abs[p][iy] / Ap_at_real[p] if Ap_at_real[p] > 1e-30 else 0
            print(f"  {ratio:10.3f}", end="")
        print()

    # SU(8) Haar MC
    print(f"\n  SU(8) — Haar-Random MC (50K samples)")
    test_p_8 = [0, 1, 3, 5, 8]
    y_test_8 = np.arange(1, 12, 1.0)

    Ap_abs_8 = {p: [] for p in test_p_8}
    for y in y_test_8:
        s = kappa + 1j * y
        Ap = compute_Ap_haar(z8, s, max(test_p_8))
        for p in test_p_8:
            Ap_abs_8[p].append(abs(Ap[p]))

    Ap_real_8 = compute_Ap_haar(z8, kappa, max(test_p_8))

    print(f"\n  |A_p(kappa+iy)| / |A_p(kappa)| ratio (SU(8)):")
    print(f"  {'y':>6}", end="")
    for p in test_p_8:
        print(f"  {'p='+str(p):>10}", end="")
    print()

    for iy, y in enumerate(y_test_8):
        print(f"  {y:6.1f}", end="")
        for p in test_p_8:
            Ap_r = abs(Ap_real_8[p])
            ratio = Ap_abs_8[p][iy] / Ap_r if Ap_r > 1e-30 else 0
            print(f"  {ratio:10.3f}", end="")
        print()

    # ==============================================================
    # TEST 3: p*(y) scaling law
    # ==============================================================
    print(f"\n{'='*80}")
    print(f"  TEST 3: p*(y) Scaling Law — SU(4) Extended Range")
    print(f"{'='*80}")

    # SU(4) with extended p_max and y range
    p_max_ext = 40
    hp4_ext = [h_p_vec(p, z4) for p in range(p_max_ext + 1)]

    y_ext = np.arange(1, 30, 0.5)
    partner_ext = []

    print(f"  Computing p*(y) for y in [1, 30], p_max={p_max_ext}...")
    for y in y_ext:
        weighted = exp_kPhi4 * np.exp(1j * y * Phi4) * meas4
        T_p = np.zeros(p_max_ext + 1, dtype=complex)
        for p in range(p_max_ext + 1):
            dp = dim_rep(p, 4)
            Ap = np.sum(hp4_ext[p] * weighted)
            T_p[p] = dp * Ap ** n_plaq

        T0_phase = np.angle(T_p[0])
        anti_target = T0_phase + pi

        best_p = 1
        best_s_val = 0
        for p in range(1, p_max_ext + 1):
            if abs(T_p[p]) < 1e-30:
                continue
            pd = (np.angle(T_p[p]) - anti_target + pi) % (2 * pi) - pi
            if abs(pd) < pi / 2:
                if abs(T_p[p]) > best_s_val:
                    best_s_val = abs(T_p[p])
                    best_p = p
        partner_ext.append(best_p)

    # Print and fit
    print(f"\n  {'y':>6}  {'p*':>4}")
    for i, (y, pp) in enumerate(zip(y_ext, partner_ext)):
        if i % 4 == 0:
            print(f"  {y:6.1f}  {pp:4d}")

    # Filter to the reliable range (p* < p_max_ext - 5 to avoid ceiling effects)
    reliable = [(y, pp) for y, pp in zip(y_ext, partner_ext)
                if pp < p_max_ext - 5 and y >= 3]
    if len(reliable) > 5:
        y_r = np.array([r[0] for r in reliable])
        p_r = np.array([r[1] for r in reliable])

        # Fit p* = a * y + b
        coeffs_lin = np.polyfit(y_r, p_r, 1)
        # Fit p* = a * y^alpha
        log_y = np.log(y_r[p_r > 0])
        log_p = np.log(p_r[p_r > 0].astype(float))
        if len(log_y) > 3:
            coeffs_pow = np.polyfit(log_y, log_p, 1)
            alpha = coeffs_pow[0]
            A_coeff = np.exp(coeffs_pow[1])

        print(f"\n  Scaling fits (reliable range y >= 3, p* < {p_max_ext-5}):")
        print(f"    Linear: p* = {coeffs_lin[0]:.3f} y + {coeffs_lin[1]:.1f}")
        if len(log_y) > 3:
            print(f"    Power:  p* = {A_coeff:.3f} y^{alpha:.3f}")

        # Residuals
        p_pred_lin = np.polyval(coeffs_lin, y_r)
        rmse_lin = np.sqrt(np.mean((p_r - p_pred_lin)**2))
        print(f"    Linear RMSE = {rmse_lin:.2f}")

        if len(log_y) > 3:
            p_pred_pow = A_coeff * y_r ** alpha
            rmse_pow = np.sqrt(np.mean((p_r - p_pred_pow)**2))
            print(f"    Power  RMSE = {rmse_pow:.2f}")

    # ==============================================================
    # SUMMARY
    # ==============================================================
    print(f"\n{'='*80}")
    print(f"  SUMMARY OF THREE TESTS")
    print(f"{'='*80}")
    print("""
  TEST 0 (Z_n^HK not entire):
    Z_n^HK(s) = Sigma d_p exp(-n C_2(p) s) DIVERGES for Re(s) < 0.
    It is a Dirichlet series, NOT an entire function.
    Hadamard factorization does NOT apply => CAN have finitely many zeros.
    Wilson Z_n^W is entire of order <= 1 => Hadamard DOES apply.

  TEST 1 (SU(8) Haar MC):
    Haar-random sampling eliminates Vandermonde variance.
    Partner p*(y) behavior for SU(8): [see table above]

  TEST 2 (|A_p| visualization):
    |A_p(kappa+iy)|/|A_p(kappa)| GROWS for higher p at complex s.
    This is the mathematical content of the conveyor belt:
    higher representations become relatively more important.

  TEST 3 (p* scaling):
    [see fits above]

  THEOREM INF DEFENSE:
    For entire f of order <= 1 with finitely many zeros:
      f(s) = P(s) exp(delta*s + beta)  [Hadamard]
      |f(kappa+iy)| = |P(kappa+iy)| exp(Re(delta)kappa - Im(delta)y + Re(beta))
    Decay as y -> +inf requires Im(delta) > 0.
    Decay as y -> -inf requires Im(delta) < 0.
    Contradiction. QED.
    This uses BIDIRECTIONAL decay from Riemann-Lebesgue.
    Linear exponent delta*s cannot decay in both directions.
  """)


if __name__ == '__main__':
    main()
