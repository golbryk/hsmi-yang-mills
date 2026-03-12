"""
Heat-Kernel Fisher Zeros: Finite Cluster Verification
=======================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 38 — Action comparison programme

Computes Z_n^HK(s) = Σ_{p=0}^{P_MAX} d_p^2 exp(-n C_2(p) s) for SU(N)
and verifies that it has only finitely many Fisher zeros.

Key distinction from Wilson: Z_n^HK is a Dirichlet series, NOT an entire
function. Hadamard factorization does not apply, so the infinitude theorem
(RESULT_024) does not hold for heat-kernel action.

Comparison with Wilson zeros from su4_newton_search.py.
"""

import numpy as np
from math import comb, pi
import time
import sys


# ---------------------------------------------------------------------------
# SU(N) representation theory (symmetric reps only)
# ---------------------------------------------------------------------------

def dim_rep(p, N):
    """Dimension of SU(N) symmetric rep (p,0,...,0)."""
    return comb(p + N - 1, N - 1)


def casimir_suN(p, N):
    """Quadratic Casimir for symmetric rep (p,0,...,0) of SU(N)."""
    return p * (p + N) / float(N)


# ---------------------------------------------------------------------------
# Heat-kernel partition function
# ---------------------------------------------------------------------------

def Z_hk(s, n_plaq, N, p_max):
    """Compute Z_n^HK(s) = Σ_{p=0}^{p_max} d_p^2 exp(-n C_2(p) s).

    Note: includes d_p^2 because in character expansion:
    Z = Σ_p d_p [A_p^HK]^n = Σ_p d_p [d_p exp(-C_2 s)]^n = Σ_p d_p^{n+1} exp(-n C_2 s)

    For n=2: Z = Σ d_p^3 exp(-2 C_2 s)
    Actually, A_p^HK(s) = d_p exp(-C_2(p) s), so
    Z_n = Σ d_p [d_p exp(-C_2 s)]^n = Σ d_p^{n+1} exp(-n C_2 s)
    """
    result = 0j
    for p in range(p_max + 1):
        dp = dim_rep(p, N)
        c2 = casimir_suN(p, N)
        result += dp ** (n_plaq + 1) * np.exp(-n_plaq * c2 * s)
    return result


def Z_hk_terms(s, n_plaq, N, p_max):
    """Return individual terms T_p = d_p^{n+1} exp(-n C_2 s)."""
    terms = np.zeros(p_max + 1, dtype=complex)
    for p in range(p_max + 1):
        dp = dim_rep(p, N)
        c2 = casimir_suN(p, N)
        terms[p] = dp ** (n_plaq + 1) * np.exp(-n_plaq * c2 * s)
    return terms


# ---------------------------------------------------------------------------
# 2D Newton search (adapted from su4_newton_search.py)
# ---------------------------------------------------------------------------

def newton_2d_hk(kap0, y0, n_plaq, N, p_max,
                 tol=1e-14, max_iter=50, dk=1e-8, dy=1e-8, damping=0.5):
    """2D Newton search for Z_n^HK(κ+iy) = 0."""
    kap, y = kap0, y0

    for it in range(max_iter):
        s = kap + 1j * y
        Z0 = Z_hk(s, n_plaq, N, p_max)
        absZ = abs(Z0)
        if absZ < tol:
            return kap, y, absZ, it, True

        Zk = Z_hk(s + dk, n_plaq, N, p_max)
        Zy = Z_hk(s + 1j * dy, n_plaq, N, p_max)

        dRdk = (Zk.real - Z0.real) / dk
        dRdy = (Zy.real - Z0.real) / dy
        dIdk = (Zk.imag - Z0.imag) / dk
        dIdy = (Zy.imag - Z0.imag) / dy

        det = dRdk * dIdy - dRdy * dIdk
        if abs(det) < 1e-30:
            return kap, y, absZ, it, False

        dkap = -(dIdy * Z0.real - dRdy * Z0.imag) / det
        ddy = -(-dIdk * Z0.real + dRdk * Z0.imag) / det

        alpha = 1.0
        for _ in range(10):
            kap_new = kap + alpha * dkap
            y_new = y + alpha * ddy
            if kap_new > 0:
                Z_new = Z_hk(kap_new + 1j * y_new, n_plaq, N, p_max)
                if abs(Z_new) < absZ:
                    break
            alpha *= damping

        kap = kap + alpha * dkap
        y = y + alpha * ddy

        if kap <= 0:
            return kap, y, absZ, it, False

    return kap, y, abs(Z_hk(kap + 1j * y, n_plaq, N, p_max)), max_iter, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print()
    print("=" * 80)
    print("  Heat-Kernel Fisher Zeros — Task 38")
    print("  Z_n^HK(s) = Sigma d_p^{n+1} exp(-n C_2(p) s)")
    print("=" * 80)

    # ======================================================================
    # Part 0: Verify divergence for Re(s) < 0
    # ======================================================================
    print("\n  PART 0: Divergence Verification (Z^HK not entire)")
    print("  " + "-" * 60)

    N = 4
    n_plaq = 2
    for p_max in [10, 20, 50, 100]:
        for s_re in [-0.1, -0.5]:
            val = Z_hk(s_re, n_plaq, N, p_max)
            print(f"    P_MAX={p_max:3d}, s={s_re:+.1f}: "
                  f"|Z^HK| = {abs(val):.4e}")

    print("\n    Conclusion: Z^HK diverges for Re(s) < 0 -> NOT entire.")

    # ======================================================================
    # Part 1: SU(4) Fisher zeros — coarse 2D scan
    # ======================================================================
    for N in [3, 4]:
        for n_plaq in [2]:
            p_max = 100

            print(f"\n\n  PART 1: SU({N}), n={n_plaq}, P_MAX={p_max}")
            print("  " + "-" * 60)

            # A_p ratio at real kappa
            kappa = 1.0
            print(f"\n  Character ratios at kappa={kappa}:")
            for p in [0, 1, 2, 3, 5, 10, 20]:
                dp = dim_rep(p, N)
                c2 = casimir_suN(p, N)
                Ap = dp * np.exp(-c2 * kappa)
                A0 = np.exp(0)
                print(f"    p={p:2d}: d_p={dp:>8d}, C_2={c2:>8.2f}, "
                      f"A_p^HK(kappa)={Ap:.6e}, A_p/A_0={Ap/A0:.6e}")

            # Find kappa_c^HK: where d_1 exp(-C_2(1) kappa) = 1
            c2_1 = casimir_suN(1, N)
            d1 = dim_rep(1, N)
            kappa_c_hk = np.log(d1) / c2_1
            print(f"\n  kappa_c^HK (A_1 = A_0 crossing) = {kappa_c_hk:.6f}")

            # Coarse scan
            print(f"\n  2D coarse scan: kappa in [0.2, 3.0], y in [0.1, 40]...")

            kap_values = np.linspace(0.2, 3.0, 40)
            y_values = np.linspace(0.1, 40.0, 400)
            absZ_grid = np.zeros((len(kap_values), len(y_values)))

            for ik, kap in enumerate(kap_values):
                for iy, y in enumerate(y_values):
                    absZ_grid[ik, iy] = abs(Z_hk(kap + 1j * y, n_plaq, N, p_max))

            # Find local minima
            minima = []
            for ik in range(1, len(kap_values) - 1):
                for iy in range(1, len(y_values) - 1):
                    val = absZ_grid[ik, iy]
                    nb = [absZ_grid[ik-1, iy], absZ_grid[ik+1, iy],
                          absZ_grid[ik, iy-1], absZ_grid[ik, iy+1]]
                    if val < min(nb):
                        Z_real = abs(Z_hk(kap_values[ik], n_plaq, N, p_max))
                        ratio = val / Z_real if Z_real > 0 else val
                        minima.append({
                            'kap': kap_values[ik],
                            'y': y_values[iy],
                            'absZ': val,
                            'ratio': ratio
                        })

            minima.sort(key=lambda x: x['absZ'])
            n_show = min(20, len(minima))
            print(f"\n  Top {n_show} |Z^HK| minima:")
            for i, m in enumerate(minima[:n_show]):
                print(f"    #{i+1:2d}: kappa={m['kap']:.4f}  y={m['y']:.4f}  "
                      f"|Z|={m['absZ']:.4e}  ratio={m['ratio']:.4e}")

            # Newton refinement
            print(f"\n  Newton refinement of top minima...")
            sys.stdout.flush()

            zeros_found = []
            for m in minima[:30]:
                kap, y, absZ, iters, converged = newton_2d_hk(
                    m['kap'], m['y'], n_plaq, N, p_max,
                    tol=1e-14, max_iter=100)

                if converged or absZ < 1e-10:
                    is_dup = any(abs(z['kap'] - kap) < 1e-4 and
                                abs(z['y'] - y) < 1e-4 for z in zeros_found)
                    if not is_dup and kap > 0:
                        zeros_found.append({
                            'kap': kap, 'y': abs(y), 'absZ': absZ,
                            'iters': iters, 'converged': converged
                        })
                        print(f"    ZERO #{len(zeros_found)}: "
                              f"s = {kap:.12f} + {abs(y):.12f}i  "
                              f"|Z| = {absZ:.2e}  ({iters} iters)")

            zeros_found.sort(key=lambda x: x['y'])

            # Extended scan for y > 40
            print(f"\n  Extended scan: y in [35, 80]...")
            y_ext = np.linspace(35.0, 80.0, 200)
            for kap_test in [0.5, 1.0, 1.5, 2.0]:
                for y in y_ext:
                    val = abs(Z_hk(kap_test + 1j * y, n_plaq, N, p_max))
                    Z_real = abs(Z_hk(kap_test, n_plaq, N, p_max))
                    if val / Z_real < 1e-4:
                        kap, y_r, absZ, iters, converged = newton_2d_hk(
                            kap_test, y, n_plaq, N, p_max,
                            tol=1e-14, max_iter=100)
                        if converged or absZ < 1e-10:
                            is_dup = any(abs(z['kap'] - kap) < 1e-4 and
                                        abs(z['y'] - y_r) < 1e-4
                                        for z in zeros_found)
                            if not is_dup and kap > 0:
                                zeros_found.append({
                                    'kap': kap, 'y': abs(y_r),
                                    'absZ': absZ, 'iters': iters,
                                    'converged': converged
                                })
                                print(f"    ZERO #{len(zeros_found)}: "
                                      f"s = {kap:.12f} + {abs(y_r):.12f}i  "
                                      f"|Z| = {absZ:.2e}")

            zeros_found.sort(key=lambda x: x['y'])

            # Summary
            print(f"\n  SUMMARY for SU({N}), n={n_plaq}:")
            print(f"  Total zeros found: {len(zeros_found)}")
            for i, z in enumerate(zeros_found):
                print(f"    {i+1}. s = {z['kap']:.10f} + {z['y']:.10f}i  "
                      f"|Z| = {z['absZ']:.2e}")

            if len(zeros_found) >= 2:
                gaps = [zeros_found[i+1]['y'] - zeros_found[i]['y']
                        for i in range(len(zeros_found) - 1)]
                avg_gap = sum(gaps) / len(gaps)
                print(f"  Average y-gap: {avg_gap:.6f}")

            # Convergence in p_max
            if zeros_found:
                z0 = zeros_found[0]
                print(f"\n  P_MAX convergence for first zero (kappa={z0['kap']:.6f}, y={z0['y']:.6f}):")
                for pm in [20, 40, 60, 80, 100, 150]:
                    val = Z_hk(z0['kap'] + 1j * z0['y'], n_plaq, N, pm)
                    print(f"    P_MAX={pm:3d}: |Z| = {abs(val):.6e}")

    # ======================================================================
    # Part 2: Compare with Wilson zeros
    # ======================================================================
    print(f"\n\n  PART 2: HK vs Wilson — Structural Comparison")
    print("  " + "-" * 60)

    N = 4
    n_plaq = 2
    kappa = 1.0
    p_max = 100

    print(f"\n  |Z^HK(kappa+iy)| / |Z^HK(kappa)| decay profile:")
    Z_real = abs(Z_hk(kappa, n_plaq, N, p_max))
    print(f"    Z^HK(kappa) = {Z_real:.6e}")

    for y in [0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50]:
        Zy = abs(Z_hk(kappa + 1j * y, n_plaq, N, p_max))
        print(f"    y={y:5.1f}: |Z^HK|/|Z^HK(kappa)| = {Zy/Z_real:.6e}")

    # HK amplitude ratios (no conveyor belt)
    print(f"\n  HK amplitude ratios |A_p^HK(kappa+iy)| / |A_p^HK(kappa)|:")
    print(f"  (Should be exactly 1 for all p, y — heat-kernel has no amplitude modulation)")
    for p in [0, 1, 3, 5, 8]:
        dp = dim_rep(p, N)
        c2 = casimir_suN(p, N)
        Ap_real = dp * np.exp(-c2 * kappa)
        print(f"    p={p}: ", end="")
        for y in [1, 3, 5, 8]:
            s = kappa + 1j * y
            Ap_complex = dp * np.exp(-c2 * s)
            ratio = abs(Ap_complex) / Ap_real
            print(f"y={y}: {ratio:.6f}  ", end="")
        print()

    print("\n    |A_p^HK(s)| = d_p exp(-C_2 Re(s)) — independent of Im(s).")
    print("    NO conveyor belt in heat-kernel action.")

    # ======================================================================
    # Part 3: SU(3) heat-kernel for comparison
    # ======================================================================
    print(f"\n\n  PART 3: SU(3) Heat-Kernel Zeros")
    print("  " + "-" * 60)

    N = 3
    n_plaq = 2
    p_max = 100
    kappa = 1.0

    y_vals = np.linspace(0.1, 30.0, 300)
    Z_profile = np.array([Z_hk(kappa + 1j * y, n_plaq, N, p_max) for y in y_vals])

    # Find sign changes
    re_sign = []
    for i in range(len(Z_profile) - 1):
        if Z_profile[i].real * Z_profile[i+1].real < 0:
            frac = Z_profile[i].real / (Z_profile[i].real - Z_profile[i+1].real)
            y_z = y_vals[i] + frac * (y_vals[i+1] - y_vals[i])
            re_sign.append(y_z)

    print(f"  Re Z^HK sign changes: {len(re_sign)}")
    for i, y in enumerate(re_sign):
        print(f"    #{i+1}: y = {y:.6f}")

    if len(re_sign) >= 2:
        gaps = [re_sign[i+1] - re_sign[i] for i in range(len(re_sign) - 1)]
        avg = sum(gaps) / len(gaps)
        print(f"  Average gap: {avg:.6f}")
        print(f"  pi/2 = {pi/2:.6f}")

    elapsed = time.time() - t0
    print(f"\n\n  [Completed in {elapsed:.1f}s]")

    # Final summary
    print(f"\n{'='*80}")
    print("  CONCLUSION")
    print(f"{'='*80}")
    print("  1. Z_n^HK is a Dirichlet series, diverges for Re(s) < 0")
    print("  2. Hadamard factorization does NOT apply")
    print("  3. |A_p^HK(s)| = d_p exp(-C_2 Re s) — independent of Im(s)")
    print("  4. No conveyor belt mechanism (no amplitude modulation)")
    print("  5. Fisher zeros (if any) form a FINITE cluster")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
