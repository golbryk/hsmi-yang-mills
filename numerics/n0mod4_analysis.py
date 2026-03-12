"""
N ≡ 0 mod 4 Fisher Zero Analysis
==================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 4b — Sub-dominant interference for N ≡ 0 mod 4

For N ≡ 0 mod 4, the dominant balanced-split saddle has Φ = 0.
The partition function at leading order is:
    Z ~ S_dom / y^{2m_dom}  (non-oscillating)

Sub-dominant saddles at k = N/2 ± 2 have Φ = ±4 with Δord = 8:
    Z ~ S_dom / y^{2m_dom} + 2 S_sub cos(8y) / y^{2m_sub}

The question: does the sub-dominant oscillating term produce zeros?
This requires the sub-dominant amplitude to exceed the dominant:
    |S_sub / y^{m_sub}| > |S_dom / y^{m_dom}|

Since m_sub > m_dom (by Δord/2 = 4), the sub-dominant term is suppressed
by y^{-4} relative to dominant. Therefore:
- For small y: sub-dominant CAN dominate (y^{-4} ratio is moderate)
- For large y: dominant always wins → NO zeros

This means there can be at most FINITELY MANY zeros, not an infinite sequence.
"""

from math import comb, pi, exp, acos, sin, cos
from itertools import combinations_with_replacement as cr


# ---------------------------------------------------------------------------
# Helpers
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


def vandermonde_order(k, N):
    return k * (k - 1) + (N - k) * (N - k - 1)


# ---------------------------------------------------------------------------
# Saddle structure for N ≡ 0 mod 4
# ---------------------------------------------------------------------------

def analyze_n0mod4(N, kappa, n_reps=20):
    """Analyze the saddle structure for N ≡ 0 mod 4."""
    assert N % 4 == 0

    k_dom = N // 2  # dominant saddle: Φ = 0
    k_sub_p = N // 2 + 2  # sub-dominant: Φ = +4
    k_sub_m = N // 2 - 2  # sub-dominant: Φ = -4

    ord_dom = vandermonde_order(k_dom, N)
    ord_sub = vandermonde_order(k_sub_p, N)
    delta_ord = ord_sub - ord_dom

    # Eigenvalues at dominant saddle
    eig_dom = [1.0] * k_dom + [-1.0] * (N - k_dom)

    # Eigenvalues at sub-dominant saddles
    eig_sub_p = [1.0] * k_sub_p + [-1.0] * (N - k_sub_p)
    eig_sub_m = [1.0] * k_sub_m + [-1.0] * (N - k_sub_m)

    # Check SU(N) constraint: det U = 1 requires (-1)^{N-k} = 1
    # For N ≡ 0 mod 4: need k even
    assert k_dom % 2 == 0, f"k_dom = {k_dom} not even"
    assert k_sub_p % 2 == 0, f"k_sub_p = {k_sub_p} not even"
    assert k_sub_m % 2 == 0, f"k_sub_m = {k_sub_m} not even"

    # Compute character expansion sums for dominant saddle
    S_dom = 0.0
    for p in range(n_reps):
        d = dimension_suN(p, N)
        chi = h_p_general(p, eig_dom)
        A_p = d * exp(-casimir_suN(p, N) * kappa)
        S_dom += d * chi ** 2 * A_p

    # Compute character expansion sums for sub-dominant saddle
    S_sub = 0.0
    S_sub_AB = 0.0
    for p in range(n_reps):
        d = dimension_suN(p, N)
        chi_p = h_p_general(p, eig_sub_p)
        chi_m = h_p_general(p, eig_sub_m)
        A_p = d * exp(-casimir_suN(p, N) * kappa)
        S_sub += d * chi_p ** 2 * A_p
        S_sub_AB += d * chi_p * chi_m * A_p

    # Check Schur parity for sub-dominant pair
    parity_ok = True
    for p in range(min(10, n_reps)):
        chi_p = h_p_general(p, eig_sub_p)
        chi_m = h_p_general(p, eig_sub_m)
        if abs(chi_m - ((-1) ** p) * chi_p) > 1e-8:
            parity_ok = False
            break

    # Ratio of sub-dominant to dominant amplitude
    # The sub-dominant term is S_sub / y^{m_sub} vs S_dom / y^{m_dom}
    # Ratio: (S_sub / S_dom) * y^{-(m_sub - m_dom)}
    # = (S_sub / S_dom) * y^{-delta_ord/2}
    amplitude_ratio = S_sub / S_dom if S_dom > 0 else float('inf')

    # Critical y where sub-dominant amplitude equals dominant:
    # amplitude_ratio * y^{-delta_ord/2} = 1
    # y_crit = amplitude_ratio^{2/delta_ord}
    if delta_ord > 0 and amplitude_ratio > 0:
        y_crit = amplitude_ratio ** (2.0 / delta_ord)
    else:
        y_crit = float('inf')

    return {
        'N': N, 'kappa': kappa,
        'k_dom': k_dom, 'k_sub_p': k_sub_p, 'k_sub_m': k_sub_m,
        'ord_dom': ord_dom, 'ord_sub': ord_sub, 'delta_ord': delta_ord,
        'S_dom': S_dom, 'S_sub': S_sub, 'S_sub_AB': S_sub_AB,
        'amplitude_ratio': amplitude_ratio,
        'y_crit': y_crit,
        'schur_parity': parity_ok,
    }


def print_analysis(N_list, kappa_list, n_reps=15):
    """Print analysis for N ≡ 0 mod 4."""
    print()
    print("=" * 90)
    print("  N ≡ 0 mod 4: Sub-Dominant Interference Analysis")
    print("  Author: Grzegorz Olbryk  |  March 2026  |  Task 4b")
    print("=" * 90)
    print()

    for N in N_list:
        if N % 4 != 0:
            continue

        print(f"  SU({N}):")
        print(f"  " + "-" * 60)

        for kappa in kappa_list:
            r = analyze_n0mod4(N, kappa, n_reps)

            print(f"  κ = {kappa:.1f}:")
            print(f"    Dominant: k={r['k_dom']}, Φ=0, "
                  f"ord={r['ord_dom']}, S_dom={r['S_dom']:.4e}")
            print(f"    Sub-dom:  k={r['k_sub_p']},{r['k_sub_m']}, Φ=±4, "
                  f"ord={r['ord_sub']}, S_sub={r['S_sub']:.4e}")
            print(f"    Δord = {r['delta_ord']}, "
                  f"amplitude ratio = {r['amplitude_ratio']:.6f}")
            print(f"    Schur parity (sub): {r['schur_parity']}")
            print(f"    y_crit (sub ≈ dom): {r['y_crit']:.4f}")

            # Interpretation
            if r['y_crit'] < 1.0:
                print(f"    → Sub-dominant NEVER dominates (y_crit < 1).")
                print(f"    → PREDICTION: NO Fisher zeros on Re s = κ line.")
            elif r['y_crit'] < pi:
                print(f"    → Sub-dominant dominates for y < {r['y_crit']:.2f}.")
                print(f"    → PREDICTION: POSSIBLY finitely many zeros "
                      f"for y < {r['y_crit']:.2f}.")
            else:
                print(f"    → Sub-dominant dominates in a wide region.")
                print(f"    → PREDICTION: Fisher zeros LIKELY exist.")

        print()

    # Summary
    print("  SUMMARY")
    print("  " + "-" * 60)
    print("  For N ≡ 0 mod 4, the dominant saddle (Φ = 0) is non-oscillating.")
    print("  The sub-dominant pair (Φ = ±4) produces oscillations that are")
    print("  suppressed by y^{-Δord/2} = y^{-4} relative to dominant.")
    print()
    print("  Key structural difference from |Φ₀| ≠ 0 cases:")
    print("  - |Φ₀| ≠ 0: infinite sequence of zeros at regular spacing")
    print("  - |Φ₀| = 0: at most finitely many zeros (sub-dominant dominance")
    print("    region is bounded)")
    print()
    print("  This is a QUALITATIVE DIFFERENCE in the Fisher zero structure.")
    print("=" * 90)
    print()


if __name__ == '__main__':
    print_analysis([4, 8, 12], [0.5, 1.0, 2.0])
