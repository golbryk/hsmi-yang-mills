"""
Heat-Kernel Quasi-Periodic Mechanism Analysis
===============================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 46 — Universality deep dive Phase 2

Formally classifies HK zeros as arising from almost-periodic interference.
Identifies dominant term pairs at each zero, computes beat frequencies,
and tests periodicity predictions.

Z_n^HK(κ+iy) = Σ_p w_p exp(-i n C_2(p) y)
  where w_p = d_p^{n+1} exp(-n C_2(p) κ) ∈ R+
  frequencies: ω_p = n C_2(p) = n p(p+N)/N

Key distinction from Wilson: HK phases are LINEAR in y (pure exponentials),
so interference is almost-periodic. Wilson phases are NONLINEAR (no simple
frequency decomposition).

Reuses: hk_fisher_zeros.py (Z_hk, dim_rep, casimir_suN)
"""

import numpy as np
from math import comb, pi
import time
import sys


# ---------------------------------------------------------------------------
# SU(N) representation theory
# ---------------------------------------------------------------------------

def dim_rep(p, N):
    return comb(p + N - 1, N - 1)


def casimir_suN(p, N):
    return p * (p + N) / float(N)


# ---------------------------------------------------------------------------
# HK partition function and term decomposition
# ---------------------------------------------------------------------------

def Z_hk(s, n_plaq, N, p_max):
    """Z_n^HK(s) = Σ d_p^{n+1} exp(-n C_2(p) s)."""
    result = 0j
    for p in range(p_max + 1):
        dp = dim_rep(p, N)
        c2 = casimir_suN(p, N)
        result += dp ** (n_plaq + 1) * np.exp(-n_plaq * c2 * s)
    return result


def hk_terms(s, n_plaq, N, p_max):
    """Return individual terms T_p = d_p^{n+1} exp(-n C_2(p) s)."""
    terms = np.zeros(p_max + 1, dtype=complex)
    for p in range(p_max + 1):
        dp = dim_rep(p, N)
        c2 = casimir_suN(p, N)
        terms[p] = dp ** (n_plaq + 1) * np.exp(-n_plaq * c2 * s)
    return terms


def hk_weights(kap, n_plaq, N, p_max):
    """Real positive weights w_p = d_p^{n+1} exp(-n C_2(p) κ)."""
    w = np.zeros(p_max + 1)
    for p in range(p_max + 1):
        dp = dim_rep(p, N)
        c2 = casimir_suN(p, N)
        w[p] = dp ** (n_plaq + 1) * np.exp(-n_plaq * c2 * kap)
    return w


def hk_frequencies(n_plaq, N, p_max):
    """Frequencies ω_p = n C_2(p)."""
    return np.array([n_plaq * casimir_suN(p, N) for p in range(p_max + 1)])


# ---------------------------------------------------------------------------
# Newton search for HK zeros (from hk_fisher_zeros.py)
# ---------------------------------------------------------------------------

def newton_2d_hk(kap0, y0, n_plaq, N, p_max,
                 tol=1e-14, max_iter=50, dk=1e-8, dy=1e-8, damping=0.5):
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
# Find HK zeros by coarse scan + Newton
# ---------------------------------------------------------------------------

def find_hk_zeros(N, n_plaq, p_max, kap_range=(0.2, 3.0), y_range=(0.1, 40.0),
                  n_kap=30, n_y=300):
    kap_values = np.linspace(kap_range[0], kap_range[1], n_kap)
    y_values = np.linspace(y_range[0], y_range[1], n_y)

    absZ = np.zeros((n_kap, n_y))
    for ik, kap in enumerate(kap_values):
        for iy, y in enumerate(y_values):
            absZ[ik, iy] = abs(Z_hk(kap + 1j * y, n_plaq, N, p_max))

    minima = []
    for ik in range(1, n_kap - 1):
        for iy in range(1, n_y - 1):
            val = absZ[ik, iy]
            nb = [absZ[ik-1, iy], absZ[ik+1, iy],
                  absZ[ik, iy-1], absZ[ik, iy+1]]
            if val < min(nb):
                minima.append({'kap': kap_values[ik], 'y': y_values[iy], 'absZ': val})

    minima.sort(key=lambda x: x['absZ'])

    zeros_found = []
    for m in minima[:40]:
        kap, y, absZ_val, iters, converged = newton_2d_hk(
            m['kap'], m['y'], n_plaq, N, p_max, tol=1e-14, max_iter=100)
        if (converged or absZ_val < 1e-10) and kap > 0:
            is_dup = any(abs(z['kap'] - kap) < 1e-4 and
                        abs(z['y'] - abs(y)) < 1e-4 for z in zeros_found)
            if not is_dup:
                zeros_found.append({'kap': kap, 'y': abs(y), 'absZ': absZ_val})

    zeros_found.sort(key=lambda x: x['y'])
    return zeros_found


# ---------------------------------------------------------------------------
# Dominant pair analysis
# ---------------------------------------------------------------------------

def analyze_dominant_pairs(zeros, n_plaq, N, p_max, p_check=20):
    """At each zero, identify which terms dominate and classify mechanism."""
    results = []

    for zf in zeros:
        kap, y = zf['kap'], zf['y']
        s = kap + 1j * y

        # Get all terms
        terms = hk_terms(s, n_plaq, N, p_max)
        abs_terms = np.abs(terms[:p_check])

        # Sort by magnitude
        order = np.argsort(-abs_terms)
        top = order[:5]

        # Check 2-term cancellation
        T_total = np.sum(terms[:p_check])

        # Try all pairs from top 5
        best_pair = None
        best_residual = abs(T_total)
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                p1, p2 = top[i], top[j]
                pair_sum = terms[p1] + terms[p2]
                remainder = T_total - pair_sum
                # How well does this pair explain the cancellation?
                if abs(pair_sum) < abs(T_total) * 10:  # pair contributes to cancellation
                    residual = abs(T_total) / (abs(terms[p1]) + abs(terms[p2]))
                    if best_pair is None or abs(pair_sum + remainder) < best_residual:
                        best_pair = (p1, p2)

        # Classify
        mechanism = "2-term" if best_pair is not None else "multi-term"

        # Beat frequency
        if best_pair:
            p1, p2 = best_pair
            freq1 = n_plaq * casimir_suN(p1, N)
            freq2 = n_plaq * casimir_suN(p2, N)
            delta_freq = abs(freq2 - freq1)
            beat_period = 2 * pi / delta_freq if delta_freq > 1e-10 else float('inf')
        else:
            p1, p2 = top[0], top[1]
            freq1 = n_plaq * casimir_suN(p1, N)
            freq2 = n_plaq * casimir_suN(p2, N)
            delta_freq = abs(freq2 - freq1)
            beat_period = 2 * pi / delta_freq if delta_freq > 1e-10 else float('inf')

        results.append({
            'kap': kap, 'y': y,
            'pair': (p1, p2),
            'mechanism': mechanism,
            'beat_period': beat_period,
            'delta_freq': delta_freq,
            'abs_top': [(int(top[k]), abs_terms[top[k]]) for k in range(min(4, len(top)))],
            'Z_residual': abs(T_total),
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print()
    print("=" * 90)
    print("  Heat-Kernel Quasi-Periodic Mechanism Analysis — Task 46")
    print("=" * 90)

    # ======================================================================
    # Part 1: Weight and frequency spectrum
    # ======================================================================
    for N in [3, 4]:
        n_plaq = 2
        p_max = 100
        kappa = 1.0

        print(f"\n\n  {'='*80}")
        print(f"  SU({N}), n={n_plaq}, P_MAX={p_max}")
        print(f"  {'='*80}")

        # Part 1a: Weight spectrum
        print(f"\n  PART 1: Weight and Frequency Spectrum")
        print("  " + "-" * 60)

        w = hk_weights(kappa, n_plaq, N, p_max)
        freq = hk_frequencies(n_plaq, N, p_max)

        print(f"\n  Weights w_p = d_p^{{n+1}} exp(-n C_2 κ) at κ={kappa}:")
        print(f"  {'p':>4} {'d_p':>10} {'C_2(p)':>10} {'w_p':>14} "
              f"{'w_p/w_0':>12} {'ω_p':>10}")
        for p in range(15):
            dp = dim_rep(p, N)
            c2 = casimir_suN(p, N)
            print(f"  {p:4d} {dp:10d} {c2:10.4f} {w[p]:14.6e} "
                  f"{w[p]/w[0]:12.6e} {freq[p]:10.4f}")

        # Effective number of terms
        w_total = np.sum(w)
        w_sorted = np.sort(w)[::-1]
        cumsum = np.cumsum(w_sorted) / w_total
        n_eff = np.searchsorted(cumsum, 0.99) + 1
        print(f"\n  Total weight: {w_total:.6e}")
        print(f"  N_eff (99% of weight): {n_eff} terms")

        # Part 1b: Frequency analysis
        print(f"\n  Frequency spectrum:")
        print(f"  ω_p = n C_2(p) = {n_plaq} · p(p+{N})/{N}")
        print(f"\n  Beat frequencies Δω_{'{p,q}'} = |ω_p - ω_q|:")
        print(f"  {'(p,q)':>8} {'Δω':>10} {'Beat period':>12} {'w_p·w_q':>14}")

        for p in range(5):
            for q in range(p + 1, min(p + 4, 8)):
                dw = abs(freq[q] - freq[p])
                bp = 2 * pi / dw if dw > 1e-10 else float('inf')
                print(f"  ({p},{q}){' '*(5-len(str(p))-len(str(q)))} "
                      f"{dw:10.4f} {bp:12.4f} {w[p]*w[q]:14.6e}")

        # ======================================================================
        # Part 2: Find HK zeros
        # ======================================================================
        print(f"\n\n  PART 2: HK Fisher Zeros")
        print("  " + "-" * 60)

        zeros = find_hk_zeros(N, n_plaq, p_max,
                              kap_range=(0.2, 3.0), y_range=(0.1, 40.0),
                              n_kap=25, n_y=300)

        print(f"  Found {len(zeros)} zeros:")
        for i, zf in enumerate(zeros):
            print(f"    #{i+1}: s = {zf['kap']:.10f} + {zf['y']:.10f}i  "
                  f"|Z| = {zf['absZ']:.2e}")

        if len(zeros) >= 2:
            gaps = [zeros[i+1]['y'] - zeros[i]['y']
                    for i in range(len(zeros) - 1)]
            avg_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            print(f"\n  Average gap: {avg_gap:.6f} ± {std_gap:.6f}")

        # ======================================================================
        # Part 3: Dominant pair analysis at each zero
        # ======================================================================
        print(f"\n\n  PART 3: Dominant Pair Analysis at Each Zero")
        print("  " + "-" * 60)

        pair_results = analyze_dominant_pairs(zeros, n_plaq, N, p_max)

        print(f"\n  {'#':>3} {'y':>9} {'pair':>8} {'mech':>10} {'beat_T':>10} "
              f"{'Δω':>8} {'|T_p1|':>12} {'|T_p2|':>12}")
        for i, r in enumerate(pair_results):
            p1, p2 = r['pair']
            print(f"  {i+1:3d} {r['y']:9.4f} ({p1},{p2}){' '*(4-len(str(p1))-len(str(p2)))} "
                  f"{r['mechanism']:>10} {r['beat_period']:10.4f} "
                  f"{r['delta_freq']:8.4f} "
                  f"{r['abs_top'][0][1]:12.4e} {r['abs_top'][1][1]:12.4e}")

        # ======================================================================
        # Part 4: Term decomposition at zeros
        # ======================================================================
        print(f"\n\n  PART 4: Full Term Decomposition at Selected Zeros")
        print("  " + "-" * 60)

        for idx in range(min(5, len(zeros))):
            zf = zeros[idx]
            s = zf['kap'] + 1j * zf['y']
            terms = hk_terms(s, n_plaq, N, p_max)

            print(f"\n  Zero #{idx+1}: s = {zf['kap']:.6f} + {zf['y']:.6f}i")
            print(f"  {'p':>4} {'|T_p|':>14} {'Re T_p':>14} {'Im T_p':>14} "
                  f"{'phase/π':>10} {'cum Re':>14} {'cum Im':>14}")

            cum_re = 0.0
            cum_im = 0.0
            for p in range(12):
                t = terms[p]
                cum_re += t.real
                cum_im += t.imag
                if abs(t) > abs(terms[0]) * 1e-10:
                    print(f"  {p:4d} {abs(t):14.6e} {t.real:+14.6e} "
                          f"{t.imag:+14.6e} {np.angle(t)/pi:10.4f} "
                          f"{cum_re:+14.6e} {cum_im:+14.6e}")

            Z_total = np.sum(terms)
            print(f"  Sum: {abs(Z_total):14.6e}  "
                  f"(Re={Z_total.real:+.4e}, Im={Z_total.imag:+.4e})")

        # ======================================================================
        # Part 5: Beat period predictions vs observed gaps
        # ======================================================================
        if len(zeros) >= 3 and len(pair_results) >= 2:
            print(f"\n\n  PART 5: Beat Period Predictions vs Observed Gaps")
            print("  " + "-" * 60)

            # Collect dominant beat frequencies
            observed_gaps = [zeros[i+1]['y'] - zeros[i]['y']
                             for i in range(len(zeros) - 1)]

            # Most common dominant pair
            pair_counts = {}
            for r in pair_results:
                key = r['pair']
                pair_counts[key] = pair_counts.get(key, 0) + 1

            most_common = sorted(pair_counts.items(), key=lambda x: -x[1])
            print(f"\n  Most common dominant pairs: {most_common[:5]}")

            # Compare beat periods with observed gaps
            print(f"\n  {'Pair':>8} {'Δω':>8} {'Beat T':>10} {'Obs ⟨Δy⟩':>10} "
                  f"{'Ratio':>8}")
            for (p1, p2), count in most_common[:5]:
                dw = abs(freq[p2] - freq[p1])
                bp = 2 * pi / dw if dw > 1e-10 else float('inf')
                ratio = avg_gap / bp if bp < float('inf') else float('nan')
                print(f"  ({p1},{p2}){' '*(4-len(str(p1))-len(str(p2)))} "
                      f"{dw:8.4f} {bp:10.4f} {avg_gap:10.4f} {ratio:8.4f}")

            # Test quasi-periodicity: are gaps approximately constant?
            cv = std_gap / avg_gap if avg_gap > 0 else float('inf')
            print(f"\n  Gap coefficient of variation: {cv:.4f}")
            if cv < 0.15:
                print(f"  => Near-periodic (CV < 0.15)")
            elif cv < 0.3:
                print(f"  => Quasi-periodic (0.15 < CV < 0.30)")
            else:
                print(f"  => Irregular (CV > 0.30)")

    # ======================================================================
    # Part 6: HK vs Wilson mechanism comparison
    # ======================================================================
    print(f"\n\n  {'='*80}")
    print(f"  PART 6: HK vs Wilson Mechanism Comparison")
    print(f"  {'='*80}")

    print("""
  HEAT-KERNEL MECHANISM (this analysis):
    Z_n^HK = Σ w_p exp(-i ω_p y)    [pure exponential sum]
    - Weights w_p decay super-exponentially: w_p ~ d_p^{n+1} exp(-n C_2 κ)
    - Frequencies ω_p = n C_2(p) are FIXED (independent of y, κ)
    - Zeros arise from BEAT INTERFERENCE of 2-3 dominant terms
    - |A_p^HK(κ+iy)| = d_p exp(-C_2 κ) independent of y → NO conveyor belt
    - Zero spacing ≈ beat period 2π/Δω for dominant pair
    - Almost-periodic: zeros quasi-regular but not exactly periodic
      (more than 2 incommensurate frequencies)

  WILSON MECHANISM (from earlier tasks):
    Z_n^W = Σ d_p [A_p^W(s)]^n      [Fourier transforms of exp(s·Φ)]
    - |A_p^W(κ+iy)| DEPENDS on y → conveyor belt (amplitude modulation)
    - Frequencies (phase velocities) are NONLINEAR: ω_p(y) varies with y
    - Zeros from combination of:
      (a) Saddle-point oscillation for |Φ₀| ≠ 0
      (b) Conveyor belt amplitude sweeping for all N
    - Z^W(κ+iy) → 0 as y → ∞ (Riemann-Lebesgue) → guarantees ∞ zeros
    - |Z^HK| does NOT decay (quasi-periodic oscillation)

  KEY DISTINCTION:
    HK zeros: finite-frequency beat interference (almost-periodic function)
    Wilson zeros: Riemann-Lebesgue decay + conveyor belt + saddle-point
    Both have ∞ zeros, but for DIFFERENT mathematical reasons.
""")

    elapsed = time.time() - t0
    print(f"\n  [Completed in {elapsed:.1f}s]")
    print("=" * 90)


if __name__ == '__main__':
    main()
