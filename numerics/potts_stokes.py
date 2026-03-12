#!/usr/bin/env python3
"""
Potts Model Stokes Analysis — Exact Zero Computation
=====================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: Paper Psi v3 — Potts model as bridge between exact (Ising) and
         asymptotic (SU(N)) Stokes mechanisms.

The q-state Potts transfer matrix has eigenvalues:
  λ_1(β) = e^β + q - 1  (non-degenerate)
  λ_2(β) = e^β - 1       ((q-1)-fold degenerate)

So Z_n(β) = λ_1^n + (q-1) λ_2^n.

Zero condition: (λ_1/λ_2)^n = -(q-1)
  → n log(λ_1/λ_2) = log(q-1) + i(2k+1)π,  k = 0, ..., n-1

For q=2 (Ising): β_k = (1/2) log(cot((2k+1)π/(2n))) - iπ/4
  → ALL zeros have Im(β) = -π/4 or +π/4 = Stokes lines ± π/4
  → Zeros lie EXACTLY on Stokes lines

For q>2: zeros lie on exact Stokes curves at distance O(log(q-1)/n)
  from the asymptotic Stokes network |λ_1| = |λ_2|.

Key results:
  1. For 2-term sums, zeros lie EXACTLY on the (n-dependent) Stokes curve
  2. The exact curve converges to the asymptotic network as n → ∞
  3. Spacing law Δ = 2π/(n|Δω|) is exact for 2-term sums
  4. q=2 gives exact closed-form zero positions
"""

import numpy as np
from math import pi
import time

t0 = time.time()


def potts_eigenvalues(beta, q):
    """Potts transfer matrix eigenvalues at complex coupling β."""
    eb = np.exp(beta)
    return eb + q - 1, eb - 1


# ===================================================================
print("=" * 90)
print("  Potts Model: Exact Fisher Zero Computation")
print("=" * 90)


# ===================================================================
# PART 1: Ising Model (q=2) — Exact Zero Formula
# ===================================================================

print("\n  PART 1: 1D Ising (q=2) — Exact Zero Locations")
print("  " + "-" * 70)
print("""
  For the Ising model: λ_+ = 2cosh β, λ_- = 2sinh β
  Z_n = λ_+^n + λ_-^n.   Zero condition: (coth β)^n = -1.

  Solution: β_k = (1/2) log(cot((2k+1)π/(2n))) ∓ iπ/4
  where k = 0, 1, ..., n-1  (those with cot > 0).

  KEY: All zeros have Im(β) = ±π/4, i.e., they lie EXACTLY on
  the Stokes lines |cosh β| = |sinh β| (where cos(2y) = 0).
""")

for n in [4, 10, 20, 50]:
    print(f"\n  n = {n}:")
    print(f"  {'k':>4s}  {'β_r':>10s}  {'y':>10s}  "
          f"{'|Z_n|':>12s}  {'on Stokes?':>12s}")

    zeros_br = []
    for k in range(n):
        alpha_k = (2 * k + 1) * pi / n
        cot_val = 1.0 / np.tan(alpha_k / 2)
        if cot_val > 0:
            beta_r = 0.5 * np.log(cot_val)
            y = -pi / 4
            # Verify
            beta = beta_r + 1j * y
            lp = 2 * np.cosh(beta)
            lm = 2 * np.sinh(beta)
            Zn = lp**n + lm**n
            on_stokes = abs(abs(lp) - abs(lm)) < 1e-10
            if k < 8 or k >= n - 2:
                print(f"  {k:4d}  {beta_r:10.6f}  {y:10.6f}  "
                      f"{abs(Zn):12.2e}  {'YES' if on_stokes else 'NO':>12s}")
            elif k == 8:
                print(f"  {'...':>4s}")
            zeros_br.append(beta_r)

    zeros_br.sort()
    if len(zeros_br) >= 2:
        gaps = np.diff(zeros_br)
        print(f"  Total zeros (y = -π/4): {len(zeros_br)}")
        print(f"  β_r range: [{zeros_br[0]:.4f}, {zeros_br[-1]:.4f}]")
        print(f"  ⟨Δβ_r⟩ = {np.mean(gaps):.6f}")


# ===================================================================
# PART 2: General Potts (q > 2) — Numerical Zero Finding
# ===================================================================

print("\n\n  PART 2: q-state Potts — Exact Zeros via Newton's Method")
print("  " + "-" * 70)
print("""
  Zero condition: (λ_1/λ_2)^n = -(q-1)
  Solved by Newton's method in the complex β plane.
  Then verify: each zero lies on the exact Stokes curve
    |λ_1/λ_2| = (q-1)^{1/n}.
""")


def find_potts_zeros(n, q, beta_r_range=(0.01, 3.0), y_range=(0.01, 6.5)):
    """Find Fisher zeros of q-state Potts Z_n by Newton's method.

    Zeros of Z_n(β) = λ_1(β)^n + (q-1) λ_2(β)^n.
    Uses grid of initial guesses + Newton refinement.
    """
    # Dense grid search for |Z_n| minima
    n_br, n_y = 200, 400
    br_arr = np.linspace(beta_r_range[0], beta_r_range[1], n_br)
    y_arr = np.linspace(y_range[0], y_range[1], n_y)
    BR, Y = np.meshgrid(br_arr, y_arr, indexing='ij')
    BETA = BR + 1j * Y

    L1, L2 = potts_eigenvalues(BETA, q)
    ZN = L1**n + (q - 1) * L2**n
    absZN = np.abs(ZN)

    # Find local minima of |Z_n|
    candidates = []
    for i in range(1, n_br - 1):
        for j in range(1, n_y - 1):
            if (absZN[i, j] < absZN[i-1, j] and absZN[i, j] < absZN[i+1, j]
                    and absZN[i, j] < absZN[i, j-1]
                    and absZN[i, j] < absZN[i, j+1]
                    and absZN[i, j] < 0.1 * np.median(absZN)):
                candidates.append(BETA[i, j])

    # Newton refinement
    zeros = []
    for beta0 in candidates:
        beta = beta0
        for _ in range(50):
            l1, l2 = potts_eigenvalues(beta, q)
            Z = l1**n + (q - 1) * l2**n
            # dZ/dβ = n λ_1^{n-1} dλ_1/dβ + (q-1) n λ_2^{n-1} dλ_2/dβ
            # dλ_1/dβ = dλ_2/dβ = e^β
            eb = np.exp(beta)
            dZ = n * eb * (l1**(n-1) + (q-1) * l2**(n-1))
            if abs(dZ) < 1e-30:
                break
            beta = beta - Z / dZ
            if abs(Z) < 1e-12:
                break

        l1f, l2f = potts_eigenvalues(beta, q)
        Zf = l1f**n + (q-1) * l2f**n
        if (abs(Zf) < 1e-8
                and beta.real > beta_r_range[0] - 0.1
                and beta.real < beta_r_range[1] + 0.1
                and beta.imag > y_range[0] - 0.1
                and beta.imag < y_range[1] + 0.1):
            # Check for duplicates
            is_dup = False
            for z in zeros:
                if abs(beta - z) < 1e-6:
                    is_dup = True
                    break
            if not is_dup:
                zeros.append(beta)

    return sorted(zeros, key=lambda z: (z.imag, z.real))


for q in [2, 3, 4]:
    print(f"\n  ═══════════════════ q = {q} ═══════════════════")

    for n in [4, 10, 20]:
        zeros = find_potts_zeros(n, q)
        if not zeros:
            print(f"  n={n:3d}: no zeros found in search region")
            continue

        # Verify each zero lies on exact Stokes curve
        print(f"\n  n = {n}, q = {q}: {len(zeros)} zeros found")
        print(f"  {'β_r':>10s}  {'y':>10s}  {'|Z_n|':>10s}  "
              f"{'|λ_1/λ_2|':>10s}  {'(q-1)^{1/n}':>12s}  "
              f"{'ratio err':>10s}  {'on exact S':>10s}")

        ratio_target = (q - 1) ** (1.0 / n)
        on_exact_count = 0

        for beta in zeros[:15]:
            l1, l2 = potts_eigenvalues(beta, q)
            Zn = l1**n + (q-1) * l2**n
            ratio = abs(l1 / l2) if abs(l2) > 1e-30 else float('inf')
            ratio_err = abs(ratio - ratio_target)
            on_exact = ratio_err < 1e-4
            if on_exact:
                on_exact_count += 1
            print(f"  {beta.real:10.6f}  {beta.imag:10.6f}  "
                  f"{abs(Zn):10.2e}  {ratio:10.6f}  {ratio_target:12.6f}  "
                  f"{ratio_err:10.2e}  {'YES' if on_exact else 'NO':>10s}")

        if len(zeros) > 15:
            # Count remaining
            for beta in zeros[15:]:
                l1, l2 = potts_eigenvalues(beta, q)
                ratio = abs(l1 / l2) if abs(l2) > 1e-30 else float('inf')
                if abs(ratio - ratio_target) < 1e-4:
                    on_exact_count += 1

        print(f"  → {on_exact_count}/{len(zeros)} zeros on exact Stokes "
              f"(|λ_1/λ_2| = (q-1)^{{1/n}} = {ratio_target:.6f})")

        # Distance from asymptotic Stokes |λ_1| = |λ_2|
        if q > 2:
            # For each zero, find dist to |λ_1| = |λ_2| curve
            # The asymptotic Stokes in (β_r, y) plane:
            # cos y = (2-q)/(2e^{β_r})
            asymp_dists = []
            for beta in zeros:
                br, y = beta.real, beta.imag
                cos_target = (2 - q) / (2 * np.exp(br))
                if abs(cos_target) <= 1:
                    y_asymp = np.arccos(cos_target)
                    # Nearest Stokes y (could be y_asymp or 2π-y_asymp + 2kπ)
                    d = min(abs(y - y_asymp), abs(y - (2*pi - y_asymp)),
                            abs(y + y_asymp), abs(y - y_asymp - 2*pi),
                            abs(y - (2*pi - y_asymp) - 2*pi))
                    asymp_dists.append(d)

            if asymp_dists:
                print(f"  ⟨dist to asymptotic Stokes⟩ = {np.mean(asymp_dists):.6f}")
                print(f"  Predicted shift ≈ log(q-1)/n = {np.log(q-1)/n:.6f}")


# ===================================================================
# PART 3: Stokes Convergence — Exact → Asymptotic as n → ∞
# ===================================================================

print("\n\n  PART 3: Convergence of Exact → Asymptotic Stokes")
print("  " + "-" * 70)
print("  For q > 2, the exact Stokes |λ_1/λ_2| = (q-1)^{1/n} → 1")
print("  converges to the asymptotic Stokes |λ_1| = |λ_2| as n → ∞.")
print("  The shift is O(log(q-1)/n).\n")

for q in [3, 4]:
    print(f"  q = {q}:")
    print(f"  {'n':>6s}  {'(q-1)^{1/n}':>12s}  "
          f"{'shift = 1-ratio':>16s}  {'log(q-1)/n':>12s}")
    for n in [2, 5, 10, 20, 50, 100, 500]:
        ratio = (q - 1) ** (1.0 / n)
        shift = ratio - 1
        theory = np.log(q - 1) / n
        print(f"  {n:6d}  {ratio:12.8f}  {shift:16.8f}  {theory:12.8f}")
    print()


# ===================================================================
# PART 4: Spacing Along Stokes Lines
# ===================================================================

print("\n  PART 4: Zero Spacing Along Stokes Lines")
print("  " + "-" * 70)

print("\n  Ising (q=2): zeros at β_k = (1/2)ln(cot((2k+1)π/(2n))) - iπ/4")
print("  Spacing in β_r direction along y = -π/4 Stokes line.\n")

print(f"  {'n':>5s}  {'#zeros':>7s}  {'⟨Δβ_r⟩':>10s}  {'n⟨Δβ_r⟩':>10s}")

for n in [4, 10, 20, 50, 100]:
    zeros_br = []
    for k in range(n):
        alpha_k = (2 * k + 1) * pi / n
        half_alpha = alpha_k / 2
        if 0 < half_alpha < pi / 2:  # cot > 0
            cot_val = 1.0 / np.tan(half_alpha)
            beta_r = 0.5 * np.log(cot_val)
            if 0 < beta_r < 5:  # reasonable range
                zeros_br.append(beta_r)

    zeros_br.sort()
    if len(zeros_br) >= 2:
        gaps = np.diff(zeros_br)
        avg_gap = np.mean(gaps)
        print(f"  {n:5d}  {len(zeros_br):7d}  {avg_gap:10.6f}  "
              f"{n*avg_gap:10.4f}")


# ===================================================================
# PART 5: Potts Zero Density — 2D Structure
# ===================================================================

print("\n\n  PART 5: Potts Zero Distribution in Complex β Plane")
print("  " + "-" * 70)

for q in [3, 4]:
    print(f"\n  q = {q}:")
    for n in [10, 20]:
        zeros = find_potts_zeros(n, q, y_range=(0.01, 8.0))
        if not zeros:
            continue

        # Group by approximate y-value (Stokes line)
        y_vals = [z.imag for z in zeros]
        print(f"\n  n = {n}: {len(zeros)} zeros")
        print(f"  Im(β) values of zeros: "
              f"{', '.join(f'{y:.3f}' for y in sorted(set(round(y, 2) for y in y_vals)))}")

        # Check on exact Stokes
        ratio_target = (q - 1) ** (1.0 / n)
        on_exact = 0
        for beta in zeros:
            l1, l2 = potts_eigenvalues(beta, q)
            ratio = abs(l1 / l2) if abs(l2) > 1e-30 else float('inf')
            if abs(ratio - ratio_target) < 1e-3:
                on_exact += 1
        print(f"  On exact Stokes (|λ_1/λ_2|={(q-1)}^{{1/{n}}}): "
              f"{on_exact}/{len(zeros)}")


# ===================================================================
# PART 6: Paper Summary Table
# ===================================================================

print("\n\n  PART 6: Summary for Paper Psi v3")
print("  " + "-" * 70)
print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  Model          K  Zeros on Stokes    Shift from S    c_0/c_1  │
  ├─────────────────────────────────────────────────────────────────┤
  │  Ising (q=2)    1  EXACT (all)        0 (c_0=c_1)    1        │
  │  Potts q=3      1  EXACT (all)        log(2)/n        1/2      │
  │  Potts q=4      1  EXACT (all)        log(3)/n        1/3      │
  │  Potts q=10     1  EXACT (all)        log(9)/n        1/9      │
  │  SU(4) Wilson  ~14  94% within Δ/4    O(1/n)          varies   │
  └─────────────────────────────────────────────────────────────────┘

  Key insight:
  - 2-term sums: zeros lie EXACTLY on the (n-dependent) Stokes curve
  - Shift from asymptotic Stokes = log(c_1/c_0)/n → 0 as n → ∞
  - For c_0 = c_1 (Ising): zero shift, exact asymptotic Stokes
  - For c_0 ≠ c_1 (Potts q>2): non-zero shift, converges as 1/n
""")

# Ising zero formula for the paper
print("  Ising exact formula:")
print("  β_k = (1/2) ln cot((2k+1)π/(2n)) ± iπ/4")
print("  All zeros on Stokes lines y = ±π/4 (i.e., cos(2y) = 0)")
print()

# Print explicit zeros for n=10 Ising
n = 10
print(f"  Ising n={n} zeros (Im β = -π/4):")
print(f"  {'k':>4s}  {'Re β':>12s}")
for k in range(n):
    alpha_k = (2 * k + 1) * pi / n
    half_alpha = alpha_k / 2
    if 0 < half_alpha < pi / 2:
        cot_val = 1.0 / np.tan(half_alpha)
        beta_r = 0.5 * np.log(cot_val)
        print(f"  {k:4d}  {beta_r:12.6f}")


# ===================================================================
# PART 7: Potts Analytic Stokes Lines in (β_r, y) Plane
# ===================================================================

print("\n\n  PART 7: Stokes Network in the Complex β Plane")
print("  " + "-" * 70)

print("\n  Asymptotic Stokes |λ_1| = |λ_2|:")
print(f"  |e^β + q-1|² - |e^β - 1|² = 2q e^{{β_r}} cos y + q² - 2q = 0")
print(f"  → cos y = (2-q)/(2e^{{β_r}})")

print(f"\n  {'q':>3s}  {'β_r=0.5':>12s}  {'β_r=1.0':>12s}  {'β_r=2.0':>12s}")

for q in [2, 3, 4, 6, 10]:
    vals = []
    for br in [0.5, 1.0, 2.0]:
        rhs = (2 - q) / (2 * np.exp(br))
        if abs(rhs) <= 1:
            y_st = np.arccos(rhs)
            vals.append(f"{y_st:.4f}")
        else:
            vals.append("none")
    print(f"  {q:3d}  {vals[0]:>12s}  {vals[1]:>12s}  {vals[2]:>12s}")


print(f"\n\n  [Completed in {time.time()-t0:.1f}s]")
print("=" * 90)
