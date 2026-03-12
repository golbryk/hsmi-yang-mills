"""
Level Crossing Statistics of Fisher Zeros
===========================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 49 — Universality deep dive Phase 2

Collects Fisher zeros from Tasks 45-48, computes spacing distribution P(s)
and r-statistic. Compares with:
  - Poisson: r ≈ 0.386, P(s) = exp(-s)
  - GOE: r ≈ 0.536, P(s) ≈ (π/2)s exp(-πs²/4)
  - Picket-fence: r = 1, P(s) = δ(s-1)

Also computes level crossing gap statistics — spacing between |A_p|=|A_{p+1}|
crossings along scan lines.

Depends on: Tasks 45, 47 results (zeros across actions and N values).
"""

import numpy as np
from math import comb, pi
import time
import sys


# ---------------------------------------------------------------------------
# SU(N) infrastructure
# ---------------------------------------------------------------------------

def dim_rep(p, N):
    return comb(p + N - 1, N - 1)


def casimir_suN(p, N):
    return p * (p + N) / float(N)


def h_p_vec(p, z):
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
    dim = N - 1
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
    measure = w * V2 / (2 * np.pi) ** dim
    norm = np.sum(measure).real
    measure = measure / norm
    return z, theta_all, measure


# ---------------------------------------------------------------------------
# Z evaluation + zero finding by sign changes
# ---------------------------------------------------------------------------

def compute_Z_scan(Phi_action, measure, hp_list, dims, n_reps, n_plaq,
                   kappa, y_values):
    exp_kPhi = np.exp(kappa * Phi_action)
    Z_vals = np.zeros(len(y_values), dtype=complex)
    for iy, y in enumerate(y_values):
        weighted = exp_kPhi * np.exp(1j * y * Phi_action) * measure
        Z = 0j
        for p in range(n_reps):
            Ap = np.sum(hp_list[p] * weighted)
            Z += dims[p] * Ap ** n_plaq
        Z_vals[iy] = Z
    return Z_vals


def find_approximate_zeros(Z_vals, y_values):
    """Find y-locations where Re Z changes sign (approximate Fisher zeros)."""
    zeros = []
    for i in range(len(Z_vals) - 1):
        if Z_vals[i].real * Z_vals[i+1].real < 0:
            frac = abs(Z_vals[i].real) / (abs(Z_vals[i].real) + abs(Z_vals[i+1].real))
            y_zero = y_values[i] + frac * (y_values[i+1] - y_values[i])
            zeros.append(y_zero)
    return np.array(zeros)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_r_statistic(gaps):
    """Compute average ratio of consecutive gaps: r = ⟨min/max⟩."""
    if len(gaps) < 2:
        return float('nan')
    ratios = []
    for i in range(len(gaps) - 1):
        mn = min(gaps[i], gaps[i+1])
        mx = max(gaps[i], gaps[i+1])
        if mx > 1e-10:
            ratios.append(mn / mx)
    return np.mean(ratios) if ratios else float('nan')


def compute_spacing_distribution(gaps, n_bins=20):
    """Compute P(s) where s = Δy/⟨Δy⟩ (unfolded spacings)."""
    if len(gaps) < 3:
        return None, None
    avg = np.mean(gaps)
    s = np.array(gaps) / avg
    bin_edges = np.linspace(0, max(3.0, np.max(s) + 0.1), n_bins + 1)
    hist, edges = np.histogram(s, bins=bin_edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def poisson_ps(s):
    """Poisson P(s) = exp(-s)."""
    return np.exp(-s)


def goe_ps(s):
    """GOE Wigner surmise P(s) = (π/2) s exp(-π s²/4)."""
    return (pi / 2) * s * np.exp(-pi * s**2 / 4)


# ---------------------------------------------------------------------------
# Level crossing gaps
# ---------------------------------------------------------------------------

def compute_level_crossings(Phi_action, measure, hp_list, n_reps, kappa, y_values):
    """Find |A_p| = |A_{p+1}| crossings along y for fixed κ."""
    exp_kPhi = np.exp(kappa * Phi_action)
    n_y = len(y_values)

    # Compute |A_p(κ+iy)| for each y
    absAp = np.zeros((n_reps, n_y))
    for iy, y in enumerate(y_values):
        weighted = exp_kPhi * np.exp(1j * y * Phi_action) * measure
        for p in range(n_reps):
            absAp[p, iy] = abs(np.sum(hp_list[p] * weighted))

    # Find crossings
    all_crossings = []
    for p in range(min(10, n_reps - 1)):
        diff = absAp[p] - absAp[p + 1]
        for iy in range(n_y - 1):
            if diff[iy] * diff[iy + 1] < 0:
                frac = abs(diff[iy]) / (abs(diff[iy]) + abs(diff[iy + 1]))
                y_cross = y_values[iy] + frac * (y_values[iy + 1] - y_values[iy])
                all_crossings.append({'y': y_cross, 'p': p})

    all_crossings.sort(key=lambda x: x['y'])
    return all_crossings


# ---------------------------------------------------------------------------
# HK zeros
# ---------------------------------------------------------------------------

def find_hk_zeros_approx(N, n_plaq, p_max, kappa, y_values):
    """Find HK Fisher zeros by Re Z sign changes."""
    Z_vals = np.zeros(len(y_values), dtype=complex)
    for iy, y in enumerate(y_values):
        s = kappa + 1j * y
        val = 0j
        for p in range(p_max + 1):
            dp = dim_rep(p, N)
            c2 = casimir_suN(p, N)
            val += dp ** (n_plaq + 1) * np.exp(-n_plaq * c2 * s)
        Z_vals[iy] = val

    return find_approximate_zeros(Z_vals, y_values)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print()
    print("=" * 90)
    print("  Level Crossing Statistics of Fisher Zeros — Task 49")
    print("=" * 90)

    kappa = 1.0
    n_plaq = 2
    y_values = np.linspace(0.1, 25.0, 2000)

    # ======================================================================
    # PART 1: Collect zeros across actions (SU(4))
    # ======================================================================
    print(f"\n  PART 1: Collect Fisher Zeros Across Actions (SU(4))")
    print("  " + "-" * 70)

    N = 4
    n_quad = 40
    n_reps = 14

    print(f"  Building Weyl grid...", end=" ", flush=True)
    z, theta_all, measure = build_weyl_grid(N, n_quad)
    Phi1 = np.sum(np.cos(theta_all), axis=1)
    Phi2 = np.sum(np.cos(2 * theta_all), axis=1)
    hp_list = [h_p_vec(p, z) for p in range(n_reps)]
    dims = [dim_rep(p, N) for p in range(n_reps)]
    print("Done.")

    actions = {
        'Wilson': Phi1,
        'Symanzik': (5.0/3.0) * Phi1 + (-1.0/12.0) * Phi2,
        'Iwasaki': 3.648 * Phi1 + (-0.331) * Phi2,
        'DBW2': 12.2688 * Phi1 + (-1.4086) * Phi2,
    }

    action_zeros = {}
    for act_name, Phi_action in actions.items():
        Z_vals = compute_Z_scan(Phi_action, measure, hp_list, dims, n_reps,
                                n_plaq, kappa, y_values)
        zeros = find_approximate_zeros(Z_vals, y_values)
        action_zeros[act_name] = zeros
        print(f"    {act_name}: {len(zeros)} zeros in y ∈ [0.1, 25]")

    # HK zeros
    hk_zeros = find_hk_zeros_approx(N, n_plaq, 100, kappa, y_values)
    action_zeros['Heat-kernel'] = hk_zeros
    print(f"    Heat-kernel: {len(hk_zeros)} zeros")

    # ======================================================================
    # PART 2: r-statistic for each action
    # ======================================================================
    print(f"\n\n  PART 2: r-Statistic (Ratio of Consecutive Spacings)")
    print("  " + "-" * 70)
    print(f"\n  Reference: Poisson r = 0.386, GOE r = 0.536, picket-fence r = 1.000")
    print(f"\n  {'Action':<14} {'#zeros':>7} {'⟨Δy⟩':>8} {'r':>8} {'σ(r)':>8} {'Match':>12}")
    print(f"  {'-'*14} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")

    for act_name, zeros in action_zeros.items():
        if len(zeros) < 3:
            print(f"  {act_name:<14} {len(zeros):7d} {'N/A':>8} {'N/A':>8} "
                  f"{'N/A':>8} {'N/A':>12}")
            continue

        gaps = np.diff(zeros)
        avg_gap = np.mean(gaps)
        r = compute_r_statistic(gaps)

        # Classify
        if r > 0.85:
            match = "picket-fence"
        elif r > 0.48:
            match = "GOE-like"
        elif r > 0.35:
            match = "Poisson"
        else:
            match = "clustered"

        # Bootstrap r uncertainty
        r_samples = []
        for _ in range(200):
            idx = np.random.choice(len(gaps) - 1, size=len(gaps) - 1, replace=True)
            g_boot = gaps[idx]
            r_samples.append(compute_r_statistic(g_boot))
        sigma_r = np.std(r_samples)

        print(f"  {act_name:<14} {len(zeros):7d} {avg_gap:8.4f} {r:8.4f} "
              f"{sigma_r:8.4f} {match:>12}")

    # ======================================================================
    # PART 3: r-statistic across N values (Wilson only)
    # ======================================================================
    print(f"\n\n  PART 3: r-Statistic vs N (Wilson Action)")
    print("  " + "-" * 70)

    N_values = [3, 4, 5]
    n_quad_map = {3: 60, 4: 40, 5: 30}
    n_reps_map = {3: 12, 4: 14, 5: 12}

    y_scan_long = np.linspace(0.1, 30.0, 3000)

    print(f"\n  {'N':>3} {'N mod 4':>7} {'#zeros':>7} {'⟨Δy⟩':>8} "
          f"{'r':>8} {'Match':>12}")
    print(f"  {'-'*3} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*12}")

    for N_test in N_values:
        nq = n_quad_map[N_test]
        nr = n_reps_map[N_test]

        z_t, theta_t, meas_t = build_weyl_grid(N_test, nq)
        Phi_t = np.sum(np.cos(theta_t), axis=1)
        hp_t = [h_p_vec(p, z_t) for p in range(nr)]
        dims_t = [dim_rep(p, N_test) for p in range(nr)]

        Z_vals = compute_Z_scan(Phi_t, meas_t, hp_t, dims_t, nr,
                                n_plaq, kappa, y_scan_long)
        zeros = find_approximate_zeros(Z_vals, y_scan_long)

        if len(zeros) >= 3:
            gaps = np.diff(zeros)
            avg_gap = np.mean(gaps)
            r = compute_r_statistic(gaps)
            match = ("picket-fence" if r > 0.85 else
                     "GOE-like" if r > 0.48 else
                     "Poisson" if r > 0.35 else "clustered")
        else:
            avg_gap = float('nan')
            r = float('nan')
            match = "N/A"

        print(f"  {N_test:3d} {N_test % 4:7d} {len(zeros):7d} {avg_gap:8.4f} "
              f"{r:8.4f} {match:>12}")

    # ======================================================================
    # PART 4: Spacing distribution P(s)
    # ======================================================================
    print(f"\n\n  PART 4: Spacing Distribution P(s)")
    print("  " + "-" * 70)
    print(f"  s = Δy / ⟨Δy⟩ (unfolded spacing)")

    for act_name, zeros in action_zeros.items():
        if len(zeros) < 5:
            continue

        gaps = np.diff(zeros)
        centers, hist = compute_spacing_distribution(gaps, n_bins=15)
        if centers is None:
            continue

        print(f"\n  {act_name} (n={len(zeros)} zeros):")
        print(f"  {'s':>6} {'P(s)':>8} {'Poisson':>10} {'GOE':>10}")
        for i in range(len(centers)):
            s = centers[i]
            p_obs = hist[i]
            p_poi = poisson_ps(s)
            p_goe = goe_ps(s)
            bar = '#' * int(p_obs * 10)
            print(f"  {s:6.2f} {p_obs:8.3f} {p_poi:10.3f} {p_goe:10.3f}  {bar}")

    # ======================================================================
    # PART 5: Level crossing gap statistics
    # ======================================================================
    print(f"\n\n  PART 5: Level Crossing Gap Statistics (SU(4), Wilson)")
    print("  " + "-" * 70)

    y_lc = np.linspace(0.1, 15.0, 1000)
    crossings = compute_level_crossings(Phi1, measure, hp_list, n_reps,
                                        kappa, y_lc)

    if len(crossings) >= 3:
        y_cross = [c['y'] for c in crossings]
        lc_gaps = np.diff(y_cross)
        avg_lc = np.mean(lc_gaps)
        r_lc = compute_r_statistic(lc_gaps)

        print(f"  Total crossings: {len(crossings)}")
        print(f"  Average gap: {avg_lc:.4f}")
        print(f"  r-statistic: {r_lc:.4f}")

        # Breakdown by pair
        pair_counts = {}
        for c in crossings:
            p = c['p']
            key = f"|A_{p}|=|A_{p+1}|"
            pair_counts[key] = pair_counts.get(key, 0) + 1

        print(f"\n  Crossing breakdown:")
        for key, count in sorted(pair_counts.items()):
            print(f"    {key}: {count}")

    # ======================================================================
    # PART 6: HK r-statistic
    # ======================================================================
    print(f"\n\n  PART 6: HK Spacing Statistics")
    print("  " + "-" * 70)

    for N_test in [3, 4]:
        y_hk = np.linspace(0.1, 40.0, 4000)
        hk_z = find_hk_zeros_approx(N_test, n_plaq, 100, kappa, y_hk)

        if len(hk_z) >= 3:
            gaps = np.diff(hk_z)
            avg_gap = np.mean(gaps)
            r = compute_r_statistic(gaps)
            match = ("picket-fence" if r > 0.85 else
                     "GOE-like" if r > 0.48 else
                     "Poisson" if r > 0.35 else "clustered")
            print(f"  SU({N_test}) HK: {len(hk_z)} zeros, ⟨Δy⟩={avg_gap:.4f}, "
                  f"r={r:.4f} ({match})")

    # ======================================================================
    # Summary
    # ======================================================================
    elapsed = time.time() - t0

    print(f"\n\n  {'='*90}")
    print(f"  SUMMARY")
    print(f"  {'='*90}")
    print(f"""
  r-STATISTIC CLASSIFICATION:
    Poisson (uncorrelated):        r ≈ 0.386
    GOE (level repulsion):         r ≈ 0.536
    Picket-fence (rigid spacing):  r = 1.000

  RESULTS:
    - Actions with strong conveyor belt → near-regular spacing (r → 1?)
    - Actions with weak belt (DBW2) → more irregular (r closer to Poisson?)
    - HK zeros from beat interference → specific r value
    - Wilson vs HK: different mechanisms → different statistics?

  SPACING DISTRIBUTION:
    - Poisson: P(s) = exp(-s) (exponential)
    - GOE: P(s) ~ s exp(-πs²/4) (level repulsion at s=0)
    - Picket-fence: P(s) = δ(s-1)
""")
    print(f"  [Completed in {elapsed:.1f}s]")
    print("=" * 90)


if __name__ == '__main__':
    main()
