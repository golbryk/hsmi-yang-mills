"""
Spacing Universality Law: C_f = ⟨Δy⟩ · n
==========================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Tasks 50-52 — Three critical tests

Three tests:
  (1) Is C_f = 2π/σ_Φ or similar? (C_f vs distribution of f(U))
  (2) Does ⟨Δy⟩·n = const hold for n = 5,6,8? (extend beyond n=4)
  (3) Does C_f(N) have a thermodynamic limit? (N = 3..16 via MC)

PART 1: Compute σ_Φ, range(Φ), ⟨|Φ|⟩ for all actions, compare with C_f.
PART 2: n = 2,3,4,5,6,8 sign-change count for Wilson and Symanzik SU(4).
PART 3: Haar MC for N = 8,12,16: Z_n(κ+iy) sign changes, ⟨Δy⟩·n.
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
# Sign change count (for Z_n on scan line)
# ---------------------------------------------------------------------------

def count_sign_changes_scan(Phi_action, measure, hp_list, dims, n_reps,
                            n_plaq, kappa, y_values):
    """Count Re Z sign changes along y scan line."""
    exp_kPhi = np.exp(kappa * Phi_action)
    count = 0
    prev_reZ = None
    for y in y_values:
        weighted = exp_kPhi * np.exp(1j * y * Phi_action) * measure
        Z = 0j
        for p in range(n_reps):
            Ap = np.sum(hp_list[p] * weighted)
            Z += dims[p] * Ap ** n_plaq
        if prev_reZ is not None and prev_reZ * Z.real < 0:
            count += 1
        prev_reZ = Z.real
    return count


def find_zero_gaps(Phi_action, measure, hp_list, dims, n_reps, n_plaq,
                   kappa, y_values):
    """Find approximate zero y-locations and return gaps."""
    exp_kPhi = np.exp(kappa * Phi_action)
    Z_vals = np.zeros(len(y_values), dtype=complex)
    for iy, y in enumerate(y_values):
        weighted = exp_kPhi * np.exp(1j * y * Phi_action) * measure
        Z = 0j
        for p in range(n_reps):
            Ap = np.sum(hp_list[p] * weighted)
            Z += dims[p] * Ap ** n_plaq
        Z_vals[iy] = Z

    zeros = []
    for i in range(len(Z_vals) - 1):
        if Z_vals[i].real * Z_vals[i+1].real < 0:
            frac = abs(Z_vals[i].real) / (abs(Z_vals[i].real) + abs(Z_vals[i+1].real))
            zeros.append(y_values[i] + frac * (y_values[i+1] - y_values[i]))

    if len(zeros) >= 2:
        return np.diff(zeros)
    return np.array([])


# ---------------------------------------------------------------------------
# Haar MC sampling for large N
# ---------------------------------------------------------------------------

def haar_eigenvalues(N, n_samples):
    """Generate Haar-random SU(N) eigenvalues via QR decomposition."""
    thetas = np.zeros((n_samples, N))
    for i in range(n_samples):
        # Random complex Gaussian matrix
        G = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2)
        Q, R = np.linalg.qr(G)
        # Fix diagonal phases of R
        d = np.diagonal(R)
        ph = d / np.abs(d)
        Q = Q * ph[np.newaxis, :]
        # Eigenvalues of Q are e^{i theta_j}
        eigs = np.linalg.eigvals(Q)
        angles = np.angle(eigs)
        # Project to SU(N): subtract mean phase
        angles = angles - np.mean(angles)
        thetas[i] = angles
    return thetas


def mc_Z_scan(N, n_plaq, kappa, y_values, n_samples, n_reps,
              action_type='wilson'):
    """Monte Carlo computation of Z_n(κ+iy) via Haar sampling."""
    thetas = haar_eigenvalues(N, n_samples)
    z = np.exp(1j * thetas)  # (n_samples, N)

    # Precompute h_p for all samples
    hp = np.zeros((n_reps, n_samples), dtype=complex)
    for p in range(n_reps):
        hp[p] = h_p_vec_flat(p, z)

    # Compute Φ based on action
    if action_type == 'wilson':
        Phi = np.sum(np.cos(thetas), axis=1)  # Re Tr U
    elif action_type == 'symanzik':
        Phi1 = np.sum(np.cos(thetas), axis=1)
        Phi2 = np.sum(np.cos(2 * thetas), axis=1)
        Phi = (5.0/3.0) * Phi1 + (-1.0/12.0) * Phi2
    else:
        Phi = np.sum(np.cos(thetas), axis=1)

    dims = [dim_rep(p, N) for p in range(n_reps)]

    # Vandermonde
    V2 = np.ones(n_samples)
    for j in range(N):
        for k in range(j + 1, N):
            V2 *= np.abs(z[:, j] - z[:, k]) ** 2

    # Weights (Haar measure already built in by sampling, but need Vandermonde
    # correction since we sample from U(N) not SU(N) Haar with Vandermonde)
    # Actually, QR gives Haar-uniform U(N), eigenvalues already have the
    # Vandermonde factor absorbed. So weights = 1/n_samples.
    # But we need exp(κ Φ) weighting for the Boltzmann factor.
    weight = np.exp(kappa * Phi) / n_samples

    # For each y, compute Z_n
    Z_vals = np.zeros(len(y_values), dtype=complex)
    for iy, y in enumerate(y_values):
        w_complex = weight * np.exp(1j * y * Phi)
        Z = 0j
        for p in range(n_reps):
            Ap = np.sum(hp[p] * w_complex)
            Z += dims[p] * Ap ** n_plaq
        Z_vals[iy] = Z

    return Z_vals


def h_p_vec_flat(p, z):
    """h_p for z of shape (n_samples, N)."""
    n_pts = z.shape[0]
    if p == 0:
        return np.ones(n_pts, dtype=complex)
    psums = np.array([np.sum(z ** k, axis=1) for k in range(1, p + 1)])
    h = np.zeros((p + 1, n_pts), dtype=complex)
    h[0] = 1.0
    for k in range(1, p + 1):
        h[k] = sum(psums[j] * h[k - 1 - j] for j in range(k)) / k
    return h[p]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print()
    print("=" * 90)
    print("  Spacing Universality Law: C_f = ⟨Δy⟩ · n")
    print("  Three Critical Tests")
    print("=" * 90)

    # ======================================================================
    # PART 1: C_f vs σ_Φ relationship
    # ======================================================================
    print(f"\n  PART 1: C_f vs Φ-Distribution Statistics")
    print("  " + "-" * 70)

    N = 4
    n_quad = 40
    n_reps = 14
    kappa = 1.0

    print(f"  Building SU({N}) Weyl grid...", end=" ", flush=True)
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
        'Alpha(0.5)': np.sign(Phi1) * np.abs(Phi1) ** 0.5,
        'Alpha(2.0)': np.sign(Phi1) * np.abs(Phi1) ** 2.0,
    }

    # Get C_f from n=2,3,4 (Task 48 results)
    # Recompute quickly: just get ⟨Δy⟩ for n=2 and multiply by 2
    y_scan = np.linspace(0.1, 25.0, 2000)

    print(f"\n  Computing C_f = ⟨Δy⟩·n for each action at n=2...")
    sys.stdout.flush()

    Cf_data = {}
    for act_name, Phi_action in actions.items():
        gaps = find_zero_gaps(Phi_action, measure, hp_list, dims, n_reps,
                              2, kappa, y_scan)
        if len(gaps) >= 1:
            Cf = np.mean(gaps) * 2  # n=2, so C_f = ⟨Δy⟩ · 2
            Cf_data[act_name] = Cf
        else:
            Cf_data[act_name] = float('nan')
        print(f"    {act_name}: {len(gaps)+1} zeros, "
              f"⟨Δy⟩={np.mean(gaps) if len(gaps)>0 else 0:.4f}, "
              f"C_f={Cf_data[act_name]:.4f}")

    # Compute Φ statistics under Haar measure
    print(f"\n  Φ-distribution statistics under HAAR measure:")
    print(f"  {'Action':<14} {'σ_Φ':>8} {'range':>8} {'⟨|Φ|⟩':>8} "
          f"{'⟨Φ²⟩':>8} {'σ²':>8}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    phi_stats = {}
    for act_name, Phi_action in actions.items():
        mean_phi = np.sum(Phi_action * measure)
        var_phi = np.sum(Phi_action**2 * measure) - mean_phi**2
        sigma = np.sqrt(max(0, var_phi))
        abs_mean = np.sum(np.abs(Phi_action) * measure)
        rng = np.max(Phi_action) - np.min(Phi_action)
        mean_sq = np.sum(Phi_action**2 * measure)
        phi_stats[act_name] = {
            'sigma': sigma, 'range': rng, 'abs_mean': abs_mean,
            'mean_sq': mean_sq, 'var': var_phi}
        print(f"  {act_name:<14} {sigma:8.4f} {rng:8.4f} {abs_mean:8.4f} "
              f"{mean_sq:8.4f} {var_phi:8.4f}")

    # Compute Φ statistics under BOLTZMANN measure exp(κΦ)dU/Z
    print(f"\n  Φ-distribution statistics under BOLTZMANN measure exp({kappa}·Φ):")
    print(f"  {'Action':<14} {'σ_B':>8} {'⟨Φ⟩_B':>8} {'⟨Φ²⟩_B':>8}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8}")

    boltz_stats = {}
    for act_name, Phi_action in actions.items():
        w = np.exp(kappa * Phi_action) * measure
        Z_norm = np.sum(w)
        w_norm = w / Z_norm
        mean_B = np.sum(Phi_action * w_norm)
        mean2_B = np.sum(Phi_action**2 * w_norm)
        var_B = mean2_B - mean_B**2
        sigma_B = np.sqrt(max(0, var_B))
        boltz_stats[act_name] = {'sigma': sigma_B, 'mean': mean_B, 'var': var_B}
        print(f"  {act_name:<14} {sigma_B:8.4f} {mean_B:8.4f} {mean2_B:8.4f}")

    # Test hypotheses
    print(f"\n\n  HYPOTHESIS TESTING: C_f = f(Φ-statistics)")
    print("  " + "-" * 70)

    print(f"\n  {'Action':<14} {'C_f':>8} {'2π/σ_H':>8} {'2π/σ_B':>8} "
          f"{'2π/⟨|Φ|⟩':>9} {'π/σ_H':>8} {'π/σ_B':>8} {'4π/range':>9}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*9}")

    for act_name in actions:
        Cf = Cf_data[act_name]
        sH = phi_stats[act_name]['sigma']
        sB = boltz_stats[act_name]['sigma']
        am = phi_stats[act_name]['abs_mean']
        rng = phi_stats[act_name]['range']

        h1 = 2*pi/sH if sH > 0 else float('inf')
        h2 = 2*pi/sB if sB > 0 else float('inf')
        h3 = 2*pi/am if am > 0 else float('inf')
        h4 = pi/sH if sH > 0 else float('inf')
        h5 = pi/sB if sB > 0 else float('inf')
        h6 = 4*pi/rng if rng > 0 else float('inf')

        print(f"  {act_name:<14} {Cf:8.4f} {h1:8.4f} {h2:8.4f} "
              f"{h3:9.4f} {h4:8.4f} {h5:8.4f} {h6:9.4f}")

    # Compute ratios to find best fit
    print(f"\n  Ratio C_f / prediction (closest to 1.0 wins):")
    print(f"  {'Action':<14} {'C/[2π/σH]':>10} {'C/[2π/σB]':>10} "
          f"{'C/[π/σB]':>10} {'C/[4π/rng]':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for act_name in actions:
        Cf = Cf_data[act_name]
        sH = phi_stats[act_name]['sigma']
        sB = boltz_stats[act_name]['sigma']
        rng = phi_stats[act_name]['range']
        if np.isnan(Cf) or Cf == 0:
            continue

        r1 = Cf / (2*pi/sH) if sH > 0 else float('nan')
        r2 = Cf / (2*pi/sB) if sB > 0 else float('nan')
        r3 = Cf / (pi/sB) if sB > 0 else float('nan')
        r4 = Cf / (4*pi/rng) if rng > 0 else float('nan')

        print(f"  {act_name:<14} {r1:10.4f} {r2:10.4f} "
              f"{r3:10.4f} {r4:10.4f}")

    # ======================================================================
    # PART 2: Larger n (n = 2,3,4,5,6,8) for Wilson and Symanzik
    # ======================================================================
    print(f"\n\n  {'='*90}")
    print(f"  PART 2: ⟨Δy⟩·n = const for n = 2,3,4,5,6,8")
    print("  " + "-" * 70)

    n_values = [2, 3, 4, 5, 6, 8]
    y_scan_dense = np.linspace(0.1, 30.0, 3000)

    test_actions = {'Wilson': Phi1,
                    'Symanzik': (5.0/3.0) * Phi1 + (-1.0/12.0) * Phi2}

    print(f"\n  {'Action':<12} {'n':>3} {'#sign_ch':>9} {'⟨Δy⟩':>8} "
          f"{'⟨Δy⟩·n':>8} {'σ(Δy)':>8} {'CV':>6}")
    print(f"  {'-'*12} {'-'*3} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    Cf_by_n = {}

    for act_name, Phi_action in test_actions.items():
        for n_plaq in n_values:
            print(f"  {act_name:<12} {n_plaq:3d}", end="", flush=True)

            gaps = find_zero_gaps(Phi_action, measure, hp_list, dims, n_reps,
                                  n_plaq, kappa, y_scan_dense)
            n_zeros = len(gaps) + 1 if len(gaps) > 0 else 0
            sc = count_sign_changes_scan(Phi_action, measure, hp_list, dims,
                                         n_reps, n_plaq, kappa,
                                         np.linspace(0.1, 30.0, 1000))

            if len(gaps) >= 2:
                avg = np.mean(gaps)
                std = np.std(gaps)
                cv = std / avg
                Cf_n = avg * n_plaq
                Cf_by_n[(act_name, n_plaq)] = Cf_n
                print(f" {sc:9d} {avg:8.4f} {Cf_n:8.4f} {std:8.4f} {cv:6.3f}")
            elif len(gaps) == 1:
                avg = gaps[0]
                Cf_n = avg * n_plaq
                Cf_by_n[(act_name, n_plaq)] = Cf_n
                print(f" {sc:9d} {avg:8.4f} {Cf_n:8.4f} {'—':>8} {'—':>6}")
            else:
                Cf_by_n[(act_name, n_plaq)] = float('nan')
                print(f" {sc:9d} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>6}")

    # Check constancy
    print(f"\n  Constancy test:")
    for act_name in test_actions:
        vals = [Cf_by_n[(act_name, n)] for n in n_values
                if not np.isnan(Cf_by_n.get((act_name, n), float('nan')))]
        if len(vals) >= 3:
            avg_Cf = np.mean(vals)
            std_Cf = np.std(vals)
            cv_Cf = std_Cf / avg_Cf
            print(f"  {act_name}: C_f values = {['%.3f' % v for v in vals]}")
            print(f"    ⟨C_f⟩ = {avg_Cf:.4f} ± {std_Cf:.4f} (CV = {cv_Cf:.3f})")
            if cv_Cf < 0.05:
                print(f"    => EXCELLENT constancy (CV < 5%)")
            elif cv_Cf < 0.10:
                print(f"    => GOOD constancy (CV < 10%)")
            elif cv_Cf < 0.20:
                print(f"    => FAIR constancy (CV < 20%)")
            else:
                print(f"    => POOR constancy (CV > 20%)")

    # ======================================================================
    # PART 3: C_f(N) for N = 3,4,5,6 (Weyl) and N = 8,12,16 (MC)
    # ======================================================================
    print(f"\n\n  {'='*90}")
    print(f"  PART 3: C_f(N) — Does C_f Have a Thermodynamic Limit?")
    print("  " + "-" * 70)

    # Small N via Weyl quadrature
    N_weyl = [3, 4, 5, 6]
    n_quad_map = {3: 60, 4: 40, 5: 25, 6: 18}
    n_reps_map = {3: 10, 4: 14, 5: 10, 6: 8}
    n_plaq = 2

    print(f"\n  Weyl quadrature for N = {N_weyl}:")
    print(f"  {'N':>3} {'N mod 4':>7} {'|Φ₀|':>5} {'#zeros':>7} {'⟨Δy⟩':>8} "
          f"{'C_f':>8} {'σ_Φ':>8} {'C_f·σ_Φ':>10}")
    print(f"  {'-'*3} {'-'*7} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    Cf_N = {}

    for N_test in N_weyl:
        nq = n_quad_map[N_test]
        nr = n_reps_map[N_test]

        z_t, theta_t, meas_t = build_weyl_grid(N_test, nq)
        Phi_t = np.sum(np.cos(theta_t), axis=1)  # Wilson
        hp_t = [h_p_vec(p, z_t) for p in range(nr)]
        dims_t = [dim_rep(p, N_test) for p in range(nr)]

        # σ_Φ under Haar
        mean_phi = np.sum(Phi_t * meas_t)
        var_phi = np.sum(Phi_t**2 * meas_t) - mean_phi**2
        sigma_phi = np.sqrt(max(0, var_phi))

        y_scan_N = np.linspace(0.1, 30.0, 3000)
        gaps = find_zero_gaps(Phi_t, meas_t, hp_t, dims_t, nr,
                              n_plaq, kappa, y_scan_N)

        Phi0 = {3: 1, 4: 0, 5: 1, 6: 2}.get(N_test, 0)

        if len(gaps) >= 1:
            avg_gap = np.mean(gaps)
            Cf = avg_gap * n_plaq
            Cf_N[N_test] = Cf
            print(f"  {N_test:3d} {N_test%4:7d} {Phi0:5d} {len(gaps)+1:7d} "
                  f"{avg_gap:8.4f} {Cf:8.4f} {sigma_phi:8.4f} "
                  f"{Cf * sigma_phi:10.4f}")
        else:
            Cf_N[N_test] = float('nan')
            print(f"  {N_test:3d} {N_test%4:7d} {Phi0:5d} {'<2':>7} "
                  f"{'N/A':>8} {'N/A':>8} {sigma_phi:8.4f} {'N/A':>10}")

    # Large N via Haar MC
    print(f"\n  Haar MC for large N (Wilson, n={n_plaq}, κ={kappa}):")
    print(f"  (n_samples=50000, n_reps adapted)")

    N_mc = [8, 12, 16]
    n_samples = 50000
    n_reps_mc = {8: 6, 12: 5, 16: 4}
    y_mc = np.linspace(0.1, 30.0, 1000)

    print(f"\n  {'N':>3} {'n_reps':>6} {'#sign_ch':>9} {'⟨Δy⟩':>8} "
          f"{'C_f':>8} {'σ_Φ_th':>8} {'C_f·σ_Φ':>10}")
    print(f"  {'-'*3} {'-'*6} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for N_test in N_mc:
        nr = n_reps_mc[N_test]
        print(f"  {N_test:3d}", end="", flush=True)

        # σ_Φ theoretical for Wilson: Var(Re Tr U) = N/2 under Haar
        # Actually Var(Σ cos θ_j) = N·Var(cos θ) = N/2 for i.i.d.
        # But SU(N) constraint reduces this slightly.
        # For now compute empirically
        thetas = haar_eigenvalues(N_test, n_samples)
        Phi_mc = np.sum(np.cos(thetas), axis=1)
        sigma_mc = np.std(Phi_mc)

        # Compute Z_n via MC
        z_mc = np.exp(1j * thetas)
        hp_mc = np.zeros((nr, n_samples), dtype=complex)
        for p in range(nr):
            hp_mc[p] = h_p_vec_flat(p, z_mc)
        dims_mc = [dim_rep(p, N_test) for p in range(nr)]

        exp_kPhi = np.exp(kappa * Phi_mc)

        Z_vals_mc = np.zeros(len(y_mc), dtype=complex)
        for iy, y in enumerate(y_mc):
            w = exp_kPhi * np.exp(1j * y * Phi_mc) / n_samples
            Z = 0j
            for p in range(nr):
                Ap = np.sum(hp_mc[p] * w)
                Z += dims_mc[p] * Ap ** n_plaq
            Z_vals_mc[iy] = Z

        # Count sign changes
        sc = 0
        for i in range(len(Z_vals_mc) - 1):
            if Z_vals_mc[i].real * Z_vals_mc[i+1].real < 0:
                sc += 1

        # Gaps
        zeros_mc = []
        for i in range(len(Z_vals_mc) - 1):
            if Z_vals_mc[i].real * Z_vals_mc[i+1].real < 0:
                frac = abs(Z_vals_mc[i].real) / (abs(Z_vals_mc[i].real) +
                                                   abs(Z_vals_mc[i+1].real))
                zeros_mc.append(y_mc[i] + frac * (y_mc[i+1] - y_mc[i]))

        if len(zeros_mc) >= 2:
            gaps_mc = np.diff(zeros_mc)
            avg_gap = np.mean(gaps_mc)
            Cf = avg_gap * n_plaq
            Cf_N[N_test] = Cf
            print(f" {nr:6d} {sc:9d} {avg_gap:8.4f} {Cf:8.4f} "
                  f"{sigma_mc:8.4f} {Cf * sigma_mc:10.4f}")
        else:
            Cf_N[N_test] = float('nan')
            print(f" {nr:6d} {sc:9d} {'N/A':>8} {'N/A':>8} "
                  f"{sigma_mc:8.4f} {'N/A':>10}")

    # C_f(N) summary
    print(f"\n\n  C_f(N) Summary (Wilson, n={n_plaq}):")
    print(f"  {'N':>3} {'C_f':>8}")
    for N_test in sorted(Cf_N.keys()):
        if not np.isnan(Cf_N.get(N_test, float('nan'))):
            print(f"  {N_test:3d} {Cf_N[N_test]:8.4f}")

    # Check if C_f has a limit
    vals_sorted = [(N_t, Cf_N[N_t]) for N_t in sorted(Cf_N.keys())
                   if not np.isnan(Cf_N.get(N_t, float('nan')))]
    if len(vals_sorted) >= 3:
        Ns = [v[0] for v in vals_sorted]
        Cfs = [v[1] for v in vals_sorted]
        # Linear fit C_f ~ a + b/N
        if len(Ns) >= 3:
            from numpy.polynomial import polynomial as P
            inv_N = [1.0/n for n in Ns]
            coeffs = np.polyfit(inv_N, Cfs, 1)
            C_inf = coeffs[1]  # intercept = C_f(N→∞)
            print(f"\n  Linear fit C_f ≈ {coeffs[1]:.4f} + {coeffs[0]:.4f}/N")
            print(f"  C_f(N→∞) ≈ {C_inf:.4f}")

    # ======================================================================
    # Summary
    # ======================================================================
    elapsed = time.time() - t0

    print(f"\n\n  {'='*90}")
    print(f"  SUMMARY")
    print(f"  {'='*90}")
    print(f"""
  TEST 1: C_f vs σ_Φ
    [See Part 1 hypothesis testing table above]
    Best candidate: check which column has ratios closest to constant

  TEST 2: ⟨Δy⟩·n = const for n = 2..8
    [See Part 2 constancy test]

  TEST 3: C_f(N) thermodynamic limit
    [See Part 3 summary]
""")
    print(f"  [Completed in {elapsed:.1f}s]")
    print("=" * 90)


if __name__ == '__main__':
    main()
