#!/usr/bin/env python3
"""
Stokes Geometry Figure + q-state Potts Universality Test

Part 1: 2D Stokes map for SU(4) Wilson — |A_p(κ+iy)| = |A_q(κ+iy)| contours
        overlaid with Fisher zeros. THE killer figure.
Part 2: q-state Potts model (q=3,4,10) — exact 2-eigenvalue exponential sum.
Part 3: Analytic C_f estimate from eigenphase velocities.

Author: Grzegorz Olbryk
Date: 2026-03-10
"""

import numpy as np
from itertools import combinations
from pathlib import Path
import sys, time

PAPER_DIR = Path(__file__).resolve().parents[1] / 'papers'

t0 = time.time()


# ===================================================================
# SU(4) Weyl quadrature infrastructure
# ===================================================================

def build_weyl_grid(N, n_quad):
    """Build SU(N) Weyl integration grid."""
    pts = np.linspace(0, 2*np.pi, n_quad, endpoint=False) + np.pi/n_quad
    grids = np.meshgrid(*([pts]*(N-1)), indexing='ij')
    thetas_free = np.stack([g.ravel() for g in grids], axis=-1)
    theta_last = -thetas_free.sum(axis=-1, keepdims=True)
    thetas = np.concatenate([thetas_free, theta_last], axis=-1)

    # Vandermonde squared measure
    n_pts = thetas.shape[0]
    vdm2 = np.ones(n_pts)
    for i in range(N):
        for j in range(i+1, N):
            diff = thetas[:, i] - thetas[:, j]
            vdm2 *= 4 * np.sin(diff/2)**2

    vdm2 = np.abs(vdm2)
    weights = vdm2 / vdm2.sum()
    return thetas, weights


def h_p_vec(thetas, p_max):
    """Complete homogeneous symmetric polynomials h_0..h_pmax."""
    n_pts, N = thetas.shape
    z = np.exp(1j * thetas)
    result = np.zeros((p_max+1, n_pts), dtype=complex)
    result[0] = 1.0

    # Power sums
    pk = np.zeros((p_max+1, n_pts), dtype=complex)
    for k in range(1, p_max+1):
        pk[k] = np.sum(z**k, axis=1)

    # Newton's identity
    for p in range(1, p_max+1):
        s = np.zeros(n_pts, dtype=complex)
        for k in range(1, p+1):
            s += pk[k] * result[p-k]
        result[p] = s / p

    return result


def dim_rep(p, N):
    """Dimension of symmetric rep of SU(N)."""
    from math import comb
    return comb(p + N - 1, N - 1)


def compute_Ap(s_val, thetas, weights, p_max, f_vals):
    """Compute A_p(s) for p=0..p_max."""
    hp = h_p_vec(thetas, p_max)
    boltz = np.exp(s_val * f_vals)
    Ap = np.zeros(p_max + 1, dtype=complex)
    for p in range(p_max + 1):
        Ap[p] = np.sum(weights * hp[p] * boltz)
    return Ap


# ===================================================================
print("="*80)
print("  Stokes Geometry + Universality Tests")
print("="*80)

# ===================================================================
# PART 1: 2D Stokes Map for SU(4) Wilson
# ===================================================================

print("\n  PART 1: 2D Stokes Map — SU(4) Wilson")
print("  " + "-"*60)

N = 4
n_quad = 30  # smaller for 2D grid speed
p_max = 10
n_plaq = 2

print(f"  Building SU({N}) Weyl grid (n_quad={n_quad})...", end=" ", flush=True)
thetas, weights = build_weyl_grid(N, n_quad)
z = np.exp(1j * thetas)
f_wilson = np.real(z.sum(axis=1))
print("Done.")

# 2D grid in (κ, y) plane
n_kappa = 40
n_y = 200
kappa_range = np.linspace(0.3, 2.5, n_kappa)
y_range = np.linspace(0.1, 25.0, n_y)

print(f"  Computing |A_p| on {n_kappa}×{n_y} grid...", end=" ", flush=True)

# |A_p(κ+iy)| array: shape (n_kappa, n_y, p_max+1)
Ap_abs = np.zeros((n_kappa, n_y, p_max+1))

hp = h_p_vec(thetas, p_max)

for ik, kappa in enumerate(kappa_range):
    for iy, y in enumerate(y_range):
        s = kappa + 1j * y
        boltz = np.exp(s * f_wilson)
        for p in range(p_max+1):
            Ap_abs[ik, iy, p] = np.abs(np.sum(weights * hp[p] * boltz))

print("Done.")

# Find dominant rep at each point
p_dom = np.zeros((n_kappa, n_y), dtype=int)
for ik in range(n_kappa):
    for iy in range(n_y):
        weighted = np.array([dim_rep(p, N) * Ap_abs[ik, iy, p]**n_plaq
                             for p in range(p_max+1)])
        p_dom[ik, iy] = np.argmax(weighted)

# Find Stokes crossings: where p_dom changes
print("  Finding Stokes curves (dominant-pair boundaries)...", end=" ", flush=True)

# For the figure: track where |A_p| = |A_q| for adjacent dominant regions
stokes_points_y = []  # (κ, y, p, q)

for ik in range(n_kappa):
    kappa = kappa_range[ik]
    for iy in range(n_y - 1):
        if p_dom[ik, iy] != p_dom[ik, iy+1]:
            y_mid = (y_range[iy] + y_range[iy+1]) / 2
            p1, p2 = p_dom[ik, iy], p_dom[ik, iy+1]
            stokes_points_y.append((kappa, y_mid, min(p1,p2), max(p1,p2)))

for iy in range(n_y):
    y = y_range[iy]
    for ik in range(n_kappa - 1):
        if p_dom[ik, iy] != p_dom[ik+1, iy]:
            kappa_mid = (kappa_range[ik] + kappa_range[ik+1]) / 2
            p1, p2 = p_dom[ik, iy], p_dom[ik+1, iy]
            stokes_points_y.append((kappa_mid, y, min(p1,p2), max(p1,p2)))

print(f"{len(stokes_points_y)} boundary points.")

# Now find Fisher zeros on several κ scan lines
print("  Finding Fisher zeros on scan lines...", flush=True)

kappa_scans = [0.5, 1.0, 1.5, 2.0]
all_zeros = []  # (κ, y)

for kappa in kappa_scans:
    y_fine = np.linspace(0.1, 25.0, 5000)
    Z_vals = np.zeros(len(y_fine), dtype=complex)

    for iy_f, y in enumerate(y_fine):
        s = kappa + 1j * y
        boltz = np.exp(s * f_wilson)
        Z = 0.0
        for p in range(p_max+1):
            dp = dim_rep(p, N)
            Ap = np.sum(weights * hp[p] * boltz)
            Z += dp * Ap**n_plaq
        Z_vals[iy_f] = Z

    # Sign changes of Re Z
    re_Z = Z_vals.real
    for i in range(len(re_Z)-1):
        if re_Z[i] * re_Z[i+1] < 0:
            # Linear interpolation
            y_zero = y_fine[i] - re_Z[i] * (y_fine[i+1] - y_fine[i]) / (re_Z[i+1] - re_Z[i])
            all_zeros.append((kappa, y_zero))

print(f"  Found {len(all_zeros)} Fisher zeros across {len(kappa_scans)} scan lines.")

for kappa in kappa_scans:
    kz = [(k,y) for k,y in all_zeros if abs(k-kappa)<0.01]
    print(f"    κ={kappa:.1f}: {len(kz)} zeros")

# Also compute level crossing contours for specific pairs
print("  Computing |A_p|=|A_q| contours for specific pairs...")

# Focus on κ = 1.0 slice for detailed overlay
kappa_fig = 1.0
y_dense = np.linspace(0.1, 25.0, 2000)

Ap_dense = np.zeros((len(y_dense), p_max+1))
Ap_phase = np.zeros((len(y_dense), p_max+1))

for iy, y in enumerate(y_dense):
    s = kappa_fig + 1j * y
    boltz = np.exp(s * f_wilson)
    for p in range(p_max+1):
        val = np.sum(weights * hp[p] * boltz)
        Ap_dense[iy, p] = np.abs(val)
        Ap_phase[iy, p] = np.angle(val)

# Find all |A_p| = |A_q| crossings at κ=1
crossings_1d = []
for p in range(p_max):
    for q in range(p+1, min(p+4, p_max+1)):
        diff = Ap_dense[:, p] - Ap_dense[:, q]
        for i in range(len(diff)-1):
            if diff[i] * diff[i+1] < 0:
                y_cross = y_dense[i] - diff[i]*(y_dense[i+1]-y_dense[i])/(diff[i+1]-diff[i])
                crossings_1d.append((y_cross, p, q))

crossings_1d.sort()
zeros_k1 = sorted([y for k,y in all_zeros if abs(k-kappa_fig)<0.01])

print(f"  κ=1: {len(crossings_1d)} crossings, {len(zeros_k1)} zeros")

# Detailed overlay: for each zero, find nearest crossing
print("\n  Overlay analysis (κ=1.0):")
print(f"  {'zero y':>10s}  {'near cross y':>12s}  {'pair':>8s}  {'dist':>8s}")
for yz in zeros_k1[:20]:
    best_d = 999
    best_cross = None
    best_pair = None
    for yc, p, q in crossings_1d:
        d = abs(yz - yc)
        if d < best_d:
            best_d = d
            best_cross = yc
            best_pair = (p, q)
    print(f"  {yz:10.4f}  {best_cross:12.4f}  ({best_pair[0]:d},{best_pair[1]:d})  {best_d:8.4f}")


# ===================================================================
# Generate matplotlib figure
# ===================================================================

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("\n  [matplotlib not available — skipping figure generation]")

if HAS_MPL:
    print("\n  Generating figures...", flush=True)

    # Figure 1: The Killer Figure — 1D overlay at κ=1
    fig, axes = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [3, 1]})

    ax = axes[0]
    ax.set_title(r'Fisher Zeros and Stokes Curves — SU(4) Wilson, $n=2$, $\kappa=1.0$', fontsize=14)

    # Plot |A_p(κ+iy)| curves (weighted by d_p)
    colors_rep = plt.cm.viridis(np.linspace(0, 0.9, p_max+1))
    for p in range(min(8, p_max+1)):
        dp = dim_rep(p, N)
        log_weighted = np.log10(dp * Ap_dense[:, p]**n_plaq + 1e-300)
        ax.plot(y_dense, log_weighted, color=colors_rep[p], alpha=0.5, lw=1,
                label=f'$d_{p}|A_{p}|^{n_plaq}$' if p < 6 else None)

    # Mark |A_p| = |A_q| crossings
    for yc, p, q in crossings_1d:
        ax.axvline(yc, color='gray', alpha=0.15, lw=0.5)

    # Mark Fisher zeros
    for yz in zeros_k1:
        ax.axvline(yz, color='red', alpha=0.7, lw=1.5, linestyle='--')

    ax.set_xlabel('$y = \\mathrm{Im}\\, s$', fontsize=12)
    ax.set_ylabel('$\\log_{10}(d_p |A_p|^n)$', fontsize=12)
    ax.legend(fontsize=9, ncol=3, loc='upper right')
    ax.set_xlim(0, 25)

    # Add zero markers at top
    for yz in zeros_k1:
        ax.plot(yz, ax.get_ylim()[1], 'rv', markersize=8, clip_on=False)

    # Bottom panel: dominant pair index
    ax2 = axes[1]
    ax2.set_title('Dominant representation index', fontsize=12)

    p_dom_dense = np.zeros(len(y_dense), dtype=int)
    for iy in range(len(y_dense)):
        weighted = np.array([dim_rep(p, N) * Ap_dense[iy, p]**n_plaq
                             for p in range(p_max+1)])
        p_dom_dense[iy] = np.argmax(weighted)

    ax2.plot(y_dense, p_dom_dense, 'b-', lw=1.5)
    for yz in zeros_k1:
        ax2.axvline(yz, color='red', alpha=0.5, lw=1, linestyle='--')
    ax2.set_xlabel('$y = \\mathrm{Im}\\, s$', fontsize=12)
    ax2.set_ylabel('$p^*(y)$', fontsize=12)
    ax2.set_xlim(0, 25)

    plt.tight_layout()
    fig_path = str(PAPER_DIR / 'fig_stokes_1d_overlay.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()

    # Figure 2: 2D Stokes map in (κ, y) plane
    fig, ax = plt.subplots(figsize=(10, 12))

    # Color by dominant rep
    im = ax.pcolormesh(kappa_range, y_range, p_dom.T,
                        cmap='tab10', shading='auto', alpha=0.3)
    plt.colorbar(im, ax=ax, label='Dominant rep $p^*$')

    # Stokes boundary points
    sk = [s[0] for s in stokes_points_y]
    sy = [s[1] for s in stokes_points_y]
    ax.scatter(sk, sy, c='gray', s=1, alpha=0.3, label='Stokes boundary')

    # Fisher zeros
    zk = [z[0] for z in all_zeros]
    zy = [z[1] for z in all_zeros]
    ax.scatter(zk, zy, c='red', s=30, marker='x', linewidths=1.5,
               zorder=10, label='Fisher zeros')

    ax.set_xlabel('$\\kappa = \\mathrm{Re}\\, s$', fontsize=12)
    ax.set_ylabel('$y = \\mathrm{Im}\\, s$', fontsize=12)
    ax.set_title(r'Stokes Map — SU(4) Wilson, $n=2$: Fisher zeros on Stokes boundaries', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(kappa_range[0], kappa_range[-1])
    ax.set_ylim(y_range[0], y_range[-1])

    fig_path_2d = str(PAPER_DIR / 'fig_stokes_2d_map.png')
    plt.savefig(fig_path_2d, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path_2d}")
    plt.close()

    # Figure 3: Ising exact overlay
    fig, ax = plt.subplots(figsize=(10, 5))

    beta_r = 0.5
    y_ising = np.linspace(0, 12, 2000)

    lp_abs = np.zeros(len(y_ising))
    lm_abs = np.zeros(len(y_ising))
    Z_ising = np.zeros(len(y_ising), dtype=complex)
    n_ising = 10

    for i, y in enumerate(y_ising):
        beta = beta_r + 1j * y
        lp = 2 * np.cosh(beta)
        lm = 2 * np.sinh(beta)
        lp_abs[i] = np.abs(lp)
        lm_abs[i] = np.abs(lm)
        Z_ising[i] = lp**n_ising + lm**n_ising

    ax.plot(y_ising, np.log10(lp_abs), 'b-', lw=2, label='$|\\lambda_+|$')
    ax.plot(y_ising, np.log10(lm_abs), 'r-', lw=2, label='$|\\lambda_-|$')

    # Stokes lines
    for k in range(8):
        y_stokes = np.pi/4 + k*np.pi/2
        if y_stokes < 12:
            ax.axvline(y_stokes, color='green', alpha=0.5, lw=1.5, linestyle=':',
                       label='Stokes: $|\\lambda_+|=|\\lambda_-|$' if k==0 else None)

    # Fisher zeros
    re_Z = Z_ising.real
    ising_zeros = []
    for i in range(len(re_Z)-1):
        if re_Z[i] * re_Z[i+1] < 0:
            y_z = y_ising[i] - re_Z[i]*(y_ising[i+1]-y_ising[i])/(re_Z[i+1]-re_Z[i])
            ising_zeros.append(y_z)

    for yz in ising_zeros:
        ax.axvline(yz, color='red', alpha=0.4, lw=1, linestyle='--')
    ax.plot([yz for yz in ising_zeros if yz < 12],
            [ax.get_ylim()[1]]*len([yz for yz in ising_zeros if yz < 12]),
            'rv', markersize=6, clip_on=False, label=f'Fisher zeros ($n={n_ising}$)')

    ax.set_xlabel('$y = \\mathrm{Im}\\,\\beta$', fontsize=12)
    ax.set_ylabel('$\\log_{10}|\\lambda_\\pm|$', fontsize=12)
    ax.set_title(f'1D Ising: Fisher zeros exactly on Stokes lines ($n={n_ising}$, $\\beta_r={beta_r}$)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 12)

    plt.tight_layout()
    fig_path_ising = str(PAPER_DIR / 'fig_ising_stokes.png')
    plt.savefig(fig_path_ising, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path_ising}")
    plt.close()


# ===================================================================
# PART 2: q-state Potts Model Universality
# ===================================================================

print("\n\n  PART 2: q-state Potts Model — Universality Test")
print("  " + "-"*60)
print("  Z_n(β) = λ_1(β)^n + (q-1)·λ_2(β)^n")
print("  λ_1 = e^β + q - 1,  λ_2 = e^β - 1")
print("  Stokes: |λ_1| = |λ_2|")

for q in [3, 4, 10]:
    print(f"\n  === q = {q} Potts ===")

    beta_r = 0.5
    y_scan = np.linspace(0.01, 20.0, 10000)

    # Find Stokes crossings: |e^β + q - 1| = |e^β - 1|
    stokes_y = []
    prev_diff = None
    for y in y_scan:
        beta = beta_r + 1j * y
        eb = np.exp(beta)
        lam1 = eb + q - 1
        lam2 = eb - 1
        diff = np.abs(lam1) - np.abs(lam2)
        if prev_diff is not None and diff * prev_diff < 0:
            # Interpolate
            y_prev = y - (y_scan[1] - y_scan[0])
            y_cross = y_prev - prev_diff * (y - y_prev) / (diff - prev_diff)
            stokes_y.append(y_cross)
        prev_diff = diff

    print(f"  Stokes crossings in [0, 20]: {len(stokes_y)}")
    if len(stokes_y) > 0:
        print(f"  First few: {', '.join(f'{y:.4f}' for y in stokes_y[:5])}")

    # Find Fisher zeros for various n
    print(f"\n  {'n':>5s}  {'#zeros':>7s}  {'⟨Δy⟩':>8s}  {'n⟨Δy⟩':>8s}  {'⟨d_stokes⟩':>11s}  {'med_d/⟨Δy⟩':>11s}")

    for n in [2, 3, 4, 5, 8, 10, 20, 50]:
        # Compute Z_n
        Z_vals = np.zeros(len(y_scan), dtype=complex)
        for i, y in enumerate(y_scan):
            beta = beta_r + 1j * y
            eb = np.exp(beta)
            lam1 = eb + q - 1
            lam2 = eb - 1
            Z_vals[i] = lam1**n + (q-1) * lam2**n

        # Find zeros
        re_Z = Z_vals.real
        zeros = []
        for i in range(len(re_Z)-1):
            if re_Z[i] * re_Z[i+1] < 0:
                y_z = y_scan[i] - re_Z[i]*(y_scan[i+1]-y_scan[i])/(re_Z[i+1]-re_Z[i])
                zeros.append(y_z)

        if len(zeros) < 2:
            print(f"  {n:5d}  {len(zeros):7d}  {'---':>8s}  {'---':>8s}  {'---':>11s}  {'---':>11s}")
            continue

        gaps = np.diff(zeros)
        avg_gap = np.mean(gaps)

        # Distance to nearest Stokes crossing
        dists = []
        for yz in zeros:
            if len(stokes_y) > 0:
                d_min = min(abs(yz - ys) for ys in stokes_y)
            else:
                d_min = float('inf')
            dists.append(d_min)

        med_d = np.median(dists)
        avg_d = np.mean(dists)

        print(f"  {n:5d}  {len(zeros):7d}  {avg_gap:8.4f}  {n*avg_gap:8.4f}  {avg_d:11.4f}  {med_d/avg_gap:11.4f}")


# ===================================================================
# PART 3: Analytic C_f estimate from eigenphase velocities
# ===================================================================

print("\n\n  PART 3: Analytic C_f Estimate")
print("  " + "-"*60)
print("  C_f = ⟨2π / |dθ_p/dy - dθ_q/dy|⟩_crossings")

# Use the dense κ=1 data
print("\n  Wilson SU(4), κ=1.0:")

# Compute phase velocities at each crossing
dy = y_dense[1] - y_dense[0]
dtheta_dy = np.zeros((len(y_dense), p_max+1))
for p in range(p_max+1):
    dtheta_dy[:, p] = np.gradient(np.unwrap(Ap_phase[:, p]), dy)

# At each crossing, compute the phase velocity difference
Cf_estimates = []
for yc, p, q in crossings_1d:
    # Find nearest y index
    idx = np.argmin(np.abs(y_dense - yc))
    dw = abs(dtheta_dy[idx, p] - dtheta_dy[idx, q])
    if dw > 0.01:  # avoid degenerate crossings
        Cf_local = 2*np.pi / dw
        Cf_estimates.append(Cf_local)

if Cf_estimates:
    print(f"  Number of crossings with |Δω| > 0.01: {len(Cf_estimates)}")
    print(f"  ⟨C_f⟩ from eigenphase velocities: {np.mean(Cf_estimates):.4f}")
    print(f"  Median C_f: {np.median(Cf_estimates):.4f}")
    print(f"  C_f from zero spacing (n=2): {2 * np.mean(np.diff(zeros_k1)):.4f}")
    print(f"  C_f from Newton search: ~2.50 (known)")

    # Histogram of local C_f values
    print(f"\n  Distribution of local C_f values:")
    cf_arr = np.array(Cf_estimates)
    for pct in [10, 25, 50, 75, 90]:
        print(f"    {pct}th percentile: {np.percentile(cf_arr, pct):.4f}")

# Also for Ising: exact C_f
print("\n  Ising (β_r = 0.5):")
beta_r_ising = 0.5
# At Stokes line y = π/4:
y_test = np.pi/4
beta = beta_r_ising + 1j * y_test
lp = 2*np.cosh(beta)
lm = 2*np.sinh(beta)
# Phase velocity = d/dy[arg(λ)]
eps = 1e-6
lp_plus = 2*np.cosh(beta_r_ising + 1j*(y_test+eps))
lm_plus = 2*np.sinh(beta_r_ising + 1j*(y_test+eps))
dtheta_p = (np.angle(lp_plus) - np.angle(lp)) / eps
dtheta_m = (np.angle(lm_plus) - np.angle(lm)) / eps
delta_omega = abs(dtheta_p - dtheta_m)
Cf_ising = 2*np.pi / delta_omega

print(f"  |dθ_+/dy| at y=π/4: {abs(dtheta_p):.6f}")
print(f"  |dθ_-/dy| at y=π/4: {abs(dtheta_m):.6f}")
print(f"  |Δω| = {delta_omega:.6f}")
print(f"  C_f = 2π/|Δω| = {Cf_ising:.4f}")
print(f"  Predicted ⟨Δy⟩ for n=10: {Cf_ising/10:.4f}")

# Check against actual n=10 spacing
ising_gaps = np.diff(ising_zeros) if len(ising_zeros) > 1 else []
if len(ising_gaps) > 0:
    print(f"  Actual ⟨Δy⟩ for n=10: {np.mean(ising_gaps):.4f}")


# Potts q=3: exact C_f
print("\n  Potts q=3 (β_r = 0.5):")
q_potts = 3
y_test = stokes_y[0] if stokes_y else 1.0  # use first Stokes crossing
beta = 0.5 + 1j * y_test
eb = np.exp(beta)
lam1 = eb + q_potts - 1
lam2 = eb - 1
eb_p = np.exp(0.5 + 1j*(y_test+eps))
lam1_p = eb_p + q_potts - 1
lam2_p = eb_p - 1
dth1 = (np.angle(lam1_p) - np.angle(lam1)) / eps
dth2 = (np.angle(lam2_p) - np.angle(lam2)) / eps
dw_potts = abs(dth1 - dth2)
Cf_potts = 2*np.pi / dw_potts if dw_potts > 0 else float('inf')
print(f"  First Stokes crossing at y = {y_test:.4f}")
print(f"  |Δω| = {dw_potts:.6f}")
print(f"  C_f = 2π/|Δω| = {Cf_potts:.4f}")


print(f"\n\n  [Completed in {time.time()-t0:.1f}s]")
print("="*80)
