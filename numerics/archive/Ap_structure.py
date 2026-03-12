"""
A_p Structure Analysis: Two-frequency decomposition
=====================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026

Decomposes A_p(κ+iy) into its oscillatory components to identify
the exact interference mechanism producing Fisher zero spacing.

Key question: Does A_p have ONE dominant frequency (balanced saddle only)
or TWO comparable frequencies (two saddles)?
"""

import numpy as np
from math import comb, pi


def precompute_grid_su3(n_pts=250):
    """Precompute SU(3) Weyl integration grid."""
    theta = np.linspace(0, 2*pi, n_pts, endpoint=False)
    T1, T2 = np.meshgrid(theta, theta)
    T3 = -(T1 + T2)
    Z1, Z2, Z3 = np.exp(1j*T1), np.exp(1j*T2), np.exp(1j*T3)
    vand = np.abs(Z1 - Z2)**2 * np.abs(Z1 - Z3)**2 * np.abs(Z2 - Z3)**2
    Phi = (Z1 + Z2 + Z3).real
    return Z1, Z2, Z3, vand, Phi


def compute_hp_grid(p, Z1, Z2, Z3):
    """h_p via Newton's identities on grid."""
    if p == 0:
        return np.ones_like(Z1)
    pk = [Z1**k + Z2**k + Z3**k for k in range(1, p+1)]
    h = [np.ones_like(Z1, dtype=complex)]
    for k in range(1, p+1):
        hk = sum(pk[j] * h[k-1-j] for j in range(k)) / k
        h.append(hk)
    return h[p]


def precompute_grid_suN(N, n_pts=100):
    """Precompute SU(N) Weyl integration grid via N-1 angles."""
    # N-1 independent angles
    theta_grid = np.linspace(0, 2*pi, n_pts, endpoint=False)
    # For N=3 we can do full grid; for larger N we use random sampling
    if N <= 4:
        grids = np.meshgrid(*[theta_grid for _ in range(N-1)])
        thetas = [g.ravel() for g in grids]
    else:
        # Random sampling for large N
        n_samples = n_pts ** 2  # Same effective number
        thetas = [np.random.uniform(0, 2*pi, n_samples) for _ in range(N-1)]

    # Last angle from constraint
    theta_N = -sum(thetas)

    # Eigenvalues
    all_thetas = thetas + [theta_N]
    Z = [np.exp(1j * t) for t in all_thetas]

    # Vandermonde |Δ|²
    vand = np.ones(len(Z[0]))
    for j in range(N):
        for k in range(j+1, N):
            vand *= np.abs(Z[j] - Z[k])**2

    # Phi = Re Tr U
    Phi = sum(z.real for z in Z)

    return Z, vand, Phi


def compute_hp_general(p, Z):
    """h_p for general N using Newton's identities on grid."""
    N = len(Z)
    if p == 0:
        return np.ones_like(Z[0])
    z_arr = np.array(Z)  # shape (N, n_pts)
    pk = [np.sum(z_arr**k, axis=0) for k in range(1, p+1)]
    h = [np.ones_like(Z[0], dtype=complex)]
    for k in range(1, p+1):
        hk = sum(pk[j] * h[k-1-j] for j in range(k)) / k
        h.append(hk)
    return h[p]


def compute_Ap_su3(p, kappa, y, grid):
    """Compute unnormalized A_p(κ+iy) for SU(3)."""
    Z1, Z2, Z3, vand, Phi = grid
    s = kappa + 1j * y
    weight = vand * np.exp(s * Phi)
    hp = compute_hp_grid(p, Z1, Z2, Z3)
    dp = comb(p + 2, 2)
    return np.sum(hp * weight) / dp


def main():
    print("=" * 80)
    print("  A_p Structure Analysis: SU(3)")
    print("=" * 80)

    kappa = 1.0
    N_PTS = 250

    print(f"\n  Parameters: κ={kappa}, grid={N_PTS}²")
    grid = precompute_grid_su3(N_PTS)

    # ================================================================
    # Part 1: Compute A_p at high y to extract asymptotic structure
    # ================================================================
    print("\n  Part 1: A_p(κ+iy) at large y — amplitude and phase")
    print("  " + "-" * 60)

    y_large = np.arange(30, 60, 0.1)

    for p in [0, 1, 2, 3]:
        dp = comb(p+2, 2)
        Ap_vals = np.array([compute_Ap_su3(p, kappa, y, grid) for y in y_large])

        # Phase and amplitude
        phase = np.unwrap(np.angle(Ap_vals))
        amp = np.abs(Ap_vals)
        log_amp = np.log(amp + 1e-30)

        # Fit phase to linear: phase ≈ ω * y + φ₀
        from numpy.polynomial import polynomial as P
        coeffs_phase = np.polyfit(y_large, phase, 1)
        omega = coeffs_phase[0]

        # Fit log amplitude to: log|A_p| ≈ -α log(y) + const
        coeffs_amp = np.polyfit(np.log(y_large), log_amp, 1)
        alpha = -coeffs_amp[0]

        print(f"\n  A_{p} (d_p={dp}):")
        print(f"    Phase velocity ω = {omega:.6f} (expected -1)")
        print(f"    Amplitude decay power α = {alpha:.4f} (|A_p| ~ y^{{-α}})")

        # Extract the sub-dominant oscillation
        # Remove dominant oscillation: multiply by exp(iy)
        Ap_demod = Ap_vals * np.exp(1j * y_large)  # should be ~ slowly varying
        phase_demod = np.unwrap(np.angle(Ap_demod))
        amp_demod = np.abs(Ap_demod)

        # Look for remaining oscillation in the demodulated signal
        # FFT of demodulated signal
        fft_demod = np.fft.fft(Ap_demod - np.mean(Ap_demod))
        dy = y_large[1] - y_large[0]
        freqs = np.fft.fftfreq(len(Ap_demod), d=dy)
        power = np.abs(fft_demod)**2

        pos = freqs > 0.01
        pf, pp = freqs[pos], power[pos]
        top3 = np.argsort(pp)[-3:][::-1]

        print(f"    Demodulated (×exp(iy)) FFT peaks:")
        for idx in top3:
            omega_sub = 2 * pi * pf[idx]
            # In original frame: ω_original = -1 + ω_sub
            omega_orig = -1 + omega_sub
            print(f"      ω_demod = {omega_sub:8.4f} → ω_orig = {omega_orig:8.4f}, "
                  f"power = {pp[idx]:.4e}")

        # Also check: ratio of demodulated amplitude oscillation to mean
        amp_ratio = np.std(amp_demod) / np.mean(amp_demod)
        print(f"    |A_p × exp(iy)| variation: std/mean = {amp_ratio:.6f}")

    # ================================================================
    # Part 2: Direct two-frequency fit
    # ================================================================
    print(f"\n\n  Part 2: Two-frequency fit A_p ≈ a exp(iΦ₁y)/y^α + b exp(iΦ₂y)/y^β")
    print("  " + "-" * 60)

    # For SU(3), candidate frequencies: Φ₁ = -1 (balanced), Φ₂ = +3 (identity), -3/2 (center)
    candidate_Phi2 = [3.0, -1.5, 1.0, -3.0]

    for p in [0, 1, 2]:
        dp = comb(p+2, 2)
        Ap_vals = np.array([compute_Ap_su3(p, kappa, y, grid) for y in y_large])

        print(f"\n  A_{p}: testing Φ₂ candidates")

        best_Phi2 = None
        best_residual = float('inf')

        for Phi2 in candidate_Phi2:
            # Model: A_p ≈ a × exp(-iy)/y^α + b × exp(iΦ₂y)/y^β
            # With α, β to be determined
            # Demodulate by exp(-iy): Ap × exp(iy) ≈ a/y^α + b × exp(i(Φ₂+1)y)/y^β
            Ap_demod = Ap_vals * np.exp(1j * y_large)

            # Extract the oscillation at frequency (Φ₂+1):
            osc_freq = Phi2 + 1  # angular frequency in demodulated frame
            proj_cos = Ap_demod * np.exp(-1j * osc_freq * y_large)
            # This should be ~ b/y^β (slowly varying) + interference terms

            # Compute power at this frequency
            # Use a simple projection: average of e^{-i ω y} × signal
            # This gives the Fourier coefficient at frequency ω
            coeff = np.mean(proj_cos)
            power = np.abs(coeff)

            # Also compute residual after removing both components
            # Simplistic: model Ap ≈ c1 exp(-iy) + c2 exp(iΦ₂ y)
            # Solve least squares
            E1 = np.exp(-1j * y_large)
            E2 = np.exp(1j * Phi2 * y_large)
            M = np.column_stack([E1, E2])
            # Weighted by 1/y^2 to emphasize large-y behavior
            W = np.diag(1.0 / y_large**2)
            sol, residual, _, _ = np.linalg.lstsq(W @ M, W @ Ap_vals, rcond=None)
            c1, c2 = sol
            fit = M @ sol
            residual = np.sqrt(np.mean(np.abs((Ap_vals - fit) / Ap_vals)**2))

            if residual < best_residual:
                best_residual = residual
                best_Phi2 = Phi2
                best_c1, best_c2 = c1, c2

            print(f"    Φ₂ = {Phi2:5.1f}: |c₁|={abs(c1):.2e}, |c₂|={abs(c2):.2e}, "
                  f"ratio=|c₂/c₁|={abs(c2)/abs(c1):.6f}, residual={residual:.6f}")

        print(f"    → Best Φ₂ = {best_Phi2}, residual = {best_residual:.6f}")
        print(f"      c₁ = {best_c1:.4e}, c₂ = {best_c2:.4e}")

    # ================================================================
    # Part 3: Compute Z_n for different n, test spacing formula
    # ================================================================
    print(f"\n\n  Part 3: Z_n spacing for n=2,3,4,5")
    print("  " + "-" * 60)

    y_range = np.arange(5, 80, 0.05)  # finer grid
    P_MAX = 20

    # Precompute all A_p(κ+iy)
    print(f"  Computing A_p for p=0..{P_MAX}, {len(y_range)} y-values...")
    Ap_all = np.zeros((len(y_range), P_MAX+1), dtype=complex)
    for yi, y in enumerate(y_range):
        if yi % 300 == 0:
            print(f"    y = {y:.1f} ({yi+1}/{len(y_range)})...")
        s = kappa + 1j * y
        Z1, Z2, Z3, vand, Phi = grid
        weight = vand * np.exp(s * Phi)
        for p in range(P_MAX+1):
            hp = compute_hp_grid(p, Z1, Z2, Z3)
            dp = comb(p+2, 2)
            Ap_all[yi, p] = np.sum(hp * weight) / dp

    for n in [2, 3, 4, 5]:
        # Z_n = Σ d_p A_p^n
        Z_n = np.zeros(len(y_range), dtype=complex)
        for p in range(P_MAX+1):
            dp = comb(p+2, 2)
            Z_n += dp * Ap_all[:, p]**n

        # Find zeros (|Z_n| minima below threshold, or Re Z_n sign changes)
        # Better: find points where |Z_n| has local minima
        abs_Z = np.abs(Z_n)

        # Sign changes in Re(Z_n)
        re_Z = Z_n.real
        zeros_re = []
        for i in range(len(re_Z) - 1):
            if re_Z[i] * re_Z[i+1] < 0:
                t = re_Z[i] / (re_Z[i] - re_Z[i+1])
                zeros_re.append(y_range[i] + t * (y_range[i+1] - y_range[i]))

        # Also sign changes in Im(Z_n)
        im_Z = Z_n.imag
        zeros_im = []
        for i in range(len(im_Z) - 1):
            if im_Z[i] * im_Z[i+1] < 0:
                t = im_Z[i] / (im_Z[i] - im_Z[i+1])
                zeros_im.append(y_range[i] + t * (y_range[i+1] - y_range[i]))

        # |Z_n| local minima
        minima_y = []
        for i in range(1, len(abs_Z) - 1):
            if abs_Z[i] < abs_Z[i-1] and abs_Z[i] < abs_Z[i+1]:
                # Parabolic interpolation
                y0, y1, y2 = y_range[i-1], y_range[i], y_range[i+1]
                f0, f1, f2 = abs_Z[i-1], abs_Z[i], abs_Z[i+1]
                if f1 < 0.1 * np.median(abs_Z):  # Only deep minima
                    minima_y.append(y1)

        print(f"\n  Z_{n}:")
        print(f"    Re(Z_n) sign changes: {len(zeros_re)}")
        print(f"    |Z_n| deep minima: {len(minima_y)}")

        # Use |Z_n| minima for gap analysis (more reliable for complex zeros)
        if len(minima_y) > 1:
            gaps = np.diff(minima_y)
            expected = pi / n
            print(f"    Mean gap (|Z_n| minima): {np.mean(gaps):.4f} "
                  f"(expected π/{n} = {expected:.4f})")
            if len(gaps) > 3:
                # Exclude first 2 and last gaps (edge effects)
                interior_gaps = gaps[2:-1] if len(gaps) > 5 else gaps[1:]
                print(f"    Mean gap (interior): {np.mean(interior_gaps):.4f}")
                print(f"    Std gap (interior):  {np.std(interior_gaps):.4f}")

        if len(zeros_re) > 1:
            gaps_re = np.diff(zeros_re)
            expected = pi / n
            print(f"    Mean gap (Re sign change): {np.mean(gaps_re):.4f}")
            # Show first 8 gaps
            for i, g in enumerate(gaps_re[:8]):
                print(f"      Δy_{i} = {g:.4f}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
