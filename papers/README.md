# Paper Series — HSMI Yang-Mills Programme

**Author:** Grzegorz Olbryk  <g.olbryk@gmail.com>
**Date:** March 2026

This directory contains the working papers of the HSMI Yang-Mills programme,
studying Fisher zeros of SU(N) lattice Yang-Mills partition functions.

---

## Current Papers

| File | Content | Status |
|------|---------|--------|
| `paper_Pi_v27.docx` | **Main paper** — Fisher zeros of two-plaquette SU(N), N odd; Theorem 2Seq | Submission-ready |
| `paper_Rho_v1.tex` | **Unified theorem** — ⟨Δ⟩ = π/(n\|Φ₀\|) for all N (\|Φ₀\| ≠ 0), all n | Draft |
| `paper_Chi_v1.tex` | **Companion** — N ≡ 0 mod 4: infinitude, conveyor belt, Wilson vs HK | Draft |
| `paper_Psi_v3.tex` | **Stokes phenomena** — Fisher zeros as eigenvalue degeneracies; IFT+Rouché proof | Draft |
| `paper_Omega_C_v7_final.docx` | Full classification of single-plaquette spacing: Δ = 2π/(N − t_vdm(N)) | Final |
| `paper_Sigma_v12.docx` | Lee-Yang failure for SU(N) odd; μ_N = 1/2^{N-1} | Final |
| `paper_Tau_final.docx` | Certified SU(3) zero β₁ = 1.10583 + 2.50155i | Final |
| `paper_Xi_v8.docx` | 6 certified SU(3) Fisher zeros | Final |
| `paper_Omega_v11.docx` | Single-plaquette Δ_k → π/2 for SU(3) | Final |

### Figures

| File | Description |
|------|-------------|
| `fig_stokes_2d_map.png` | 2D Stokes map: Fisher zeros lie on eigenvalue degeneracy boundaries |
| `fig_stokes_1d_overlay.png` | 1D eigenvalue magnitude overlay at κ=1 |
| `fig_ising_stokes.png` | Ising model: exact Stokes correspondence |
| `fig_stokes_ising_L8.png` | Ising L=8 strip: Stokes map with transfer-matrix eigenvalues |
| `fig_phase_detection.png` | β_c(L) finite-size scaling + Potts β_c vs q |
| `fig_potts_stokes_comparison.png` | 2×2 Potts Stokes maps (q=2,3,4,5) |
| `fig_stokes_concentration.png` | Log-log Stokes gap vs N: α ≈ −1, R² > 0.91 |

---

## Theorem Hierarchy

```
Sigma v12: Lee-Yang failure for SU(N) odd
    |
    +-- Tau, Xi: Certified zeros for SU(3)
    |
Omega-C v7: Single-plaquette spacing Δ = 2π/(N − t_vdm)
    |
Pi v27: Two-plaquette spacing ⟨Δ⟩ = π/2  [main result, N odd]
    |
    +-- Rho v1: Unified ⟨Δ⟩ = π/(n|Φ₀|) for all N (|Φ₀| ≠ 0), all n
    |     +-- Improved Rouché: ε = 4 universal, y₀ reduced 41%
    |     +-- Thermodynamic limit: density |Φ₀|/π
    |
    +-- Chi v1: N ≡ 0 mod 4 (conveyor belt, infinitude, Wilson vs HK)
    |     +-- Entirety classification: 6 standard actions
    |     +-- G-reg theorem: R = ∞, super-exponential decay
    |
    +-- Psi v3: Fisher zeros as Stokes phenomena
          +-- Zeros concentrate on Stokes network S = {|λ_p| = |λ_q|}
          +-- dist(zero, S) = O(N⁻¹ log N), verified α ≈ −1, R² > 0.91
          +-- Models: Ising strip, q-state Potts, SU(N) gauge theory
```

---

## Key Lemmas (Pi v27)

| Lemma | Content |
|-------|---------|
| CP-N | Permutohedron A_{N-1}: balanced split gives Φ_VH = ±1 for N odd |
| VO-N | Vandermonde order: ord(k,N) = k(k-1) + (N-k)(N-k-1) |
| SA | Schur parity: s_λ(−z) = (−1)^|λ| s_λ(z) ⟹ S_A = S_B |
| SP-N | Stationary phase: A_λ(iy) ∼ v_λ e^{iΦy} / y^{m_N} |
| O2-N | Casimir convergence: d_λ ≤ C exp(c|λ|), C₂(λ) ≥ |λ| |
| R-N | Rouché with explicit threshold y₀ = (4C_N/μ_N)^{1/power} |
| TSI | Two-Saddle Interference: ⟨Δ⟩ = π/ω |

---

## Author

Grzegorz Olbryk
g.olbryk@gmail.com
March 2026
