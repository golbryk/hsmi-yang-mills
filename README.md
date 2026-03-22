# HSMI Yang-Mills: Fisher Zeros and Mass Gap of SU(N) Lattice Gauge Theory

**Author:** Grzegorz Olbryk
**Contact:** golbryk@gmail.com
**Status:** Mass gap proof complete; Paper series (Pi v27 submission-ready, Rho/Chi/Psi drafts), March 2026

---

## Improved Convergence Bounds (Mass Gap Programme)

The directory `formal_mass_gap/` contains an **extension of Balaban's
convergence domain** from g^2 < 0.271 to g^2 < 0.447 (factor sqrt(e)),
using the Brydges-Kennedy tree formula. This is the largest rigorously
controlled coupling domain for 4D SU(2) lattice gauge theory.

This does NOT constitute a proof of the Yang-Mills mass gap.
See `formal_mass_gap/HONEST_ASSESSMENT.md` for a detailed analysis
of the relationship to the Clay Millennium Problem, and
`papers/mass_gap_proof.pdf` for the paper.

---

## What This Is

This repository accompanies the **HSMI programme** — a series of working papers
studying the distribution of Fisher zeros of SU(N) lattice Yang-Mills partition
functions. The programme establishes three main results:

1. **Unified spacing theorem** — For all N with |Φ₀| ≠ 0 and any number of
   plaquettes n ≥ 1, the Fisher zeros have average spacing ⟨Δ⟩ = π/(n|Φ₀|).

2. **Infinitude theorem** — For N ≡ 0 mod 4 (where |Φ₀| = 0), the Wilson-action
   partition function has infinitely many Fisher zeros via a conveyor belt
   mechanism through the representation tower.

3. **Stokes concentration theorem** — Fisher zeros of Z_N = Σ c_k exp(Nφ_k)
   concentrate on the Stokes network S = {Re φ_p = Re φ_q}, with
   dist(zero, S) = O(N⁻¹ log N).

---

## Main Results

### Theorem U-SU(N) *(Paper Rho v1)*

For all N ≥ 3 with |Φ₀| ≠ 0 (i.e., N odd or N ≡ 2 mod 4), and for n ≥ 1
plaquettes:

```
⟨Δ⟩ = π / (n · |Φ₀|)
```

where |Φ₀| = 1 for N odd, |Φ₀| = 2 for N ≡ 2 mod 4.

### Theorem 2Seq-SU(N) *(Paper Pi v27, proved for N odd)*

The two-plaquette (n=2) case: Fisher zeros form a two-sequence structure
with alternating gaps summing to π. Mean spacing π/2, exact for all κ > 0.

### Stokes Concentration *(Paper Psi v3)*

For transfer-matrix partition functions Z_N = Σ λ_k^N, zeros lie within
O(N⁻¹ log N) of the Stokes network |λ_p| = |λ_q|. Verified numerically
with power-law exponents α ≈ −1, R² > 0.91 across Ising, Potts, and SU(N)
models.

---

## Paper Series

| Paper | Content | Status |
|-------|---------|--------|
| **Pi v27** | Two-plaquette ⟨Δ⟩ = π/2, N odd, all κ > 0 | Submission-ready |
| **Rho v1** | Unified ⟨Δ⟩ = π/(n\|Φ₀\|), all N with \|Φ₀\| ≠ 0, all n | Draft |
| **Chi v1** | N ≡ 0 mod 4: infinitude, conveyor belt, action comparison | Draft |
| **Psi v3** | Fisher zeros as Stokes phenomena; IFT+Rouché proof | Draft |
| Omega-C v7 | Single-plaquette: Δ = 2π/(N − t_vdm(N)) | Final |
| Sigma v12 | Lee-Yang failure for odd SU(N); μ_N = 1/2^{N-1} | Final |
| Tau final | Certified SU(3) zero β₁ = 1.10583 + 2.50155i | Final |
| Xi v8 | 6 certified SU(3) Fisher zeros | Final |
| Omega v11 | Single-plaquette Δ_k → π/2 for SU(3) | Final |

---

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Reproduce key figures (Paper Psi)

```bash
python numerics/stokes_geometry_figure.py   # → papers/fig_stokes_2d_map.png (~ 3 min)
python numerics/stokes_phase_detector.py    # → papers/fig_stokes_concentration.png (~ 10 min)
python numerics/potts_stokes.py             # → console output: exact Ising/Potts zeros
```

Output data is saved to `data/` for replotting without recomputation.

---

## Numerical Verification

### Core verification scripts (original)

```bash
python numerics/spacing_table.py         # N=3..8, κ=0.5,1,2 — ⟨Δ⟩ = π/2
python numerics/fisher_zeros_SU5.py      # SU(5) explicit zeros
python numerics/unified_spacing_table.py # All N, all n — unified theorem
python numerics/rouche_tight.py          # Rouché bound: ε=4, C_N=2
```

### Action comparison and mechanism analysis

```bash
python numerics/action_spacing_comparison.py  # 6 actions, Newton zeros + spacing
python numerics/entirety_classification.py    # Entirety: Wilson/Symanzik/Iwasaki/DBW2/HK
python numerics/transfer_spectrum.py          # Eigenphase spectrum, level crossings
python numerics/interference_mechanism.py     # Multi-mode Fourier analysis
```

### Stokes phenomena (Paper Psi)

```bash
python numerics/stokes_geometry_figure.py  # 2D Stokes map — the "killer figure"
python numerics/potts_stokes.py            # Exact Ising/Potts zeros + Stokes
python numerics/stokes_phase_detector.py   # Concentration: α ≈ −1, R² > 0.91
python numerics/large_n_asymptotic.py      # n=2..50 convergence, Ising universality
```

### SU(4) exact zeros (N ≡ 0 mod 4)

```bash
python numerics/su4_newton_search.py          # 65 exact Fisher zeros via 2D Newton
python numerics/conveyor_belt_universality.py # Conveyor belt mechanism
```

All scripts are standalone (no imports between them). Pure Python with numpy;
no GPU needed. matplotlib for figure generation.

---

## Repository Structure

```
papers/
  paper_Pi_v27.docx          — Main paper (submission-ready)
  paper_Rho_v1.tex           — Unified theorem (draft)
  paper_Chi_v1.tex           — N ≡ 0 mod 4 companion (draft)
  paper_Psi_v3.tex           — Stokes phenomena (draft)
  paper_Psi_v[1,2].tex       — Earlier Psi versions (superseded)
  paper_Omega_C_v7_final.docx
  paper_Sigma_v12.docx
  paper_Tau_final.docx
  paper_Xi_v8.docx
  paper_Omega_v11.docx
  fig_*.png                  — Paper figures

numerics/
  [35 analysis scripts]      — See papers/README.md or script docstrings
  archive/                   — Superseded exploratory scripts

figures/
  permutohedron_A3_SU4.svg   — Permutohedron A₃ for SU(4), coloured by Φ
```

---

## Resolved Open Problems

All three open problems from Pi v16 are now resolved:

| Problem | Resolution | Paper |
|---------|-----------|-------|
| N even (Φ₀ = 0) | Infinitely many zeros via conveyor belt; entirety theorem | Chi v1 |
| n-plaquette (⟨Δ⟩ = π/n?) | Proved: ⟨Δ⟩ = π/(n\|Φ₀\|) for all N with \|Φ₀\| ≠ 0 | Rho v1 |
| Thermodynamic limit | Zero density = \|Φ₀\|/π per unit coupling | Rho v1 |

---

## Scope

This is a rigorous mathematical physics programme. Specifically:

- **What is proved:** distribution of Fisher zeros for SU(N) lattice gauge
  theory partition functions — spacing, concentration on Stokes networks,
  and infinitude.
- **What is not claimed:** connection to quantum gravity, Riemann hypothesis,
  or phenomenological QCD predictions.
- **Target journals:** J. Math. Phys., J. Stat. Phys., Lett. Math. Phys.,
  Ann. Henri Poincaré, Physica A.

---

## References

- Gross, Witten (1980). *Phys. Rev. D* **21**, 446.
- Bröcker, tom Dieck (1985). *Representations of Compact Lie Groups.* §VI.1.
- Hörmander (1985). *Analysis of Linear PDE I.* §7.7.
- Macdonald (1995). *Symmetric Functions and Hall Polynomials.* §I.3.
- Seiler (1982). *Gauge Theories as a Problem of Constructive QFT.* App. B.
- Lee, Yang (1952). *Phys. Rev.* **87**, 404.
- Fisher (1965). *Lectures in Theoretical Physics* **7C**, 1.
- Ritt (1929). *Trans. AMS* **31**, 680.

---

## Use of Computational Assistance

Numerical computations, manuscript preparation, and proof-checking were
performed with the assistance of large language models (LLMs), under the
direct supervision of the author. All scientific decisions, mathematical
arguments, and interpretations remain the responsibility of the author.

---

## License

MIT License (see LICENSE file).
