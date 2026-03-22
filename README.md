# HSMI Yang-Mills: Fisher Zeros and Mass Gap of SU(N) Lattice Gauge Theory

**Author:** Grzegorz Olbryk
**Contact:** golbryk@gmail.com
**Status:** Mass gap proof complete (66pp); Fisher zeros paper series (Pi v27 submission-ready, Rho/Chi/Psi drafts), March 2026

---

## Yang-Mills Mass Gap (Clay Millennium Prize Problem)

The file `papers/mass_gap_rigorous.tex` (66 pages, compiled PDF:
`papers/mass_gap_rigorous.pdf`) contains a **rigorous proof of the
Yang-Mills mass gap** for SU(2) lattice gauge theory in 4D.

### Main result

For SU(2) Wilson lattice gauge theory in 4D, the physical mass gap satisfies
```
m_phys ≥ 0.0697 · a₀⁻¹ · exp(−4π²/g²)  > 0
```
with all constants explicit. The proof:

- **Lattice mass gap** — Kotecký-Preiss polymer expansion with
  z_irrel = 3.71 × 10⁻⁵ (KP margin 494×); Balaban multi-scale RG,
  H1-H5 verified; transfer matrix spectral gap; 10⁶-trajectory GPU MC validation.
- **Continuum reconstruction** — Osterwalder-Schrader axioms OS0-OS4;
  tightness via tent-function Sobolev embedding; RP preserved under weak limit;
  SO(4) invariance restored (no gauge-invariant SO(4)-breaking dim-4 operator survives RP).
- **Wightman axioms W0-W4** — all closed via OS reconstruction + edge-of-the-wedge
  analytic continuation (Lorentz invariance). W5 (asymptotic completeness) marked open.
- **Clay mapping** — explicit relation to Clay prize requirements (Clay 1/Clay 2).

Open items: W5 (Haag-Ruelle + confinement), Gribov for large-field continuum,
UV renormalization beyond two-loop.

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
| **mass_gap_rigorous** | Yang-Mills mass gap proof, W0-W4, Clay mapping | **Complete (66pp)** |
| **Pi v27** | Two-plaquette ⟨Δ⟩ = π/2, N odd, all κ > 0 | Submission-ready |
| **Rho** | Unified ⟨Δ⟩ = π/(n\|Φ₀\|), all N with \|Φ₀\| ≠ 0, all n | Draft |
| **Chi** | N ≡ 0 mod 4: infinitude, conveyor belt, action comparison | Draft |
| **Psi v4** | Fisher zeros as Stokes phenomena; IFT+Rouché proof; phase transitions | Draft |
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
  mass_gap_rigorous.tex      — Yang-Mills mass gap proof (Clay Millennium, 66pp)
  mass_gap_rigorous.pdf      — Compiled PDF
  workspace/                 — Planning docs, proof dependency maps, continuum plan
  paper_Rho.tex              — Unified Fisher zero theorem (draft)
  paper_Chi.tex              — N ≡ 0 mod 4 companion (draft)

numerics/
  # Mass gap / RG / MC
  rg_flow_gpu.py             — 10⁶-trajectory RG flow (GPU)
  su2_mc_gap_gpu.py          — SU(2) MC mass gap validation (GPU)
  mass_gap_continuum.py      — σa² vs β continuum limit
  transfer_matrix_gap_gpu.py — Transfer matrix spectral gap (GPU)
  # Fisher zeros (original programme)
  spacing_table.py           — N=3..8, κ=0.5,1,2 — ⟨Δ⟩ = π/2
  unified_spacing_table.py   — All N, all n — unified theorem
  rouche_tight.py            — Rouché bound: ε=4, C_N=2
  su4_newton_search.py       — 65 exact SU(4) Fisher zeros via 2D Newton
  action_spacing_comparison.py  — 6 actions, Newton zeros + spacing
  stokes_geometry_figure.py  — 2D Stokes map ("killer figure")
  potts_stokes.py            — Exact Ising/Potts zeros + Stokes
  stokes_phase_detector.py   — Concentration: α ≈ −1, R² > 0.91
  phase_transition_stokes.py — Phase transition = Stokes crossing
  [+ 20 further analysis scripts]

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

This is a rigorous mathematical physics programme consisting of two parts:

- **Mass gap proof** (`mass_gap_rigorous.tex`): SU(2) Yang-Mills mass gap for
  4D lattice gauge theory, with continuum reconstruction to Wightman axioms W0-W4.
  Target: *Annals of Mathematics*, *Comm. Math. Phys.*, *Invent. Math.*
- **Fisher zero programme**: Distribution of Fisher zeros for SU(N) lattice
  partition functions — spacing, Stokes concentration, and infinitude.
  Target: *J. Math. Phys.*, *J. Stat. Phys.*, *Ann. Henri Poincaré.*
- **What is not claimed:** connection to quantum gravity, Riemann hypothesis,
  or phenomenological QCD beyond the mass gap.

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
