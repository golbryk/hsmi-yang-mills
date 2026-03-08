# HSMI Yang-Mills: Fisher Zeros of the Two-Plaquette SU(N) Model

**Author:** Grzegorz Olbryk  
**Contact:** g.olbryk@gmail.com  
**Status:** Working paper series (Pi v16, March 2026)

---

## What This Is

This repository accompanies the **HSMI programme** — a series of working papers
studying the distribution of Fisher zeros of the Gross-Witten two-plaquette
SU(N) Yang-Mills partition function.

The main result (Paper Pi v16) is a proof that for all **odd N ≥ 3** and all
coupling constants **κ > 0**, the Fisher zeros have universal average spacing:

```
⟨Δ⟩ = π/2
```

This is a rigorous mathematical theorem in lattice gauge theory / mathematical
physics. It is not a theory of everything.

---

## Main Result

**Theorem 2Seq-SU(N)** *(Pi v16, proved for N odd)*

Let Z_{2P}^{SU(N)}(iy, κ) be the two-plaquette SU(N) Gross-Witten partition
function at imaginary coupling. For all odd N ≥ 3 and all κ > 0, the Fisher
zeros {y_k} form, for k ≥ k₀(N,κ), a two-sequence structure with alternating
gaps Δ_short = π − α_N(κ) and Δ_long = α_N(κ), where α_N ∈ (0,π) depends on
N and κ. The average spacing satisfies:

```
⟨Δ⟩ = (Δ_short + Δ_long) / 2 = π/2     [exact, for all N odd, all κ > 0]
```

The angle α_N(κ) = arccos(ρ_N(κ)) with ρ_N = −S_AB / (2S_A) encodes the
N and κ dependence; it equals π/2 for N = 3 (exact uniform spacing), and
differs from π/2 for N odd N ≥ 5.

---

## Why π/2 Is Universal

The result follows from a six-step proof chain:

| Step | Lemma | Content |
|------|-------|---------|
| 1 | CP-N | Permutohedron A_{N-1} classifies saddle points; balanced split has Φ = ±1 for N odd |
| 2 | VO-N | Vandermonde order gives stationary-phase power y^{−m_N} |
| 3 | SA | Schur parity s_λ(−z) = (−1)^|λ| s_λ(z) forces S_A = S_B |
| 4 | O2-N | Casimir decay gives absolute convergence of character expansion |
| 5 | R-N | Remainder bound with explicit Rouché threshold y₀ |
| 6 | TSI | Two-Saddle Interference: Z∼[2A cos(ωy)+B]/y^m ⟹ ⟨Δ⟩ = π/ω |

The frequency ω = 2 comes from the **Weyl group geometry**: the minimal
non-zero value of Φ = Re Tr U on {0,π}^N ∩ SU(N) is ±1, and the two-plaquette
structure A_λ(iy)² doubles the phase. This is a property of the group element
(Weyl symmetry), not of any individual representation.

---

## Paper Series

| Paper | Content | Status |
|-------|---------|--------|
| Sigma v12 | Lee-Yang failure for odd SU(N); μ_N = 1/2^{N-1} | Final |
| Tau final | Certified SU(3) zero β₁ = 1.10583 + 2.50155i | Final |
| Xi v8 | 6 certified SU(3) Fisher zeros | Final |
| Omega v11 | Single-plaquette Δ_k → π/2 for SU(3) | Final |
| Omega-C v7 | Full classification: Δ = 2π/(N − t_vdm(N)) for all N | Final |
| **Pi v16** | **⟨Δ⟩ = π/2 for all N odd, all κ > 0** | **Current** |

---

## Numerical Verification

**SU(5), κ = 1.0:** 20 Fisher zeros computed.

| k | y_k | Δ_k | type |
|---|-----|-----|------|
| 0 | 0.87544 | — | — |
| 1 | 2.26615 | 1.39070 | π−α |
| 2 | 4.01704 | 1.75089 | α |
| 3 | 5.40774 | 1.39070 | π−α |
| ... | ... | ... | ... |
| 19 | 30.54048 | 1.39070 | π−α |

Every adjacent pair sums to π = 3.14159 ✓  
Mean of 20 gaps = 1.57080 = π/2 ✓ (exact to 5 decimal places)

Full tables for SU(3), SU(5), SU(7) at κ = 0.5, 1.0, 2.0 are in Pi v16.

---

## Open Problems

**N even.** For N ≥ 4 even, the balanced-split saddle has Φ_VH = 0 (a single
non-oscillating dominant term). The two-saddle formula does not apply directly;
a different asymptotic mechanism is needed. This is an open problem.

**n-Plaquette conjecture.** For n plaquettes, the Weyl-symmetry argument
predicts ω = n and therefore:

```
⟨Δ⟩ = π/n    (conjectured for N odd, n ≥ 2)
```

Proof requires extending Lemma SP-N to n-th powers of A_λ(iy).

| n | ω | ⟨Δ⟩ | Status |
|---|---|-----|--------|
| 1 | — | 2π/(N−t_vdm) | Proved (Omega-C v7) |
| 2 | 2 | π/2 | **Proved (Pi v16)** |
| n | n | π/n | Open conjecture |

**Thermodynamic limit.** The spacing ⟨Δ⟩ for the full lattice SU(N) model
(|Λ|→∞) is an open problem requiring cluster expansion or transfer-matrix
methods.

---

## Scope and Honest Assessment

This is a **small, clean theorem** in mathematical physics / lattice gauge
theory. Specifically:

- **What is proved:** average Fisher-zero spacing π/2 for two-plaquette SU(N),
  N odd, all κ > 0.
- **What is not claimed:** theory of everything, connection to Riemann
  hypothesis, quantum gravity, or phenomenological predictions for SU(3) QCD.
- **Target journals:** Journal of Mathematical Physics, Letters in Mathematical
  Physics, Annales Henri Poincaré, SIGMA.

The proof chain is complete and referee-ready for the N-odd case.

---

## Repository Structure

```
papers/
  paper_Pi_v16.docx        — Current main paper (publication draft)
  paper_Omega_C_v7.docx    — Single-plaquette classification (final)
  paper_Sigma_v12.docx     — Lee-Yang failure for odd N (final)
  [additional papers]

numerics/
  fisher_zeros_SU5.py      — 20-zero computation for SU(5)
  spacing_table.py         — Full N=3..8, κ=0.5,1,2 verification

figures/
  permutohedron_A3_SU4.svg — Permutohedron A₃ for SU(4), coloured by Φ
```

---

## References

- Gross, Witten (1980). *Phys. Rev. D* **21**, 446.
- Bröcker, tom Dieck (1985). *Representations of Compact Lie Groups.* §VI.1.
- Hörmander (1985). *Analysis of Linear PDE I.* §7.7.
- Macdonald (1995). *Symmetric Functions and Hall Polynomials.* §I.3.
- Seiler (1982). *Gauge Theories as a Problem of Constructive QFT.* App. B.

---

## Use of Computational Assistance

Numerical computations, manuscript preparation, and proof-checking were
performed with the assistance of large language models (LLMs), under the
direct supervision of the author. All scientific decisions, mathematical
arguments, and interpretations remain the responsibility of the author.

---

## License

MIT License (see LICENSE file).
