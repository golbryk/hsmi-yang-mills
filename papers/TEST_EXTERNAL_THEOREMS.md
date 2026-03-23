# EXTERNAL THEOREM AUDIT — mass_gap_rigorous.tex

For each external theorem used: exact version needed, hypotheses, where verified.

## 1. SPECTRAL THEOREM (self-adjoint operators)

**Version needed:** For a bounded self-adjoint operator T on a Hilbert space H,
there exists a unique spectral measure E such that T = integral lambda dE(lambda).
Consequence: ||T^n|| = ||T||^n for self-adjoint T >= 0.

**Hypotheses:**
- T must be self-adjoint
- T must be bounded (or at least densely defined and closed)

**Where verified in paper:**
- thm:OS_reconstruction Step 4: T = transfer matrix, self-adjoint by construction
  (T is the product of Haar-integrated link operators, each self-adjoint)
  CHECK: The OS construction Step 4 states "T is a contraction and self-adjoint"
  — self-adjointness follows from theta-invariance of the measure. VERIFIED.

**VERDICT: OK** — hypotheses explicitly checked.

---

## 2. PROKHOROV COMPACTNESS THEOREM

**Version needed:** A family of probability measures on a Polish space is tight
if and only if it is relatively compact (precompact) in the weak topology.

**Hypotheses:**
- The space must be Polish (complete separable metric)
- The family must be tight (for every eps, exists compact K with mu(K^c) < eps for all mu)

**Where verified in paper:**
- prop:tightness: S'(R^4) is Polish (nuclear Frechet space, hence metrizable).
  CHECK: Is this stated? Line ~4800: "S'(R^4) carries the initial topology of all
  Sobolev seminorms" — this is correct, S'(R^4) with weak-* topology is Polish.
- Tightness: proved via uniform H^{-s} bound + Rellich-Kondrachov.

**VERDICT: OK** — but the paper should explicitly state that S'(R^4) is Polish.
POTENTIAL ISSUE: S'(R^4) with the strong topology is NOT metrizable. The paper
uses the weak-* topology (initial topology of Sobolev seminorms), which IS Polish.
This should be stated more carefully.

STATUS: MINOR GAP — add one sentence clarifying the topology.

---

## 3. RELLICH-KONDRACHOV COMPACT EMBEDDING

**Version needed:** H^{-s}(R^4) embeds compactly into H^{-(s+1)}(R^4) for s > 0.

**Hypotheses:**
- Sobolev spaces on R^4 (not on a bounded domain)
- Need s > 0

**CRITICAL ISSUE:** Rellich-Kondrachov on R^4 does NOT give compact embedding!
It gives compact embedding on BOUNDED DOMAINS only. On R^4, H^s embeds into
H^{s-1} continuously but NOT compactly.

The paper uses this at line ~4800 for tightness on S'(R^4). This is the standard
approach but requires an additional argument: the LOCALIZATION trick.
Since the Schwinger functions have uniform exponential decay (eq 2pt_exp_decay),
they are essentially supported on a bounded region of diameter ~1/m_phys.
The exponential tail can be estimated separately.

STATUS: POTENTIAL SERIOUS GAP — Rellich-Kondrachov on R^4 requires localization.
The exponential decay provides this but the argument must be made explicit.

---

## 4. BESSEL FUNCTION BOUNDS

**Version needed:** I_nu(x) / I_{nu-1}(x) < 1 for x > 0, nu >= 1.
Also: I_nu(x) = sum_{m>=0} (x/2)^{nu+2m} / (m! Gamma(nu+m+1)).

**Hypotheses:** x > 0, nu >= 1.

**Where verified:** lem:Bessel, thm:OS Step 4.
The Bessel series is a standard convergent power series. The monotonicity
follows from term-by-term comparison.

**VERDICT: OK** — self-contained proof in the paper.

---

## 5. DOMINATED CONVERGENCE THEOREM

**Version needed:** If f_n -> f pointwise and |f_n| <= g with g integrable,
then integral f_n -> integral f.

**Hypotheses:** Need a dominating function.

**Where verified:**
- thm:continuum_limit(iv): S_2^{c,(a_j)}(t) -> S_2^cont(t) pointwise.
  Dominating function: |S_2^c| <= 1 (eq Sn_uniform). Since the integral is
  over a compact domain (or the decay is exponential), dominated convergence applies.

**VERDICT: OK** — dominating function is |S_n| <= 1 (explicit).

---

## 6. EDGE-OF-THE-WEDGE (BOGOLIUBOV)

**Version needed:** If f is analytic in the forward tube T_n and SO(4)-invariant
on the Euclidean section, then f extends to an SO(1,3)-invariant function
on the Minkowski boundary.

**Hypotheses:**
- f must be analytic (holomorphic) in the tube
- f must be a tempered distribution on the boundary

**Where verified:**
- lem:lorentz_from_euclidean: Steps 1-4 verify analyticity (Paley-Wiener-Schwartz)
  and SO(4) invariance (from the lattice).
  The paper cites [ReedSimon4, Thm 2.14] for the abstract theorem.

**VERDICT: OK** — but the citation to Reed-Simon should be accompanied by
verification of the hypotheses. The paper does verify analyticity and
temperedness. The key hypothesis (invariance on the Euclidean section)
is verified by Remark rem:euclidean_inv.

---

## 7. CAUCHY-SCHWARZ INEQUALITY

**Version needed:** |<f,g>| <= ||f|| ||g|| in any inner product space.

**VERDICT: OK** — axiom of Hilbert space, no hypotheses to check.

---

## 8. SCHUR INEQUALITY (operator norms)

**Version needed:** For a matrix A, ||A||_{2->2} <= sqrt(max_i sum_j |a_{ij}| * max_j sum_i |a_{ij}|).

**Where verified:** thm:CT Step 3 (Combes-Thomas proof).
The paper computes both row and column sums explicitly.

**VERDICT: OK** — computation is explicit.

---

## SUMMARY OF EXTERNAL THEOREM AUDIT

| Theorem | Status | Issue |
|---------|--------|-------|
| Spectral | OK | Self-adjointness verified |
| Prokhorov | MINOR GAP | State S'(R^4) is Polish explicitly |
| Rellich-Kondrachov | POTENTIAL GAP | Compact embedding on R^4 needs localization |
| Bessel | OK | Self-contained |
| Dominated convergence | OK | |S_n| <= 1 dominates |
| Edge-of-wedge | OK | Hypotheses verified |
| Cauchy-Schwarz | OK | Axiom |
| Schur | OK | Explicit computation |

## CRITICAL FINDING

**Rellich-Kondrachov on R^4** is the most serious potential issue.
The compact embedding H^{-s} -> H^{-(s+1)} holds on bounded domains
but NOT on R^4. The paper needs the LOCALIZATION argument: exponential
decay of S_2 confines the support effectively to a bounded region,
after which R-K applies. This argument exists implicitly (the uniform
exponential decay bound) but is not stated explicitly.

**Action needed:** Add a lemma making the localization explicit.
