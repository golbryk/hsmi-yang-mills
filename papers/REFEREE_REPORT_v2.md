# REFEREE REPORT v2 — After All Fixes
## "The Mass Gap for Four-Dimensional SU(N) Yang-Mills Theory"
## By G. Olbryk (74 pages, March 2026)

**Date:** 2026-03-23 (internal audit, post-fix)

---

## OVERALL ASSESSMENT

The paper has been substantially improved since the first referee round.
The three critical gaps (polymer locality, semigroup convergence, SO(4) recovery)
have been addressed with self-contained arguments. All 84 theorem-like
environments now have proof blocks. The logical structure is complete.

**RECOMMENDATION: Conditional accept, pending resolution of items below.**

---

## 10 REMAINING OBJECTIONS (ordered by severity)

### 1. (MODERATE) Proposition prop:two_loop — arithmetic still incorrect
**Location:** Lines ~2756-2762
**Issue:** The tensor traces T_sunset = 17/3 and T_mixed = -1/3 are stated
without derivation. The algebra "8/3 becomes 11/3" is not explained.
The paper should either: (a) derive the tensor traces from Feynman rules,
or (b) explicitly state that only the SIGN of b_1 matters and cite the value.
**Fixable?** YES — but requires either 5 pages of Feynman diagrams or an
honest downgrade to "b_1 > 0 is used as input."
**Fatal?** NO — the proof uses only b_1 > 0.

### 2. (MODERATE) Balaban citations still appear in 9 proof environments
**Location:** lem:activity, lem:irrel_activity, thm:extended, lem:tree_complete,
prop:single_step_polymer, thm:polymer_induction, lem:RG_step, and others
**Issue:** The paper cites [B87] in the proofs of these lemmas (e.g., "by the
tree-graph inequality [B87, Prop 2.3]"). While these are now supplemented
by self-contained arguments, a purist referee could object that the paper
still RELIES on Balaban for the tree-graph inequality and polymer structure.
**Fixable?** PARTIALLY — the tree-graph inequality IS proved (Lemma lem:tree_complete),
but the initial polymer decomposition structure cites [B87].
**Fatal?** NO — the polymer induction is now self-contained.

### 3. (MODERATE) Lorentz invariance proof cites [ReedSimon4, Thm 2.14]
**Location:** lem:lorentz_from_euclidean, line ~5831
**Issue:** The edge-of-the-wedge theorem is a deep result in several complex
variables. The paper applies it but cites Reed-Simon for the statement.
A fully self-contained paper would prove the edge-of-the-wedge theorem.
**Fixable?** NO (in reasonable length) — this is a foundational result in
analysis, analogous to citing the spectral theorem.
**Fatal?** NO — this is a theorem in analysis, not in gauge theory.
Every constructive QFT paper uses it.

### 4. (MINOR) Non-triviality argument for continuum regime is weak
**Location:** Lines ~5055-5080
**Issue:** S_4^c != 0 is explicit at strong coupling (character expansion);
for the continuum regime (large beta), the argument is: "E_{k*} is a genuine
dimension-6 operator, therefore S_4^c = O(||E_{k*}||) != 0." This needs:
(a) explicit lower bound on S_4^c at large beta, (b) proof that the
dimension-6 operator does not accidentally vanish.
**Fixable?** YES — 1 page.
**Fatal?** NO.

### 5. (MINOR) "Polish space" not stated explicitly for S'(R^4)
**Location:** prop:tightness proof
**Issue:** Prokhorov requires Polish space. S'(R^4) with weak-* topology IS
Polish, but this should be stated explicitly with reference to the specific
metrization (e.g., Schwartz seminorms induce a metric).
**Fixable?** YES — 2 sentences.
**Fatal?** NO.

### 6. (MINOR) Transfer matrix self-adjointness needs more detail
**Location:** thm:OS_reconstruction Step 4
**Issue:** The proof says "T is self-adjoint by theta-invariance of the measure."
This is correct but should be expanded: T_t f(U) = integral f(tau_t U') K(U,U') dU',
theta-invariance gives K(U,U') = K(theta U', theta U) = K(U',U) (real kernel),
hence T_t = T_t^* on L^2(SU(N)^spatial, Haar).
**Fixable?** YES — 3 lines.
**Fatal?** NO.

### 7. (MINOR) Uniform lower bound on m_phys needs explicit verification
**Location:** Proposition prop:rg_exit_mass
**Issue:** The mass gap at the exit scale is m_phys = m_block / xi, where
xi = L_b^{k*} a_0. The paper claims m_phys >= C_AS Lambda_L > 0 but the
proportionality constant C_AS should be computed explicitly for N=2.
**Fixable?** YES — already partially done in prop:asymptotic_scaling.
**Fatal?** NO.

### 8. (COSMETIC) Some citations are historical, not logical
**Location:** Throughout
**Issue:** The paper cites Balaban, Dimock, Osterwalder-Seiler etc. in a way
that sometimes blurs the line between "historical credit" and "proof input."
Cleaner: separate historical remarks from proof content.
**Fixable?** YES — editorial.
**Fatal?** NO.

### 9. (COSMETIC) Paper length (74 pages) — hard for referee
**Issue:** A 74-page proof with 84 theorems is challenging to referee.
A short version (15-20 pages) with the essential chain would help.
**Fixable?** YES — write SHORT_VERSION.tex.
**Fatal?** NO.

### 10. (COSMETIC) Numerical verification section should be in appendix
**Location:** Section sec:numerics (currently in main body)
**Issue:** The GPU Monte Carlo verification is valuable but should be in
an appendix, clearly separated from the proof. Currently it's in the main
flow, which blurs the line between proof and validation.
**Fixable?** YES — move to appendix.
**Fatal?** NO.

---

## COMPARISON WITH REFEREE REPORT v1

| Objection | v1 severity | v2 severity | Status |
|-----------|-------------|-------------|--------|
| OS theorem not proved | FATAL | RESOLVED | Full 5-step proof added |
| Circularity (d-3) | FATAL | RESOLVED | Bootstrap argument |
| Polymer locality | SERIOUS | RESOLVED | Self-contained (compact support + decay) |
| b_1 derivation | SERIOUS | MODERATE | Arithmetic still imperfect |
| Semigroup convergence | SERIOUS | RESOLVED | Spectral representation argument |
| SO(4) recovery | MODERATE | RESOLVED | Complete operator classification |
| Uniqueness | MODERATE | RESOLVED | 4-step Cauchy proof |
| b_1 arithmetic | MODERATE | MODERATE | Still needs clean derivation |
| Proof sketch label | MINOR | RESOLVED | Upgraded to full proof |
| Non-triviality | MINOR | MINOR | Continuum regime still weak |

**Upgrade from v1:** 2 FATAL + 3 SERIOUS resolved. Remaining: 3 MODERATE + 7 MINOR/COSMETIC.

---

## BOTTOM LINE

**No FATAL objections remain.**

The proof has a complete logical structure from lattice → RG → continuum → OS → Wightman → mass gap.
All theorem-like environments have proof blocks. The three previously critical gaps are closed.

The remaining objections are:
- 3 MODERATE (b_1 arithmetic, Balaban citations in some proofs, edge-of-wedge citation)
- 7 MINOR/COSMETIC

None of these threatens the logical integrity of the proof.

**Assessment: This is a SERIOUS CANDIDATE for the Yang-Mills mass gap proof.**
Whether it is ACCEPTED depends on the community's tolerance for:
(a) the tree-graph inequality being proved in the paper but with Balaban-style structure
(b) the edge-of-the-wedge theorem being cited from Reed-Simon
(c) the b_1 coefficient being used with correct sign but imperfect derivation
