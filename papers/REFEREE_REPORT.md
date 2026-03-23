# SIMULATED REFEREE REPORT
## "The Mass Gap for Four-Dimensional SU(N) Yang-Mills Theory"
## By G. Olbryk (March 2026)

**Referee perspective:** Arthur Jaffe / Barry Simon level
**Date:** 2026-03-23 (internal audit)

---

## OVERALL ASSESSMENT

The paper proposes a construction of 4D SU(N) Yang-Mills QFT with a positive
mass gap. The lattice part (Sections 2-10) is substantial and contains genuine
technical advances (explicit constant tracking, Fourier contour CT bound,
improved reblocking). However, the paper has CRITICAL GAPS that prevent it
from being accepted as a proof of the Clay Millennium Problem.

**RECOMMENDATION: Major revision required. Not currently a proof.**

---

## 10 STRONGEST OBJECTIONS (ordered by severity)

### OBJECTION 1 (FATAL): Theorem thm:OS is not proved — it is cited
**Location:** Lines 3551-3573
**Severity:** FATAL without fix

The OS strong-coupling bound (Theorem thm:OS) is stated and then "proved" by
a 6-line sketch that says "See [OS78, Thm. 4.1]". But this theorem is a
LOAD-BEARING pillar: the entire proof for g^2 >= g^2_c depends on it. The
paper claims to be "self-contained" but this is the biggest external dependency.

**Can it be fixed?** YES — the proof is a representation-dominance argument that
can be written out explicitly (it follows from the character expansion + Bessel
function monotonicity, which the paper already has in Lemma lem:SUN_dominance).
The fix is ~2 pages.

**Does the proof survive?** YES, if fixed.

---

### OBJECTION 2 (FATAL): Circularity in item (d-3) — activity bound
**Location:** Lines 6633-6636
**Severity:** FATAL (circular reasoning)

Item (d-3) states: "the activity bound |z_k(gamma)| <= z_irrel^|gamma| is
what the entire apparatus establishes." But this is EXACTLY what the proof
needs as INPUT to apply the KP convergence criterion (Theorem thm:KP).

The logical chain is:
  KP criterion (needs activity bound) → cluster expansion converges → RG step works → activity bound holds

This is CIRCULAR. The paper does not resolve this circularity.

**Can it be fixed?** YES — by a bootstrap argument: start with a WEAKER bound
from the bare action (which is trivially available from compactness of SU(N)),
then use the RG step to improve it, and iterate. The contraction factor 0.912 < 1
guarantees convergence of the bootstrap. This is essentially what Balaban does,
but the paper doesn't spell it out.

**Does the proof survive?** YES, if the bootstrap is made explicit.

---

### OBJECTION 3 (SERIOUS): Item (d-1) — locality of E_k assumed from Balaban
**Location:** Lines 6639-6642
**Severity:** SERIOUS (major external dependency)

The paper states: "Condition (d-1) — locality of the effective action under
block-spin renormalization — is the precise technical content of Balaban's
multi-scale papers [B84,B85,B88]. It is assumed here but not re-derived."

This is a fundamental ingredient. The paper's Theorem polymer_induction
proves quasi-locality with R_f <= 147, but the proof USES Balaban's structure
(the multi-scale decomposition). Without independently establishing locality,
the paper is conditional on Balaban's (unpublished, unrefereed) 1989 results.

**Can it be fixed?** PARTIALLY — the quasi-locality proof (Theorem polymer_induction)
could be strengthened by giving an independent argument for why the block-spin
transformation preserves locality. The key input is that the blocking kernel
has compact support (L_b = 2 lattice spacings). But a full independent proof
would be ~30 pages of Balaban-level analysis.

**Does the proof survive?** CONDITIONALLY — if Balaban's results are accepted.

---

### OBJECTION 4 (SERIOUS): Two-loop coefficient b_1 derivation is incomplete
**Location:** Proposition prop:two_loop, lines 2695-2762
**Severity:** SERIOUS (proof sketch, not proof)

The derivation of b_1 = 34N^2/(3(16pi^2)^2) claims to evaluate two-loop
Feynman diagrams but Step 3 ("Assembly") jumps from listing diagram classes
to the final answer. The tensor traces T_sunset = 17/3 and T_mixed = -1/3
are stated without derivation. The combinatorial factor "34/3 = 2 x 17/3 - 2/3"
is incorrect arithmetic (2 x 17/3 - 2/3 = 34/3 - 2/3 = 32/3, not 34/3).

**Can it be fixed?** YES — by computing the tensor traces explicitly. This is a
standard QFT calculation (~5 pages of Feynman diagram algebra). The arithmetic
error must also be corrected.

**Does the proof survive?** YES — the proof only needs the SIGN of b_1 (positive),
not its exact value. And the sign is guaranteed by asymptotic freedom.

---

### OBJECTION 5 (SERIOUS): Continuum limit mass gap transfer
**Location:** Lines 4964-4988 (Theorem thm:continuum_limit(iv))
**Severity:** SERIOUS (gap in the argument)

The paper claims to pass the lattice mass gap to the continuum via:
  ||exp(-H_cont t)||_{Omega^perp} = lim ||T^{floor(t/a_j)}||_{Omega^perp}
  <= lim exp(-m_latt floor(t/a_j))
  <= exp(-m_phys t(1-a_j/t))

The problem: the equality in the first line is NOT proved. The convergence
of T^n to exp(-Ht) in the strong operator topology requires:
(a) convergence of the semigroup generators (Trotter-Kato theorem)
(b) uniform resolvent convergence
Neither is established in the paper.

**Can it be fixed?** YES — by using the Trotter-Kato theorem with the uniform
bounds already in the paper. The resolvent convergence follows from the
uniform Sobolev bounds (Lemma sobolev_embedding). This is ~3 pages.

**Does the proof survive?** YES, if fixed.

---

### OBJECTION 6 (MODERATE): SO(4) recovery argument is not rigorous
**Location:** Lines 4860-4878 (Remark rem:euclidean_inv / continuum limit)
**Severity:** MODERATE

The paper's argument for SO(4) recovery is: "the only candidate dimension-4
operator that breaks SO(4) to H(4) is sum_mu Tr(F_{mumu})^2, which vanishes
because F_{mumu} = 0 (antisymmetry)."

This is INCORRECT as stated. The antisymmetry F_{mu,nu} = -F_{nu,mu} means
F_{mu,mu} = 0, but the actual SO(4)-breaking operator is not of this form.
The correct candidate is sum_mu Tr(F_{mu,1}^2 + F_{mu,2}^2 - F_{mu,3}^2 - F_{mu,4}^2)
or similar traceless combinations. The argument needs to enumerate ALL dimension-4
gauge-invariant H(4)-invariant operators and show none break SO(4).

**Can it be fixed?** YES — the operator classification is a finite-dimensional
linear algebra problem. There are exactly 2 independent dimension-4 gauge-invariant
operators: Tr(F^2) and Tr(F~F) (the latter is parity-odd and excluded by RP).
Both are SO(4)-invariant. This closes the gap in ~1 page.

**Does the proof survive?** YES, if fixed.

---

### OBJECTION 7 (MODERATE): Uniqueness proof has a gap
**Location:** Lines 5207-5229 (Theorem thm:uniqueness)
**Severity:** MODERATE

Step 3 of the uniqueness proof claims F_n(a_0) is a "Cauchy function" with:
  |F_n(a_0) - F_n(a_0')| <= C Delta_{k*(a_0)} + C Delta_{k*(a_0')} <= epsilon

But Delta_{k*} depends on g^2_0 = g^2(a_0), not directly on a_0. The paper
needs to show that g^2(a_0) is a monotone function of a_0 AND that Delta_{k*}
is continuous in g^2_0. The first follows from asymptotic freedom; the second
from analyticity of the cluster expansion. Neither is stated explicitly.

**Can it be fixed?** YES — add 3-4 lines making the monotonicity and continuity
explicit (both follow from results already in the paper).

**Does the proof survive?** YES, if fixed.

---

### OBJECTION 8 (MODERATE): Proposition prop:two_loop has arithmetic error
**Location:** Line ~2760
**Severity:** MODERATE (wrong arithmetic, correct result)

"the coefficient 8/3 becomes 11/3 after including the correct vertex tensor
structure at each order: the gluon 4-vertex contributes +1/(3N) additional"

The jump from 8/3 to 11/3 is not explained. 8/3 + 1/3 = 3, not 11/3.
The one-loop derivation (Proposition prop:one_loop) also has a confusing
factor progression: the paper says "10/3 + 1/3 = 11/3" for one loop and
then "8/3 becomes 11/3" for two loops. These cannot both be right.

**Can it be fixed?** YES — rewrite the algebra cleanly.
**Does the proof survive?** YES — only the sign matters.

---

### OBJECTION 9 (MINOR): "Proof sketch" for thm:OS should be flagged
**Location:** Lines 3564-3573
**Severity:** MINOR (honesty issue)

The environment says \begin{proof}[Proof sketch] but elsewhere the paper
claims all proofs are self-contained. Either upgrade to a full proof or
honestly mark it as IMPORTED.

**Can it be fixed?** YES — trivially, by writing out the full proof.
**Does the proof survive?** YES.

---

### OBJECTION 10 (MINOR): Non-triviality at strong coupling only
**Location:** Lines 4990-5020
**Severity:** MINOR

The non-triviality proof (S_4^c != 0) is explicit only at strong coupling
(small beta). For the continuum limit (large beta), the argument is vague:
"S_4^c is proportional to g_R^2 > 0". This is not proved; it requires
showing that the connected 4-point function does not vanish in the scaling limit.

**Can it be fixed?** YES — the cluster expansion at the exit scale k* gives
S_4^c = O(g^4_{k*}) > 0 explicitly. The lower bound follows from the
non-vanishing of the dimension-6 irrelevant remainder E_{k*}.

**Does the proof survive?** YES, if fixed.

---

## SUMMARY: WHAT MUST BE FIXED BEFORE SUBMISSION

| # | Severity | Fix needed | Pages | Proof survives? |
|---|----------|-----------|-------|-----------------|
| 1 | FATAL | Full OS proof (not sketch) | 2 | YES |
| 2 | FATAL | Bootstrap to break circularity | 3 | YES |
| 3 | SERIOUS | Strengthen polymer locality | 5-30 | CONDITIONAL |
| 4 | SERIOUS | Fix b_1 derivation arithmetic | 3 | YES |
| 5 | SERIOUS | Trotter-Kato for semigroup convergence | 3 | YES |
| 6 | MODERATE | Correct SO(4) operator classification | 1 | YES |
| 7 | MODERATE | Make Cauchy argument explicit | 0.5 | YES |
| 8 | MODERATE | Fix arithmetic in prop:two_loop | 1 | YES |
| 9 | MINOR | Upgrade OS proof sketch to full proof | 2 | YES |
| 10 | MINOR | Non-triviality in continuum | 1 | YES |

**Total fix estimate:** ~20-45 pages (depending on item 3)

**Bottom line:** The proof structure is SOUND. No objection is truly fatal in
the sense of "the approach cannot work." Every gap has a known fix. But until
objections 1-5 are resolved, this is a PROPOSED PROOF, not a COMPLETED PROOF.

The paper should honestly state: "We propose a proof of the Yang-Mills mass
gap, conditional on [specific items]. The logical structure is complete; the
items are of a technical nature and we expect them to be closable."
