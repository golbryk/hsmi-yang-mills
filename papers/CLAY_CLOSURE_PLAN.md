# PLAN ZAMKNIECIA DOWODU CLAY — Yang-Mills Mass Gap

Data: 2026-03-23
Autor: G. Olbryk + Claude

## STAN AKTUALNY

Paper: mass_gap_rigorous.tex (72 strony, 82 twierdzenia)
Lattice mass gap: UDOWODNIONY (m >= 0.0697 > 0 dla SU(2))
Continuum limit: ZAPROPONOWANY z 3 dziurami

## 3 DZIURY DO ZAMKNIECIA (w kolejnosci priorytetu)

### DZIURA 1 (KRYTYCZNA): Polymer locality — item (d-1)
**Lokalizacja:** linie 6674-6723
**Problem:** Quasi-lokalnosc E_k pod block-spin RG jest udowodniona
(Thm polymer_induction, R_f <= 147) ale OPIERA sie na strukturze
Balabana [B84,B85,B88]. Potrzebny niezalezny dowod.

**Strategia zamkniecia:**
Kluczowa obserwacja: blocking kernel B ma KOMPAKTOWY NOSNIK (L_b=2).
Jesli operator B ma nosnik w kuli o promieniu R_B, to:
  E_{k+1}(gamma) = B[E_k](gamma)
zalezy tylko od konfiguracji w otoczeniu R_B * gamma.
To jest GEOMETRYCZNY argument — nie potrzeba calej maszynerii Balabana.

**Kroki:**
1. Zdefiniowac formalnie blocking kernel B z kompaktowym nosnikiem
2. Udowodnic: B zachowuje lokalnosc (jesli E_k jest R-lokalne, to E_{k+1} jest (R+R_B)-lokalne)
3. Udowodnic: R_f nie rosnie nieograniczenie (R_{k+1} <= max(R_k, R_B) = R_B bo contraction)
4. Wniosek: E_k jest R_B-quasi-lokalne na kazdej skali

**Szacowany rozmiar:** 5-8 stron

### DZIURA 2 (POWAZNA): Trotter-Kato semigroup convergence
**Lokalizacja:** linie 5017-5040
**Problem:** Zbieznosc T_{a_j}^{n_j} -> exp(-Ht) w silnej topologii
operatorowej wymaga jawnej kontroli resolwenty. Argument jest
zarysowany ale nie w pelni udowodniony.

**Strategia zamkniecia:**
Uzyc wariantu Trotter-Kato dla polugrup kontrakcji:
- Mamy: T_{a_j} kontraktywne, samosprzezone
- Mamy: <Phi, T_{a_j}^{n_j} Psi> -> <Phi, exp(-Ht) Psi> (z weak convergence Schwinger functions)
- Potrzebujemy: silna zbieznosc

Kluczowy lemat: Jesli {T_n} sa kontrakcjami samosprzezonymi i
<Phi, T_n^{k_n} Psi> -> <Phi, S(t) Psi> dla gesto zbior Phi, Psi,
to T_n^{k_n} -> S(t) silnie (Reed-Simon, Thm VIII.20).

Alternatywnie: uzyc resolwenty. R_lambda(H_{a_j}) -> R_lambda(H_cont)
slabo, + uniform bound -> silna zbieznosc resolwent -> silna zbieznosc
polugrup (Kato).

**Szacowany rozmiar:** 3-4 strony

### DZIURA 3 (UMIARKOWANA): SO(4) operator classification
**Lokalizacja:** linie 4860-4878
**Problem:** Argument ze F_{mumu}=0 jest bledny (F jest antysymetryczny,
wiec F_{mumu}=0 jest trywialne). Prawdziwy argument wymaga klasyfikacji
WSZYSTKICH dim-4 gauge-invariantnych operatorow.

**Strategia zamkniecia:**
Klasyfikacja dim-4 operatorow:
- Tr(F_{mu,nu} F_{mu,nu}) — SO(4) invariant (jedyny operator dim-4 z dwoma F)
- Tr(F_{mu,nu} ~F_{mu,nu}) — pseudoskalar, wykluczony przez RP (parity)
- Nie ma innych niezaleznych dim-4 gauge-invariantnych operatorow
  (dowod: przestrzen jest 2-wymiarowa, oba elementy sa SO(4)-invariantne)

**Szacowany rozmiar:** 1-2 strony

## KOLEJNOSC WYKONANIA

1. Dziura 3 (SO(4)) — 1h, najlatwiejsza, otwiera droge
2. Dziura 2 (Trotter-Kato) — 2h, srednia trudnosc
3. Dziura 1 (polymer locality) — 4h+, najtrudniejsza, decydujaca

## KRYTERIUM SUKCESU

Po zamknieciu 3 dziur:
- Kazdy krok ma PELNY dowod w tekscie (nie sketch, nie citation as proof)
- Zero circular reasoning
- Limitations section mowi: "W5 (asymptotic completeness) remains open"
  i NIC wiecej

## CO NIE JEST POTRZEBNE

- GPU numerics (juz sa, jako validation)
- TOE paper (oddzielny projekt)
- Nowe modele fizyczne
- LSZ / hadrons / GUT
