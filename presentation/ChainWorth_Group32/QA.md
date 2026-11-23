# Defence Q&A — Group 32

**How to use this file.** Each item is one anticipated question, then a short answer, then the evidence (file + column). Where the question lands, the answer says so in the first line — a defence that concedes nothing reads worse than one that concedes precisely.

**Two questions changed the artefact, not just the answer.** A1 (what actually enters each arm) forced claim 01 to be rewritten from "chain information does not pay" to "chain-aware arms do not pay"; B9 (tenfold vs 74×) produced a number we had not reported before — the first tenfold step has a *positive* slope. Both are marked **[fixed]**.

**Where to look first.** Slide 15 ("What would change our mind") now states the refutation conditions for all four claims, plus three things that would *not* move us — use it before answering any question in groups A or B.

Numbers quoted here are the ones in `docs/研究报告.md` after the 2026-11-18 revision; every one of them traces to `results/aggregate/` and, where derived, to `defense_checks.json`.

---

## A. Construct validity

### A1 · All three arms take the same repeat-unit string — did any of them actually receive chain information? **[fixed]**

**Answer: no, and claim 01 has been reworded because of it.** Mechanically: A2 receives 2,253 columns (205 RDKit descriptors plus fingerprint bits) computed from one repeat unit; A3 receives the 2D graph of that same repeat unit (atoms, bonds, no neighbours); A4 receives the tokenised PSMILES string through a frozen polyBERT encoder (600-d) and a ridge head. **No column, token or edge in any arm carries chain length, copolymer ratio, block length, molecular-weight distribution, tacticity or morphology.** One precision, because it is the first thing a polymer chemist will check: an alternating copolymer's repeat unit is a *dimer*, so its periodic sequence really is in the string (`alternating_Cp.csv` carries both monomers in one row) — what is missing there is only *n*. For random and block copolymers the ratio and the block length are separate columns and never enter. So Λ is not "chain information versus no chain information" — it is "a chain-aware architecture versus a descriptor baseline, on identical inputs".

That is a narrower claim, and claim 01 now reads *"chain-aware arms mostly do not pay"*, not *"chain information does not pay"*. If a reader wants the stronger reading, this experiment cannot supply it — H3 is exactly that gap and it is unanswerable here (A2).

### A2 · S13 says A5 cannot be constructed, S14 says information-vs-architecture is unanswerable, yet claim 01 says chain information does not pay. Which sentence do you retract?

**Neither sentence is retracted; the subject of claim 01 is narrowed.** A5 answers a specific question — *at a fixed architecture, does injecting chain covariates help?* — and it cannot be built. Claim 01 answers a different, empirical question — *do the two chain-aware arms we did run beat the descriptor baseline?* — and the answer is no in 12 of 19 cells. The contradiction only appears if claim 01 is read as being about "information in the abstract", which is the reading A1 forbids. We rewrote 01 to say "chain-aware arms" so the two sentences no longer overlap.

### A3 · Is "not constructible" about inputs or about labels? PSMILES can be turned into oligomers in one RDKit call. Should "boundary, not debt" become "label gap"?

**Concede the naming: it is a label and field gap, and we now say so.** Two things are missing, and the second is the binding one. First, there is no chain-length field: `psmiles.parquet` has 18,152 rows with no repeat count, and the 18,036 rows carrying wPSMILES all share the same repeat field `|1|` with one identical weight block. Second — and this is why a one-line RDKit pentamer does not rescue it — **the labels are per repeat unit and do not vary with chain length**. A pentamer has no target in this release; feeding one would mean inventing the y.

So "not constructible" is precise at the release level and rhetorical at the conceptual level. We keep the machine status string (`not_constructible_data_unavailable`, already in `main_table.csv` and frozen) but the prose now reads "no chain-level field and no chain-level label in this release". "Boundary" means *the boundary of what this dataset can decide*, not "nobody could ever build such a dataset".

### A4 · In the cells both arms ran, four of eight flip sign, and the largest positive sits opposite A4's large negative. Isn't Λ measuring encoder noise?

**Concede the disagreement; reject the inference from it.** Eight tasks have both arms. Four flip sign — but two of those four are cells where **both** arms are within ±0.05 (modulus −0.016 vs +0.028; permeability −0.008 vs +0.003): the sign of a near-zero number is not a finding. The two genuine disagreements are `pb26_homopolymer_density` (+0.638 vs −0.532) and `pb26_polymetrix_tg` (+0.005 vs −0.473), both on 300–1,440-row tasks where a frozen encoder with a linear head has little to work with. That is arm capacity, and we do not hide it: **the two arms are never pooled into one "chain" estimate**, every cell is reported per arm, and §7 limit 9 lists this explicitly.

What the disagreement does refute is the reading "Λ is a property of the data alone". We never claimed that; Λ is defined per (task, arm, unit baseline), and the only cross-arm statement we make is that the majority of cells are negative in each arm separately.

### A5 · "Worth" sounds additive; Λ is a ratio that can reach −1.5 and diverges as the denominator vanishes. Which quantity is "worth" statistically?

**Λ is a residual-variance ratio, not a value.** With R²_chain and R²_unit measured on the same test rows, Λ = 1 − (1 − R²_chain)/(1 − R²_unit): the share of the unit arm's *remaining* variance that the chain arm recovers. In population terms that is 1 − E[Var(Y|X_unit, X_chain)] / E[Var(Y|X_unit)] — an R²-based analogue of a variance-reduction estimand. It is not additive across tasks, it is unbounded below, and it is undefined when the denominator goes to zero; the title's "worth" is rhetorical, and the estimand is the ratio.

Why not Shapley or conditional mutual information: both need the chain variables as inputs (a coalition structure, or a Z to condition on). This release has none — no Mw, no sequence, no tacticity column. The variance decomposition we *can* run is the one-variable case, and that is Λ. If a chain covariate existed, the natural next estimand would be the same ratio with Z added, and Shapley would become meaningful.

### A6 · Λ's baseline is a 30-trial-tuned A2. Is the "gap" systematically negative by construction?

**Concede the direction; it is against us, not for us.** A2 was tuned with a pre-registered protocol (30 trials, Murcko 5-fold CV **inside the training rows only**, `search_seed` fixed), so there is no test leakage — but the asymmetry is real: A2 gets 30 trials, A3 runs chemprop's official defaults untuned (depth 3, hidden 300, dropout 0, batch 64, max lr 1e-3, ≤100 epochs, patience 15). A stronger unit arm lowers Λ. So part of the negative sign may be tuning budget, and the sign **could** change if A3 were tuned with equal budget.

Two things bound the damage. A4 is closed-form ridge with no tuning at all, and it is also negative in 5 of its 8 cells; and the effect is listed as §7 limit 2 with the direction stated ("this biases toward our conclusion, not away from it"). Equal-budget tuning of A3 is the first experiment we would run with one more week (E1).

---

## B. Criteria and statistics

### B1 · n = 6 Spearman, ρ = 0.239 — what is the p-value? And where does the 0.5 threshold come from?

**Exact permutation p (720 permutations, two-sided): 1.00 for A3, 0.78 for A4.** So there is no evidence of any association, in either direction, and the test could not have produced evidence at this n.

The 0.5 threshold is **not a critical value** — it is the effect-size bar pre-registered in the proposal (§H1: "ρ < 0.5, or any inversion spanning ≥ 2 tiers"). At n = 6 the critical value for p < 0.05 is |ρ| ≥ 0.866, so the pre-registered bar is *stricter* than significance and has no sampling justification of its own. We now report both and label the 0.5 line as a pre-registered effect-size criterion (`defense_checks.json` → `h1`).

### B2 · Deleting one point flips the sign — why narrate leave-one-out instead of a task-level bootstrap or permutation?

**Because a task-level bootstrap with six units is degenerate — and we now show that instead of asserting it.** Resampling six tasks with replacement gives ρ intervals of **[−1, +1]** (A3) and **[−0.89, +1]** (A4); more than half the resamples are duplicates. The permutation test is the right tool at this n, and it is now reported (B1). Leave-one-out stays as a *robustness display*, not as inference: it answers "how much of this ρ is one task", which is a different question from "how uncertain is ρ".

The honest summary is that no method gives a tight interval from six units; what changes with the method is only how visibly that shows.

### B3 · H1 is triggered by a point you yourself call undecidable. Can that be "refuted"? Why not say "failed the pre-registered criterion, but this is not evidence about stratification"?

**That is exactly what the prose says; only the verdict token is shorter.** `h1_ordering.csv` stores `H1_refuted`, which means *the pre-registered rule was violated* — a statement about the rule, not about the world. §5.2 says in as many words: "按预注册判据被证伪，但这个证伪由一个不可判点主导，不足以支撑'分层与 Λ 无关'的强主张。这是'没有能力检验'，不是'检验了没通过'." We keep the token for machine continuity and refuse the inferential reading in text.

### B4 · The n-matched interval [−0.042, −0.024] is only 0.018 wide. What is the resampling unit — rows, or folds and seeds?

**Cells, not rows: the unit is (task, seed), and that is also why the interval is too narrow.** Per level there are two tier-1 tasks × three seeds = six cells per side; the bootstrap draws those six cells with replacement, then averages. Test rows are never resampled, so within-cell sampling error is excluded — deliberately, because two arms share the same rows and their difference is paired.

The consequence must be stated with the number: **the interval is conditional on these two tasks**, both from tier 1. It contains seed-to-seed variance but not task-population variance, so 0.018 is a lower bound on uncertainty, not an estimate of it. With two tasks there is no way to estimate the latter; that is why the claim is "within these two tasks, at matched n, tier 1 stays lower".

### B5 · Δ = −0.033 on a scale that runs from −1.479 to +0.638 — is that significant, or important?

**Distinguishable but small; the two words are reported separately.** Against the full 19-cell range (2.1 units) it is 1.6% — negligible. Against the comparison actually being made (tier-1 mean ≈ −0.12 vs tier-2 mean ≈ −0.02) it is about a third of the level difference, which is what makes it informative about *mechanism*: the confound is not the whole story. It is not an effect-size claim, and the report does not use it as one.

### B6 · κ flips to +0.398 when you keep only the two stable levels (300 and 1,000 rows), and 23 of 40 cells have a denominator below 0.05. Is conclusion 03 built on an unstable sign? And why no ΔR² slope?

**Concede: the sign is not robust, and we now report the ΔR² slope beside it.** Three readings of the same curve, all in `defense_checks.json` → `scale`:

| segment | Λ slope | ΔR² slope |
|:--|--:|--:|
| 300 → 3,000 (first tenfold) | **+0.073 / dex** | −0.022 / dex |
| 3,000 → 22,242 (tail) | −0.202 / dex | −0.013 / dex |
| 300 → 22,242 (full, pre-registered) | −0.086 / dex | −0.018 / dex |

The negative κ is a full-range average, and it lives entirely in the tail; in the first decade Λ *rises*. The two-stable-level variant (+0.398) is a third reading on the same ruler — a slope, not a ratio — and it agrees with that first step. Conclusion 03 is therefore written ruler by ruler — "on the ratio ruler it has not halved; on the absolute one it has" — for one system, one arm, one range, and not as "data does not help".

### B7 · H2 is a conjunction: κ < 0 holds, ratio 0.650 > 0.5 fails → "not refuted". Then what does "not confirmed" mean, if the protocol has no confirmation criterion?

**It meant nothing, and the phrase is gone.** The protocol has one rule — refute if both conditions hold — so the only licensed outcome is *refuted* or *not refuted*. "Not confirmed" was loose language that implied a two-sided test we never specified. The deck now says: "the protocol has a refutation test and no confirmation rule, so not refuted is not support." If we wanted a confirmation criterion it would have to be pre-registered separately (for instance "Λ halves and the interval of the ratio excludes 0.5"), and we do not back-fit one now.

### B8 · S14's claim 03 was "More data does not buy it back" while S11's verdict was "not refuted, not confirmed". Isn't that the upgrade you warn against?

**Yes — that was the same error the report criticises, and it is fixed.** Claim 03 now reads "More data **has not** bought it back", with the tenfold caveat in the body. The general rule, which we had written down and then violated, is: *never upgrade "could not test" into "no"*.

### B9 · κ spans 300 → 22,242 (74×), but the question and criterion say "tenfold". The tenfold point is 300 → 3,000, where Λ rises. Which was pre-registered? **[fixed]**

**The verdict uses the pre-registered full range; the tenfold segment is reported because it contradicts the narrative.** The criterion table in the proposal says "largest n versus smallest n", and that is what `kappa.csv` computes (κ = −0.0863). The word "tenfold" in the hypothesis prose, however, points at 300 → 3,000, where Λ goes 0.295 → 0.368 — a **positive** slope of +0.073/dex. We had not reported that number before this review; it is now in §5.3, in the deck (slide 11), and in `defense_checks.json`.

So: verdict unchanged, but the sentence "more data does not buy it back" cannot be supported by the tenfold segment — only by the tail, and only on average.

### B10 · Homopolymer cannot report κ, so κ = −0.0863 and the 0.650 ratio come from one system, one arm, one metric. Can n = 1 support the claim?

**No, and the claim is scoped to that.** The alternating copolymer supplies 5 levels over 1.87 dex with 5 folds; the homopolymer supplies 3 levels over 0.68 dex with the top gap at 0.158 dex, which fails the pre-registered spacing rule (D-74) and is therefore reported as a two-point line, not a slope. A4 refuses n-levels by construction (`polybert.train_one()` raises `ArmError`), so the chain arm is A3 only.

Hence: **one system, one arm, one metric**, stated in §7 limit 8 and in the conclusion itself. "On the ratio ruler it has not halved" is a statement about this system under this protocol; it is not a general law about polymers, and we do not present it as one.

### B11 · H2′ is post hoc and is the only ruler that halves. How does a reader know you did not pick the ruler after seeing the result?

**Partly by construction, partly by consequence — and the consequence cuts against us.** By construction: H2′ sits in its own section (§5.6), never enters the verdict, and §6.3 keeps `H2_not_refuted` as the judgement; the boundary paragraph states the pre-registration that would be needed. By consequence: H2′ does not flatter the thesis, it **contradicts** it — under the absolute ruler the gap halves, i.e. data *does* buy something back. A ruler chosen to rescue a claim would not be the one that undermines it.

What we cannot prove is that we would have looked at ΔR² had the ratio behaved. We can only say what a reader should do with it: treat H2′ as a pre-registration candidate, not as evidence.

### B12 · At 22,242 rows ΔR² = 0.0066, the same order as the floor. Are both rulers insufficient there?

**Yes at that level, and the number is now written next to the floor.** Against the two PB26 tasks' floors (0.0015–0.0024) it is 2.8–4.4×; against the cross-task median floor (0.0046) it is 1.4×. "Halved" is therefore true arithmetically and weak inferentially: at 22k rows the gain is one to four floors wide. Both rulers agree on "the gain approaches the floor"; neither can decide "bought back" at that level.

### B13 · How is the noise floor computed? "6 of 8 tasks clear it" has no test, and what composes the 48 rows?

**Floor = the largest |ΔR²| produced by columns that carry no information.** Same fold, same training rows, same seed, same test rows; add three constant columns (condition `constant3`) or permute the three highest-variance descriptors (`shuffled3`); take the **larger** of the two |ΔR²| values per task. 48 rows = 16 evaluation units × 3 conditions (baseline + 2); one seed, so no interval.

"Clear it" is a magnitude comparison, not a test — there is no p-value to give, because each cell is a single fit. And the count is **5 of 8**, not 6: the earlier table took the *smaller* of the two conditions on `pb_t2_modulus` (0.0015 instead of 0.0195), which was inconsistent with the other seven tasks. Fixing the convention moves the median floor from 0.0022 to **0.0046** and the cleared count from 6 to 5 (D-81).

### B14 · n* ≈ 56 sits between the design points 30 and 60. Is that interpolation or fit? Why two digits?

**Interpolation, in log n, between two design points — and two digits is a formatting artefact.** The observed levels are 30, 60 and 120 training rows; the crossing is found by linear interpolation of ΔR² against log₁₀n between 30 and 60, giving 56.1 (Tg) and 59.3 (dielectric constant). There is no fit, no interval and no third point, so the only defensible reading is **"between 30 and 60"**.

The deck now says that; the report keeps the interpolated value for traceability and states the bracket beside it. Tm never changes sign inside its window, so no n* is reported at all.

---

## C. Data and experimental design

### C1 · ρ(tier, log n) = −0.956 — if tier and n are nearly the same variable, is H1 unidentifiable by design?

**In this release, essentially yes — and that is conclusion 2.** With six tasks and a tier–n rank correlation of −0.956, the tier effect and the sample-size effect are nearly collinear; what is left to identify H1 is the non-monotone part of the ordering. We therefore do not claim "tier does not matter"; we claim "the pre-registered ordering criterion is violated, and the violation is driven by a point we cannot decide".

The matched-n probe (C2) is the closest thing to identification available here, and it only covers tier 1 versus tier 2.

### C2 · Tier 3 cannot be matched upward, so "the break is inside tier 3" is untestable. Does the supplement change H1 at all?

**No — it changes the interpretation, not the verdict.** Matching solved tier 1 versus tier 2: at 497 and 723 training rows, Δ = −0.033 and −0.087, both intervals excluding zero, so the confound is not what produces ρ ≈ 0. But tier 3 has 279 and 228 rows; there is no smaller-n tier to match *up* to, so the internal split of tier 3 (+0.068 for IWLT versus −1.479 for dielectric loss) stays descriptive. H1's verdict is unchanged; what changed is that "the two effects cancel" is no longer an available explanation.

### C3 · PB26 uses two tasks and a single seed, while PolyBench uses three. Why is the variance treatment inconsistent?

**Because the releases differ, not because we chose per hypothesis — and the cost is stated.** PB26 ships an official 5-fold protocol; the scaling experiment is that protocol at five n-levels, single seed, because the release provides one split family and the point of the experiment is the official manipulation. PolyBench ships a single 70/30 split, so seed replication is the only way to see split variance. Using one convention for both would have meant either inventing splits for PB26 (breaking the "official" claim) or dropping seeds for PolyBench (losing the only variance estimate there).

The consequence is in `docs/研究报告.md` §7 limit 4: κ has no GNN-initialisation variance; fold-level standard error is 53–60% of the bootstrap half-width, and adding it leaves the interval entirely negative.

### C4 · A4 does not consume n-levels, so H2 has one chain arm. Given that arms disagree on sign, why does one arm represent chain information?

**It does not represent "chain information"; it represents A3 at five data sizes.** This is the same concession as A4/A6: the two chain arms are not interchangeable, which is why no cell is pooled across arms. H2's result is scoped to A3 by construction, and §7 limit 8 says so. If A4 could consume n-levels, the honest experiment would be to run both and report both κ values; it cannot, and faking the levels for it would be worse than the gap.

### C5 · 65 of 4,897 checks are "exempt". What does exempt mean, and why may exempt rows enter the table?

**Exempt is a registered reason, not a pass.** Two reasons exist: `MATERIALISATION_GAP` (the PolyBench26 architecture splits are deliberately not committed, D-57, so on a fresh clone the file is missing — the row is exempt and the rebuild command is printed) and `DERIVED_TABLE` (15 diagnostic tables that carry no `run_id` by design, each registered with its reason). Exempt rows are excluded from the pass-rate denominator but must appear in the table; the gate is **zero fail**, and "missing but materialised elsewhere" is exempt while "present but hash-mismatched" is still fail.

They do not "enter the main table": every number in `main_table.csv` comes from a run record, and every run record is checked. Exempt rows are diagnostics about splits and derived tables, not evidence for a cell.

### C6 · All six PolyBench tasks have a single label source. Is "every cross-task claim is within one measurement protocol" on the conclusion page? **[fixed]**

**It was only in the limits section; it is now on the conclusion page.** Six of six main tasks are `experimental`, so the cross-source τ and leave-one-source-out analysis are `not_evaluable_single_source`. That means every "across tasks" sentence in this report is a statement inside one measurement protocol, and — by R1 — not comparable across releases either. The conclusion block now ends with that boundary instead of leaving it to §7.

### C7 · 962 "fitted runs" includes 48 A1 mean predictors. Does the headline number need changing? **[fixed]**

**Yes: 962 records, 914 fits.** A1 is 16 evaluation units × 3 seeds = 48 records, and it predicts the training mean without fitting anything. All documents now say "962 run records / 914 fits", and the deck's title slide says "914 fits". The distinction matters because "962 models" would overstate the compute by 5%.

### C8 · Three benchmarks, one hypothesis each, but R1 forbids cross-row comparison. Why three?

**Because each release supplies a different manipulation, and none supplies all three.** PolyBench is the only release that publishes the three explanatory tiers, so H1 can only live there. PB26 is the only release with an official data-size manipulation and an official architecture-transfer task, so H2 and the architecture sub-clause live there. OpenPoly is the only release with an open per-level n-sweep, so H3/H4 live there. R1 forbids *pooling* across releases, not using three of them; one benchmark cannot host all three because the manipulations do not co-exist in any single release.

---

## D. Precision of statement

### D1 · The symbol n means 300–22,242 in one figure and 30–240 in another, and slide 3 only shows the first.

**Concede: two axes share one symbol, and one slide does not say so.** In PB26 the x-axis is **training rows** at the official levels (300, 1,000, 3,000, 10,000, 22,242). In OpenPoly H4 it is **labels per property** at the self-built levels (30, 60, 120, up to 355/134/235 in the window). They are not comparable, and the two figures are never placed on a shared axis.

The deck's H4 page labels the windows as they are — **30–355, 30–235, 30–134** — and flags that they are of different width, so the three n\* are not comparable with each other. The report's §5.5 states them property by property. A reader who compares the two x-axes is comparing a 74× span with an 11.8×, 7.8× or 4.5× span.

### D2 · 23 of 40 cells are flagged unstable, but the 12/19 negative count was not filtered the same way. What is the recount?

**The recount is the same number: 12 of 19.** `lambda` is left empty whenever `1 − R²_unit < 0.05`, so every cell mean is already computed on stable rows only. Filtering again removes no cell: the only place the rule bites is `pb26_homopolymer_density`, where 12 of its 18 rows are dropped and the remaining 6 still report Λ. Per-cell counts are in `results/aggregate/defense_cells.csv` (`n_stable`, `min_denominator`), so the claim can be checked cell by cell.

### D3 · "Most stable negative", "75 computable rows", "6 of 15" — what do these mean?

**Three different denominators, now spelled out.** "Stable" = the cell's denominator `1 − R²_unit` is at least 0.05 on the rows that enter the mean. "75 computable rows" for `op_tm` = 3 seeds × 5 folds × 5 levels, all with reportable Λ. "6 of 15" for `pb26_homopolymer_density` = 6 rows out of 15 survive the denominator rule; the other 9 are not "missing data", they are rows where A2 already exceeds R² 0.95 and a ratio would be noise.

### D4 · Slide 14 calls op_tm "the data-richest task" while the same page says it has 75 computable rows, and dielectric loss has 279. **[fixed]**

**That was an error of fact in the summary line, and it is corrected.** `op_tm` is the **most stable** negative (Λ = −0.859, run-to-run sd small, 75 computable rows); `pb_t3_dielectric_loss` is the **deepest** (Λ = −1.479 over 279 rows, with A3's own R² at −0.009). They are two different cells, and "richest" applied to the 75-row cell was wrong. The deck and both conclusion documents now separate "deepest" from "most stable".

### D5 · Slide 2's quote box says "Paraphrased from PolyBench" — is that a verbatim quotation or a paraphrase?

**A paraphrase, and it is labelled as one** — both on the slide and now in the report (§1.1). The box condenses the benchmark's own limitations paragraph; it is not a quotation and should not be cited as one. The slide keeps the label; the report's blockquote gained an inline "转述，非逐字引用" line, because a blockquote with no label reads as a direct quotation.

### D6 · H4 is explicitly "descriptive only, no significance test". Why is it a hypothesis at all?

**It is a labelled observation that carries the H4 index for continuity with the proposal.** Keeping the number makes cross-references stable; calling it a hypothesis would overstate it. The report's §3 now adds a note: "H4 编号沿用提案只为对照；它不进'被检验的假设'计数——凡写'三个假设'，指的是 H1/H2/H3." The deck's H4 block says "descriptive" in its own subtitle.

---

## E. The usual closing questions

### E1 · One more week, half the budget left — which experiment?

**Equal-budget tuning of A3, on the four cells where the sign matters most.** The single largest threat to claim 01 is that A2 has 30 trials and A3 has none (A6). A 30-trial search under the same Murcko-CV protocol on `pb_t1_tg`, `pb_t1_tm`, `pb_t3_dielectric_loss` and `pb26_polymetrix_tg` costs roughly 2–3 GPU-hours on the 4070 (measured: 538 A3 runs took 2.66 GPU-h, most expensive single run 164 s). If the tuned A3 is still negative in most cells, claim 01 becomes robust to its strongest objection; if it flips, we learn that before a reviewer does.

Second choice would be the scramble control for H3, which is cheaper but cannot rescue an unconstructible arm.

### E2 · Which number are you least sure of, and which conclusion falls if it is wrong?

**The 12-of-19 count, because it depends on A2 being tuned and A3 not.** If equal-budget tuning moved A3 above A2 in, say, half the cells, claim 01 collapses and H1's premise (that Λ varies by task) changes with it. The number I am second-least sure of is κ's sign, which already fails on the first decade and on two alternative filters — which is why conclusion 03 is written with three qualifiers rather than one.

### E3 · Under what specific, measurable condition would your main conclusion be overturned?

**Two, both operational.** (i) With A3 tuned at equal budget, Λ ≥ 0 in at least half of the 19 cells on the same splits and seeds — that would overturn "chain-aware arms do not pay". (ii) A release in which chain covariates exist and A5 recovers at least half of the residual gap at matched n — that would overturn the stronger reading that the missing variance is not chain-level. Neither is testable in this data; both are stated so that someone else can test them.

### E4 · If MAE or RMSE replaced R², would Λ flip sign?

**In some cells, plausibly — and we have not run it, so we say so.** Λ is defined on R², which normalises by the label variance; RMSE is in label units, so a "relative RMSE reduction" would be a different normalisation and the ordering of cells could change. The cells most at risk are those where R²_unit is near zero (`pb_t2_permeability`, R² = −0.002): there, a ratio defined on R² is already flagged unstable, and an RMSE-based version would be numerically better behaved but would answer a different question. This is untested, and it is on the future-work list rather than in the results.

### E5 · A negative result with no new method — what is the value?

**Three things a negative result can have, and this one has them.** (i) *Pre-registration*: four criteria were fixed before the first fit, and two of them are reported as failed (H1) or unmet (H2) rather than reinterpreted afterwards. (ii) *Reproducibility*: 962 run records with hash-frozen configs and splits, 4,897 provenance checks with zero failures, bit-identical reruns on the same GPU when both determinism switches are on. (iii) *A boundary that is stated*: A5 is not "pending", it is not constructible from this release; the noise floor, the n\* bracket and the tenfold-versus-full-range discrepancy are all reported with their numbers.

What we do not claim is a method. The contribution is a measurement, and the measurement says where chain information did **not** help — which is the part of the map that is usually left blank.

### E6 · Who ran the 538 A3 records? Is the contribution table by workload or by module?

**The GPU arm — 538 A3 runs plus the polyBERT embeddings — was run by the member responsible for the GPU workstream (Yihui), on one RTX 4070; the contribution table is by module, and each row is comparable in scope.** The table deliberately does not print record counts: 538 runs that each take ~5 s and 15 runs that each need a materialised split are not comparable by count, and printing counts invites exactly the misreading the question implies. The footer of that page is the operative sentence: one protocol, shared by all four.

### E7 · How do the course's four topics connect to this work?

**One sentence on slide 4 is the whole connection, and it is thin on purpose.** The four course topics are inorganic crystals, PDE solution operators, physics-constrained PDE and LLM scientific reasoning. Our question is the same question in a different material class: *what does the representation contain, and what is therefore missing?* For crystals, composition plus unit cell is (nearly) complete; for polymers it is not, and the extra level — the chain — is what every public benchmark compresses away. The failure mode we measure is the mirror image of the LLM topic: there the model knows a lot of chemistry but not the measurement; here the representation knows the repeat unit but not the chain.

---

## One-page cheat sheet

| If they ask | Say this first |
|:--|:--|
| Which arm got chain info? | None; claim 01 now says "chain-aware arms" |
| Is A5 a boundary or a gap? | A label/field gap in this release |
| Did the two arms agree? | 4 of 8 flip; 2 of those are near-zero cells |
| Where's the p-value? | 1.00 / 0.78 exact permutation |
| Why 0.5? | Pre-registered effect-size bar, not a critical value |
| Is κ robust? | No — positive in the first decade, negative in the tail |
| Tenfold or 74×? | Verdict uses 74× (pre-registered); tenfold reported and contradicts the prose |
| One system? | Yes — alternating copolymer, A3 only |
| What changed today? | 914 fits, 12/19 stable, floor 0.0046 / 5 of 8, tenfold +0.073 |
