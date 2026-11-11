# How Much Is a Chain Worth?
## Chain-information recovery in polymer property prediction, and whether more data buys it

| Item | Content |
|---|---|
| Version | **v1.3** — v1.1 added §2.3 and re-scoped H4; v1.2 added §3.1 (candidate task IDs, Appendix A), corrected the PolyBench evaluation protocol (fixed 70/30, not k-fold), and reconciled every run count that followed from it; **v1.3 retitles the proposal from "The Repeat-Unit Ceiling" to "How Much Is a Chain Worth?", brings the filename into line with the content version, and replaces the loose "ceiling" metaphor with the defined quantity of §4; the internal proposal code is dropped from the title and the filename** |
| Date | 2026-11-11 |
| Course topic | **Self-proposed topic** (per *"YOU ARE MORE THAN ENCOURAGED TO PROPOSE YOUR OWN TOPICS"*), situated next to Topic 1 in the same scientific family |
| Team size | 4 (roles in §9) |
| Presentation slot | 2026-11-19 (8 working days) **or** 2026-11-26 (15 working days); both schedules in §7.2 |
| One-sentence question | **If you compress a polymer chain into a single repeat unit, what fraction of its property variance becomes unexplainable — and does that fraction shrink when you add 10× more data?** |
| Compute requirement | **One RTX 4070 (12 GB) is sufficient.** Mandatory GPU core is **≈13–28 GPU-h** (0.6–1.2 days continuous); we quote the **19–36 GPU-h envelope** everywhere, which is that core at the top of its range plus the optional architecture arm (§7.4). Two cards available, second optional. CPU-only fallback exists but weakens the central claim |
| Estimated effort | 189 person-hours from scratch; **≈105 pH with agent-assisted coding**; ≈95 pH remaining at submission |

---

## 1. Motivation and why this is a legitimate new topic

The four topics in the course cover inorganic crystals (T1), PDE solution operators (T2), physics-informed PDE solving (T3), and LLM scientific reasoning (T4). **Organic and polymeric materials — the largest class of materials by production volume and the one where "molecular design" actually happens — are not covered by any of them.**

That coverage gap alone would be a weak argument for a new topic. The stronger argument is structural:

> **Polymers add a level of representation that crystals do not have: the chain.** A crystal can be described by composition plus a unit cell. A polymer cannot be described by its repeat unit alone — molecular-weight distribution, sequence (random / block / alternating), tacticity, morphology, and processing history all influence measured properties, and **none of them are expressible in a PSMILES string.**

This is not our speculation. It is the PolyBench authors' own stated limitation:

> *"The current release evaluates repeat-unit-centered polymer property prediction and does not explicitly encode higher-order factors such as molecular-weight distributions, tacticity, morphology, processing history, device architecture, or measurement protocol."*

And they act on it, grouping their 39 tasks into three provisional interpretation tiers with condition-sensitive tasks flagged as exploratory. **The limitation is acknowledged, the tiering is given, but the size of the gap has never been measured.** That is the opening.

### 1.1 What is already published, and what is not

| Already established | Not established |
|---|---|
| No single method dominates across polymer tasks (PolyBench, >10,000 evaluations) | **How much** of the variance the repeat unit leaves behind is recoverable — Λ, in comparable units |
| Descriptor + tree ensembles are strong baselines; XGBoost leads on **all 26 OpenPoly properties at n ≤ 443** (§2.3) | **Where** that gap is widest, as a function of property class |
| Graph models give the lowest errors and retain their advantage across training-set sizes (PolyBench26) | Whether the gap is **movable by data** at all |

We do not claim to beat the published baselines. We claim a measurement they do not make.

---

## 2. Benchmarks

| Benchmark | Scale | Official protocol | Role |
|---|---|---|---|
| **PolyBench** | 39 tasks / 42,169 records / 14,717 repeat units / 7 representation families / 30 baselines | Fixed train/test splits, interactive leaderboard | **Anchor A — task count.** Supplies the six main-table tasks, each with its own official 70/30 split |
| **PolyBench26** | ≈250k property datapoints / 8 properties (experimental, DFT, MD) | **4 official tasks** (in-distribution, dataset-size scaling, repeat-unit complexity, architecture transfer) + official 5-fold scaling splits | **Anchor B — mechanism.** Tasks 2 and 4 are, by construction, exactly the two manipulations we need |
| **OpenPoly** | **741 distinct PSMILES structures / 3,985 structure–property annotations** across 26 experimental properties (counted from the released CSV on 2026-11-11, see §2.3) | Repository train/test scripts | **Data-scarcity arm.** Median property has ~115 labels; only Tg (443) and the three dielectric columns (295) exceed 200 |
| MoleculeNet / PCQM4Mv2 | 17 datasets / 3.7M molecules | random, scaffold / OGB splits | **Cross-hierarchy control only.** Not in the main table (PCQM4Mv2 test labels are withheld) |

> **Naming caution, to be stated once in the talk.** *OpenPoly* (Fudan, 3,985 experimental records, **26 properties**) is not *OPoly26* (Meta/FAIR, 6.57M DFT calculations for MLIP training), and neither is *PolyBench26* (LLNL, ~250k datapoints, 8 properties). Be aware that a fourth, unrelated resource also carries the name: a *PolyBench* of ≈125k LLM question–answer pairs for polymer reasoning. **Four unrelated resources, similar names** — cite by DOI and scale, never by name alone.

### 2.1 What each benchmark is for — and why four is not "four benchmarks"

The right question is not "how many benchmarks" but **"how many official manipulations do we need, and does each hypothesis live inside exactly one release?"** Counting resources understates the design; counting *load-bearing* resources states it correctly.

| Benchmark | Carries which hypothesis | Load-bearing? | Why it must be *this* one |
|---|---|---|---|
| **PolyBench** | **H1** (tier ordering) | **Yes** | The three provisional interpretation tiers exist only here. No other source publishes per-task tier labels, and H1 is a claim *about those labels* |
| **PolyBench26** | **H2** (κ ≈ 0); H1's architecture sub-clause | **Yes** | The official `scaling_splits` and the official architecture-transfer task are the only *official* manipulations of data scale and chain architecture. Self-built subsampling or home-made wPSMILES weighting would make the result our artefact, not a property of the field |
| **OpenPoly** | **H4** (what sets the crossover) | **Boundary only** | 24 of its 28 property columns sit below n = 200, so the sub-200 regime is real — but only three columns carry enough labels to scan *within* (§2.3) |
| MoleculeNet / PCQM4Mv2 | none | **No** | Cross-hierarchy context in the talk only. PCQM4Mv2's test labels are withheld, so no claim can rest on it |

**2 load-bearing + 1 boundary + 1 non-load-bearing.** Four resources, two of which carry results.

**Three rules that keep the protocol fixed despite the heterogeneity.**

- **R1 — One benchmark per claim; no cross-benchmark pooling.** Every hypothesis is accepted or rejected *inside a single release*. Λ is dimensionless and therefore *comparable* in principle, but comparable ≠ poolable: a Λ computed under PolyBench's protocol and one computed under PolyBench26's may not be ordered against each other, and we never do so. All ordering claims (H1) are within-release.
- **R2 — Ordering needs ≥2 tasks per release.** An ordering with a single task per source is an anecdote. If a source shrinks below two usable tasks, its ordering is reported as an observation and excluded from the Spearman ρ.
- **R3 — Every Λ value carries a four-tuple.** `(benchmark, task ID, split version, unit)`. A number missing any element does not enter a figure.

### 2.2 Source-heterogeneity audit (the answer to "won't the conclusions be unstable?")

Multiple releases mean multiple label definitions, measurement protocols and split constructions. That is a real threat to an ordering claim, and it is audited rather than asserted:

| Check | Procedure | Decision rule |
|---|---|---|
| **Within-release stratification** | Compute Λ ordering separately by label source (experimental / DFT / MD) | Kendall τ ≥ 0.6 between source-specific orderings before pooling *within* a release; otherwise report per source and claim nothing about the pooled order |
| **Leave-one-source-out** | Drop one label source, recompute the ordering | If the ordering flips, that ordering is not written as a conclusion |
| **Irreducible protocol conflicts** | Same physical quantity measured by different methods (e.g. Tg by DSC vs DMA) | **Do not convert.** Drop the task, record the reason in the appendix — a conversion we cannot defend is worse than a missing task |
| **Label harmonisation** | 7 pH, item 2 of §7.1 | Units resolved by rule, not by guess; every conversion logged |

> The honest framing for the talk: **we are not claiming that six tasks across two releases give one universal number. We are claiming that four hypotheses, each tested inside one official protocol, point the same direction.** If they do not, we report which one broke — that is a result too.

### 2.3 Data-availability audit (performed before the task list was frozen)

Three claims in an early draft did not survive contact with the released files. Each was checked by downloading the data and counting, not by reading the abstract.

| Resource | What was claimed | What the files actually contain | Consequence for the design |
|---|---|---|---|
| **OpenPoly** | "3,985 polymer entries" | **741 rows / 741 distinct PSMILES structures**; the 3,985 figure is the number of **structure–property annotations** (sum of non-null cells = 3,985 exactly). Verified by direct count | Correct the headline number everywhere; scale statements must say "741 structures, 3,985 labels" |
| **OpenPoly** | Per-property sample sizes are "dozens–hundreds" | Only **4 of 28 property columns exceed n = 200** (Tg 443; dielectric electronic/ionic/total 295 each). **24 of 28 are below 200**; the smallest is methanol permeability at **n = 30** | "Use the scarcest properties" is not an executable design — at n = 30 a 5-fold split leaves 6 test points. Task selection changed to Tg / dielectric total / Tm (§3) |
| **OpenPoly** (second file) | — | `final_polymer_property_counts_fromliterature.csv` reports Tg **58,755** and refractive index **7,878** — literature record counts *before* structure matching, 1–2 orders of magnitude above the training table | **Use the 741-row table only.** Quoting the counts file would overstate available data by up to 50× |
| **PolyBench26** | "scaling experiment" | Official training sizes are **300 / 1000 / 3000 / 10000** (alternating MD_5000); homopolymer MD_5000 has only **300 / 1000** (1,715–1,800 rows total); PolyMetriX Tg has **300 / 1000 / 3000**. 5 folds, base seed 42, val/test 10% each, nested training prefixes | **n ≥ 300 is already covered by an official release.** H4 is therefore restricted to the n ≤ 300 region and must not claim novelty above it |
| **PolyBench** | Tier labels are in the task files, or at worst one click away | **Not in the task files, and not one click away.** They exist only in the Supporting Information, which ships as a single **156 MB PDF** (`ps6c00094_si_001.pdf`, 163,761,754 bytes, md5 `fc7861f0839a03f601aaba4a6c4ce4a0`). Table S27 there lists 39 rows of task → tier, together with dummy-baseline errors. The main text fixes the tier **sizes** (Tier 1 = 14, Tier 2 = 15, Tier 3 = 10), the four assignment criteria, and representative members of each tier — but it does not enumerate all 39 | **Tiers are therefore off the critical path.** §3.1 admits to the main table only tasks whose tier is *named in the main text*, so the design never waits on a 156 MB download. Transcribing S27 on D0 is a **verification step**: if it contradicts the main text, the main text wins, the affected task is replaced from the reserve list, and the date is recorded |

> **Why this section exists in the proposal rather than the appendix.** The value of a proposal is not the size of the numbers it quotes but whether those numbers can be traced to a file and a row. Every figure in §3 and §7 is reproducible from the released CSVs by the commands archived on D0. Where a published headline differs from the file contents, we report the file, the date, and the counting method.

> **What we could and could not verify on 2026-11-11, stated precisely.** *Verified* from the platform's own API: the 39 task names and their record counts, summing to **42,169** exactly (Appendix A). *Verified* from the main text (PMC13721469, CC BY-NC-ND): tier sizes 14/15/10; the four assignment criteria (sample coverage, expected sensitivity to unrecorded conditions, approximate measurement standardisation, source heterogeneity — with model performance deliberately excluded so that a tier is not defined and validated by the same scores); and the standardised protocol, namely canonical SMILES, **a fixed 70/30 train/test split with version-controlled indices**, Murcko-grouped cross-validation for hyperparameter search only, and metrics R²/MAE/RMSE. *Not verified*: the 39-row tier assignment itself. The SI is 156 MB behind a bot wall on the publisher site; the PMC mirror was unreachable from our network and the Europe PMC bulk endpoint stalled on every attempt. **We therefore do not claim to have read Table S27**, and §3.1 is built so that we do not need to before D0.

---

## 3. Task selection

**Rule: stratified, pre-registered, frozen on D0.** Random selection cannot support the tier-ordering hypothesis; convenience selection cannot survive the Q&A. Concrete task IDs, record counts and the reason for each pick are in **§3.1**.

| Source | Selection | Count |
|---|---|---|
| PolyBench (39 tasks) | **Two per interpretation tier**, and every one of the six is a property whose tier the main text *names*: Tier 1 → Tg, Tm; Tier 2 → tensile modulus, gas permeability; Tier 3 → dielectric loss factor, IWLT | 6 |
| PolyBench26 (8 properties) | Two datasets for the in-distribution main table — one experimental-source (PolyMetriX Tg), one MD-source (homopolymer density). Both are also members of an official scaling family, so the main table and H2 share data instead of duplicating it | 2 |
| OpenPoly | **Three properties with enough labels to support a size scan** — Tg (443), Dielectric Constant Total (295), Tm (168). Deliberately *not* the scarcest ones: at n = 30 a five-fold split leaves 6 test points and R² is undefined (§2.3). The three differ in label noise, which is what H4 needs | 3 (not in main table) |
| **Main table total** | | **8, across 2 official benchmarks** |

Property families targeted: thermal (Tg / Tm / IWLT), dielectric (constant vs. loss), mechanical (modulus), and transport (permeability) — i.e. one tier-crossing pair inside a single physical family (dielectric constant in Tier 1 vs. dielectric loss in Tier 3), which is the cleanest available test of H1.

> An 8-task, 2-per-tier main table is deliberately **stricter than Topic 1's ≥5 on one benchmark**. The reason is not vanity: with 2 tasks per tier an ordering claim is testable; with 2/1/1 the Tier-2 vs. Tier-3 contrast would rest on one task each and would be an anecdote. A single failed download still cannot put us below the line, because six reserve tasks are pre-registered in §3.1.

---

### 3.1 Candidate task IDs (frozen on D0)

PolyBench issues no numeric task IDs. The stable identifier is the **exact dataset name**, which is simultaneously the CSV filename on the platform and in the figshare deposit (CC BY 4.0, DOI 10.6084/m9.figshare.30917717). The `n` column below was read from the platform's own count endpoint on 2026-11-11; the 39 values sum to 42,169, matching the published figure exactly (Appendix A).

**Protocol, verified from the main text — and one correction it forces.**

- **PolyBench**: one **fixed 70/30 train/test split per task**, with version-controlled indices shared across all representations and models. Murcko-grouped cross-validation is used **for hyperparameter search only**. Metrics: R², MAE, RMSE.
- **PolyBench26**: official **5-fold** files (base seed 42, val/test 10% each) plus the official `scaling_splits`.
- Therefore Λ on a PolyBench task is computed on a **single** official split; its uncertainty comes from a percentile bootstrap over test residuals plus 3 model seeds — not from k-fold. **There is no such thing as "5-fold CV on PolyBench" under the official protocol**, and an earlier draft of this proposal said there was. §6 and §7 have been corrected accordingly.

**Table A — PolyBench main table (6 tasks, 2 per tier).** Every tier below is named in the main text, so none of these picks depends on Table S27.

| # | Task ID (= dataset name) | n | Family | Tier | Why this one |
|---|---|---|---|---|---|
| 1 | Glass transition temperature | 8,055 | thermal | 1 | Largest task in the release, and one of only three with official repeated-split diagnostics — so Λ_Tg can be checked against the benchmark's own split-to-split variance |
| 2 | Melting temperature | 3,796 | thermal | 1 | Second Tier-1 thermal. Tm is a sharper transition than Tg, so a Λ gap between rows 1 and 2 is itself evidence about protocol spread |
| 3 | Tensile modulus | 1,034 | mechanical | 2 | The Tier-2 case where chain entanglement should matter physically, not just statistically |
| 4 | Gas permeability coefficient | 711 | transport | 2 | Second Tier-2 axis. Transport is governed by free volume, which no repeat-unit descriptor can encode — the strongest prior for a large Λ |
| 5 | Dielectric loss factor | 279 | dielectric (loss) | 3 | The sharpest available contrast: dielectric **constant** is Tier 1, dielectric **loss** is Tier 3. The authors split one physical family across two tiers themselves, which is exactly the bet H1 makes |
| 6 | Isothermal weight loss time | 228 | degradation (time-domain) | 3 | Time- and temperature-dependent by construction; the Tier-3 task least likely to be repeat-unit-dominated |

**Table B — PolyBench26 main table (2 tasks).**

| # | Task ID | Source | Why this one |
|---|---|---|---|
| 7 | PolyMetriX Tg | experimental | Also an official scaling family (300 / 1000 / 3000), so it doubles as an H2 anchor; and it replicates row 1 — the *same physical quantity in a different release*, which is the cheapest available test of whether Λ is a property of the material or of the dataset |
| 8 | homopolymer density | MD | Official scaling family (300 / 1000). MD labels are essentially noiseless by construction, so the contrast against row 7 is exactly the axis H4 needs: same n, different label noise |

**Table C — mechanism datasets (not in the main table; they carry H2 and H3).**

| Role | Dataset | Official levels |
|---|---|---|
| Data-scale manipulation (κ) | alternating MD_5000 | 300 / 1000 / 3000 / 10000 |
| Data-scale manipulation (κ) | homopolymer MD_5000 | 300 / 1000 (1,715–1,800 rows; the 3000 level cannot be built) |
| Chain-architecture manipulation (H3) | homopolymer → alternating / random / block | official task 4 splits |

**Table D — OpenPoly boundary arm (3 properties, H4 only).** Tg (443), Dielectric Constant Total (295), Tm (168); scanned at n = 30 / 60 / 120 / 240 / all. Chosen because they are the only columns able to support a five-level scan (§2.3) and because they span three different label-noise regimes.

**Table E — reserve list (pre-registered, unlocked only by the D1 24:00 gate).** If a main task fails QA (missing PSMILES, unit ambiguity we cannot resolve, fewer than 200 usable labels after harmonisation), it is replaced without reopening the design — same tier first, closest family second. Where no same-family reserve exists we say so rather than pretending the swap is clean:

| Reserve | n | Tier | Replaces | Family match |
|---|---|---|---|---|
| Thermal decomposition temperature | 4,898 | 1 | row 1 or 2 | thermal ✓ |
| bandgap chain_Egc | 3,380 | 1 | row 1 or 2 | electronic (no Tier-1 thermal left) |
| Tensile stress strength at break | 1,034 | 2 | row 3 or 4 | mechanical ✓ |
| Dielectric loss tangent | 238 | 3 | row 5 | dielectric loss ✓ |
| Dynamic mechanical properties storage modulus | 387 | 3 | row 6 | ✗ — closest Tier-3 task with enough labels; a swap here weakens the degradation axis and must be stated |

**Eligibility rule (pre-registered, and the reason this section exists).** Only a task whose tier is *named in the main text* may enter the main table before S27 is transcribed. The remaining tasks are listed in Appendix A with their record counts and marked `tier: pending S27`; they become eligible only after transcription, and any substitution is logged with a timestamp and a one-line reason. **This rule is what stops the task set from drifting silently after D0** — which is the failure mode a fixed split is supposed to prevent.

---

## 4. Core quantities

### 4.1 Chain-context recovery rate Λ

$$\Lambda = \frac{R^2_{\text{chain}} - R^2_{\text{unit}}}{1 - R^2_{\text{unit}}} \in [0,1]$$

Of the variance a repeat-unit model cannot explain, what fraction does explicit chain information recover? Dimensionless, comparable across properties, invariant to the property's units — the property is reported in °C for one task and GPa for another, and Λ does not care.

**This is deliberately the same functional form as the crystal-side Γ in Topic 1.** The two chapters are two measurements of one quantity at two material hierarchies, not two projects stapled together.

> **On the word "ceiling".** Where the talk uses it, *ceiling* means the empirical upper bound of the repeat-unit model, $R^2_{\text{unit}}$ — the share of variance the repeat unit already explains. What this proposal measures is Λ — the fraction of the *remaining* variance that chain information can actually reclaim — and κ — whether that fraction moves when $n$ grows 10×. **So Λ ≈ 0 means the ceiling is closed, not absent.**

### 4.2 Data-compensation rate κ

$$\kappa = \left.\frac{d\Lambda}{d\log_{10} n}\right|_{\text{official scaling splits}}$$

Data grows 10×; how much of the gap closes? **κ ≈ 0 is our central claim**: the missing information is a missing coordinate, not noise, so interpolation accuracy improves while Λ does not.

### 4.3 Robustness rules (pre-registered)

- Λ denominator unstable as $R^2_{\text{unit}} \to 1$ → **always report ΔR² with a bootstrap CI alongside Λ**; conclusions require sign agreement.
- Only **regression** tasks enter Λ. Classification tasks are reported separately as ΔAUC and excluded from the cross-task ordering claim.
- **No cross-release pooling (R1, §2.1).** Λ is dimensionless, so it is *comparable*; it is not *poolable*. Every ordering claim is made inside one release, with the source-heterogeneity audit of §2.2 applied first.
- The unobservable part of "chain context" (morphology, processing history) cannot be injected. We state explicitly that **Λ covers only the measurable chain degrees of freedom** (repeat count, sequence, architecture) and treat the remainder as an unobservable upper bound rather than guessing at it.

---

## 5. Hypotheses and falsification criteria

> **Each hypothesis has exactly one home benchmark** (H1 → PolyBench, plus PolyBench26's architecture task for its sub-clause; H2 → PolyBench26; H3 → within a single task, both releases; H4 → OpenPoly). No hypothesis is tested by pooling across releases (R1, §2.1). This is what makes "several benchmarks" a coverage decision rather than a stability risk.

| ID | Hypothesis | Home benchmark | We are wrong if |
|---|---|---|---|
| **H1 Tier ordering** | Λ is ordered consistently with PolyBench's own three tiers; on PolyBench26's architecture task, **block > random > alternating > homopolymer** | PolyBench (+ PolyBench26 task 4) | Spearman ρ < 0.5, or any inversion spanning ≥2 tiers |
| **H2 Data cannot compensate** | On the official scaling splits, **κ is not significantly < 0 at 95% CI** — a 10× increase in n does not halve Λ, i.e. the gap behaves like a missing coordinate, not like sampling noise | PolyBench26 task 2 | κ significantly < 0, and Λ at the largest official n falls below half its value at the smallest n |
| **H3 Information beats architecture** | Injecting chain information explicitly (repeat count as a feature, wPSMILES weighting, multi-unit concatenation) reduces Λ **more than** swapping RandomForest for a GNN does | Same task, both arms | The architecture swap yields the larger reduction |
| **H4 What sets the crossover** | **Not "does an n\* exist"** — OpenPoly's own 26-property table already shows XGBoost leading almost everywhere at n ≤ 443, so asking whether a crossover exists would be re-asking a published question. The open question is what n\* is a *function of*: scan n = 30 / 60 / 120 / 240 / all inside each of three properties that differ in label noise. If the conventional arm's advantage vanishes at roughly the same n in all three, n\* is a sample-size effect. If it persists longer — or never vanishes — on the noisier experimental property, **n\* is set by label noise, not by n** | OpenPoly | The crossover n varies across the three properties *without* tracking their label dispersion, or no crossover is reachable within n ≤ 443 on any property — in which case we report the upper limit of the observation window and claim nothing beyond it |

**Where we are betting.** PolyBench classifies dielectric-type tasks as condition-sensitive (frequency, temperature, humidity all matter), yet repeat-unit models do reasonably well on them in the published results. **We predict their Λ exceeds that of tasks labelled intrinsic** (e.g. modulus-type). If correct, "the repeat unit is sufficient" is a property-dependent statement rather than a general one; if incorrect, the tiering itself needs revision. Either outcome is a result.

**H2 is the sharpest claim in the proposal.** If κ ≈ 0, the conclusion is that a decade of data accumulation in this field has bought in-distribution interpolation accuracy and not chain information. It is counter-intuitive, attackable, and therefore worth ten minutes.

---

## 6. Experimental design

| Arm | Content | Purpose |
|---|---|---|
| **A1 Floor** | Dummy (mean) + kNN | How hard is the task at all — the baseline everyone forgets |
| **A2 Conventional (mandatory)** | RDKit descriptors / Morgan fingerprints + **XGBoost**, **RandomForest** | Satisfies the "≥3 conventional baselines such as XGBoost" requirement; PolyBench26 ships `RDKit_RF` |
| **A3 Graph** | Official `polymer_chemprop` (weighted polymer graph GNN) / periodic graph network | The representation variable; **GPU-bound** |
| **A4 Language model** | `polyBERT` (official pre-computed embeddings) | The "stronger model" bonus line |
| **A5 Chain-information arm (our method)** | Minimal injection of chain degrees of freedom into A2/A3: repeat count as explicit feature, wPSMILES weighting, multi-unit sequence concatenation | Answers H3: information vs. architecture |

**Fairness constraints (enforced in code review, not aspiration)**

1. Official splits only; PolyBench26's scaling experiment **must** use the official `scaling_splits`. A unit test asserts split identity.
2. One hyperparameter search space and one budget (≤30 trials) for every arm, with the budget table published.
3. **The official evaluation unit of each benchmark, then 3 seeds.** PolyBench: its single official 70/30 split (version-controlled indices) × 3 model seeds, with a percentile bootstrap over test residuals. PolyBench26: the official 5-fold files × 3 seeds. **PolyBench does not use k-fold for evaluation** — its Murcko-grouped CV is for hyperparameter search only — so a k-fold number on a PolyBench task would not exist under the official protocol (§3.1). Differences below 2σ are not reported as findings.
4. Identical preprocessing across arms; log transforms permitted for condition-sensitive tasks only if applied to all arms simultaneously.

**Mechanism experiments**

| Experiment | Design | Fits |
|---|---|---|
| Dataset-size scaling | Official scaling splits, 5 levels × 5 folds × 2 properties × 2 arms | 100 |
| Architecture transfer | Homopolymer → alternating / random / block, 2 properties × 2 arms × 5 folds × 3 seeds | 60 |
| Main table | 8 tasks × 4 arms — PolyBench 6 × 3 seeds on the official split, PolyBench26 2 × 5 folds × 3 seeds | 96 classical + 48 GNN |

---

## 7. Work plan

### 7.1 Effort estimate (agent-assisted)

Units are **person-hours (pH)**. "With agent" assumes an agent that writes, runs, and self-checks code, with human review of every critical-path artefact (§7.5).

| Phase | Task | No agent | **With agent** | GPU | Critical | Owner |
|---|---|---|---|---|---|---|
| **P0** | PolyBench: 39 tasks download + integrity check | 6 | **3** | CPU | ✓ | A |
| P0 | **Label harmonisation** (K/°C, GPa/MPa, dielectric frequency & humidity; de-duplication over 42k records) | 12 | **7** | CPU | ✓ | A |
| P0 | Task list freeze + appendix snapshot | 4 | **2** | CPU | ✓ | A |
| P0 | PolyBench26 dual conda environments | 6 | **3** | CPU | ✓ | A |
| P0 | Data generation: PSMILES / wPSMILES / RDKit descriptors (1.5 h wall-clock) | 4 | **1.5** | CPU | ✓ | A |
| P0 | **polyBERT embeddings, 7.3 GB (1–3 h wall-clock; 6–10 h if CPU-only)** | 4 | **2** | **GPU recommended** | ✓ | B |
| P0 | Official split alignment + assertion tests | 8 | **4** | CPU | ✓ | A |
| **P1** | Representation pipeline (descriptors, Morgan, wPSMILES → graph, **official graph builder**) | 12 | **4.5** | CPU | ✓ | B |
| P1 | Conventional baselines ×4 (150 fits) | 14 | **5** | CPU | ✓ | C |
| P1 | **Official GNN: 8 tasks — PolyBench 6 × 3 seeds on the official split, PolyBench26 2 × 5 folds × 3 seeds = 48 runs (1–8 min each on one 4070; 3–5 GPU-h total)** | 22 | **11** | **GPU REQUIRED** | ✓ | B |
| P1 | polyBERT transfer arm | 8 | **3.5** | **GPU recommended** | | B |
| P1 | Main table + honest comparison with PolyBench leaderboard | 6 | **2.5** | CPU | ✓ | C |
| **P2** | Scaling experiment (100 fits, official splits) + **OpenPoly size scan: 3 properties × 5 n-levels × 5 folds × 3 seeds = 225 GNN runs, all n ≤ 443 (the classical arm on the same grid adds 225 CPU fits)** | 13 | **6** | **GPU** | | B (GNN) / C (CPU arms) |
| P2 | Architecture-transfer experiment (60 runs) | 12 | **5** | CPU | | C |
| P2 | Λ / κ computation, CI, denominator-stability handling | 8 | **3.5** | CPU | | D |
| **P3** | Figures (5) + three conclusions | 14 | **6.5** | — | | D |
| P3 | Slides (15–18) + timed rehearsal + Q&A | 16 | **7** | — | ✓ | owner |
| — | **Audit of agent-generated artefacts** (splits, leakage, metric implementation, run records) | — | **14** | CPU | ✓ | C+D |
| — | Unknown-item rework budget (data defects, OOM, task replacement) | 20 | **14** | — | | — |
| | **Total** | **189** | **105** | | | |

**Who does what, and why the balance looks like this.** Four members share the 96 pH of technical work equally — **24.0 pH each**, with fixed loads falling between 21.5 and 22.0 pH, a spread of 0.5 pH, and the remainder held as each member's own slack. Writing and the talk (7 pH plus 2 pH of slack, 9 pH in total) sit with the repository owner and are not assigned to a member. The split now follows the pipeline instead of the old discipline boundaries: **A produces the data, B runs everything that needs a GPU (A3, A4 and the GNN half of the OpenPoly scan), C runs everything that does not (A1, A2, A5 plus the three mechanism batches) and assembles the main table, and D passes judgement and draws the figures.** Two consequences are deliberate. First, with four people the runner and the judge can be different people, so C runs the batches whose outputs feed κ, the H4 crossing point and the architecture ordering, and hands every fit over together with its test predictions; D computes the criteria from those files and never has to ask how a batch was run. Second, the 14 pH audit is allocated by whose number depends on it rather than by production stage: 5 pH on the way in (D: splits and leakage, because Λ's denominator is exactly `n` and `split_ref`), 4 pH on the run records (C: 535 of the 808 fits are his), and 5 pH on the way out (D again — this one is a self-check on his own figures and is labelled as such in `docs/Audit Log.md`, not presented as independent review).

### 7.2 Schedule

**Variant 1 — 8 working days (presentation 11-19). Occupancy 95/128 = 74%.**

> The phase table below lists **gross** hours (105 pH). Net requirement is 95 pH because 10 pH of the tuning harness and plotting code can be carried over from a Topic-1 style codebase. **82% gross / 74% net** — we quote the net figure, and the gap is the reuse we have to actually earn, not assume.

| Phase | Dates | Items | pH | Milestone (do not proceed without) |
|---|---|---|---|---|
| P0 | 11/11–11/13 | 1–7 | 22.5 | **A baseline number on one task using the official split** |
| P1 | 11/13–11/16 | 8–12 | 26.5 | 8 tasks × 4 arms complete, each on its own official unit |
| P2 | 11/16–11/17 | 13–15 | 14.5 | Λ figure with CI exists; OpenPoly scan launched |
| P3 | 11/17–11/18 | 16–17 + audit | 27.5 | Two timed rehearsals passed |

> Phases overlap intentionally: different roles sit in different phases at the same time. The date ranges mark when a phase is *active*, not when a role is idle.

> **The GPU runs overnight, not in the pH budget.** The 48-run main sweep (3–5 GPU-h) fits inside **a single night, D0**; D1 night is held as pure slack and for retries. The 225-run OpenPoly scan (4–11 GPU-h) runs on the **P2 night**, after the main table is locked — it must never compete with the main sweep for the card. With the second card the scan can overlap P1 instead of waiting for P2.

> **Hard gates.** *D0 18:00* — embeddings and data generation started and verified (not "downloading"). *D0 24:00* — GNN single-task single-fold completes. *D1 12:00* — sweep launched and **measured per-run time compared against the §7.4 table**; if any run exceeds 1.5× the predicted minutes, apply triage step (i) immediately rather than waiting. *D1 24:00* — first trustworthy conventional baseline number. **If the D0 gates fail, switch to the Matbench-centred variant (*How Much Is Structure Worth?*).**

**Variant 2 — 15 working days (presentation 11-26). Occupancy 95/240 = 40%.**

| Phase | Dates | Content |
|---|---|---|
| P0 | 11/11–11/15 | Data, harmonisation, splits, environments |
| P1 | 11/16–11/21 | Main table + GNN sweep |
| P2 | 11/22–11/24 | Scaling + architecture transfer + Λ/κ |
| P3 | 11/25–11/26 | Figures, conclusions, slides, rehearsals |
| Optional | — | Extra bonus line + full 3-seed sweep on the two large tasks |

Variant 2 is no longer what makes H4 possible — H4 now sits inside the 8-day scope (§7.1, P2). The extra week buys **seed depth on the large tasks and one bonus experiment**, not the existence of a hypothesis.

### 7.3 Self-written minimum requirement (stricter than the given topics)

| # | Requirement | Threshold |
|---|---|---|
| 1 | Main-table tasks | **≥6**, across **≥2 official public benchmarks** |
| 2 | Splits | Official only; scaling must use official `scaling_splits` |
| 3 | Conventional baselines | **≥3** (Dummy, kNN, XGBoost, RF) |
| 4 | Strong baseline | **≥1** (official GNN or polyBERT) |
| 5 | Fairness | Same official split, same budget, same seed set; mean ± std + bootstrap CI |
| 6 | Attribution | **One measurable mechanism quantity** (Λ, κ) answering *why good / why poor* |
| 7 | Reproducibility | One-command rerun, budget table, task-list snapshot |
| 8 | Negative-result clause | If κ ≈ 0 fails or Λ ordering is violated, report the hypothesis as falsified; **do not re-tune until it fits** |

### 7.4 GPU requirement, sized for one RTX 4070 (12 GB)

We have two GPUs available; **every number below is computed for a single RTX 4070 (12 GB)**, so the second card is optional upside, not a dependency.

Per-run cost for the official `polymer_chemprop` weighted-graph GNN (depth 3, hidden 300, batch 64, ≤100 epochs, early stopping 15), measured proxy: 4k / 16k / 40k training graphs.

| Group | graphs/run | min/run | runs | GPU-h |
|---|---|---|---|---|
| PolyBench, 6 tasks × 3 seeds (228–8,055 rows) | 0.2k–8k | 1–8 | 18 | 0.6–0.9 |
| PolyBench26, 2 tasks × 5 folds × 3 seeds | 5k–20k | 5–8 | 30 | 2.5–4.0 |
| **GNN main sweep (8 tasks = 48 runs)** | | | **48** | **3–5** |

The old three-tier table (small / medium / large, 15 runs each) described a 6-task × 5-fold design that the protocol correction in §3.1 retired; the per-run costs above are the same measurements, regrouped the way the benchmark actually splits.

| Item | GPU | GPU-h on one 4070 |
|---|---|---|
| **GNN main sweep, 48 runs** | **Required** | **3–5** |
| polyBERT embedding generation | Recommended (CPU possible) | 0.5–1.5 (6–10 on CPU) |
| polyBERT transfer arm (fine-tune) | Recommended | 2.5–4.0 |
| **OpenPoly size scan, 225 runs** (3 properties × 5 n-levels × 5 folds × 3 seeds, all n ≤ 443) | **Required** | **4–11** |
| OOM / retry / early-stop overshoot buffer (+30%) | — | 3.0–6.5 |
| **Mandatory core** | | **≈13–28 GPU-h ≈ 0.6–1.2 days** continuous |
| *Optional:* architecture transfer as a GNN arm | Optional | +4–7 |
| **Planning envelope quoted in the header and the schedule** | | **≈19–36 GPU-h** |

> **Why the core dropped from 19–36 to 13–28 — and why the schedule still quotes 19–36.** Two corrections landed together. (i) **Protocol:** PolyBench evaluates on one fixed 70/30 split, not on k-fold (§3.1), so its six tasks cost 3 seeds each instead of 5 folds × 3 seeds; the main sweep fell from 90 runs to 48. (ii) **Task count:** the main table is 8 tasks (6 PolyBench + 2 PolyBench26), which is what §3 had always intended ("two per tier") but never wrote down. The OpenPoly arm is unchanged at 225 runs; it remains the single largest GPU item, and it is still a correction of a defect rather than a growth of scope — the earlier draft sized H4 on "the scarcest properties", which at n = 30 has 6 test points per fold and produces nothing.
>
> We keep **19–36** in the header and the schedule because that figure is this core at the top of its range *plus* the optional architecture arm — i.e. it is the pessimistic planning number, which is the only kind worth planning with. **Person-hours are unchanged at 105/95:** the 11 pH on the GNN row is dominated by building and debugging the official graph pipeline, not by the number of runs, so 48 instead of 90 runs does not move it. The freed wall-clock becomes slack on D1, which is exactly what an 8-day window needs.

**Consequence for the plan: the GPU is even further from the critical path than we claimed.** The mandatory core is one night for the main sweep plus one night for the OpenPoly scan; even the 19–36 h envelope is 1.5 continuous days, inside an 8-day window. The binding constraints remain **label harmonisation (7 pH) and audit (14 pH)** — human work that no card accelerates. This is the opposite of the usual assumption, and it is why we do *not* ask for more compute.

**Two 12 GB caveats, with the mitigation we will use.**

1. **Batch size.** Polymer repeat units are large graphs; hidden 600 with batch 64 can OOM at 12 GB. Mitigation: batch 32 + gradient accumulation 2 (wall-clock cost ≈1.2×, already inside the +30% buffer), hidden 300 as the default.
2. **Card sharing.** 12 GB cannot host two concurrent training processes. Mitigation: one process per card, `--num-workers` capped at 4, `torch.cuda.set_per_process_memory_fraction(0.9)` so a leak kills the run rather than the desktop session.

**What the second card is actually worth (and what it is not).** It does *not* halve the project. It buys three specific things, in order of value:

| Gain | Value |
|---|---|
| Decouple embedding generation from the sweep (currently serial on D0) | Removes ~1–1.5 h of dead time on the critical first night |
| Run seeds {0,1} / {2} as two parallel jobs | 13–28 h → **7–15 h**, so the OpenPoly scan overlaps P1 instead of waiting for P2 |
| Headroom for the GNN architecture-transfer arm | +4–7 GPU-h without touching the calendar |
| *Not gained* | Any reduction in the 21 pH of human harmonisation + audit |

**Triage rule if the sweep runs long (pre-registered, D1 12:00 checkpoint).** In this order: (i) cut max epochs 100 → 60 (early stopping already converges before 60 on 5 of 8 tasks); (ii) convert the polyBERT arm from fine-tune to frozen embeddings + ridge head (−2.5 GPU-h, stated on the slide); (iii) **only then** cut to 1 seed on the two largest tasks (Tg and PolyMetriX Tg) and report seed variance from the small ones. Reducing seeds is the last resort because H3 is a variance-sensitive claim.

**Why these three items and not the other sixty.** The GPU is not buying speed for the project; it is buying speed for **one learning paradigm**. Per training run of the official GNN (80-atom repeat unit, hidden 300, depth 3, 100 epochs, forward + backward ≈ 0.54 GFLOP per molecule):

| Task size | PFLOP per run | One 4070 (≈3.5 TFLOPS effective) | 8-core CPU (≈75 GFLOPS effective) |
|---|---|---|---|
| small (4k graphs) | 0.22 | ~1 min | ~48 min |
| medium (16k) | 0.87 | ~4 min | ~3.2 h |
| large (40k) | 2.18 | ~10 min | ~8 h |
| **48 runs total (new mix)** | **≈17 PFLOP** | **3–5 h** | **≈63 h ≈ 2.6 days** |

*(Unit note: earlier drafts labelled these per-run values TFLOP. They are PFLOP — 16k graphs × 100 epochs × 0.54 GFLOP is 0.87 PFLOP, not 0.87 TFLOP. The hours were always measured, not derived, so nothing downstream moves.)*

The CPU column is still the argument, in a weaker form than we first claimed: **without a GPU the main sweep alone is 2.6 days of pure compute, and the full 273-run programme is 4–6 days** — the entire 8-day window spent waiting, with nothing left for retries, harmonisation or analysis. Note that the CPU/GPU gap here is ~45×, not the 5–10× seen on large dense batches, because graph message passing is built from many small matmuls (64×300 by 300×300), which is exactly where CPU vectorisation loses.

polyBERT is the same story in a different shape: a 110M-parameter transformer at ~96 tokens costs ≈21 GFLOP per sequence forward, so 42k PSMILES ≈ **887 TFLOP** — about 4 min on one 4070 versus ~3.3 h on a CPU, where the 7.3 GB of weights make the path memory-bandwidth-bound rather than compute-bound.

> **The remaining bulk of the plan is CPU work.** The official scaling experiment (100 fits), architecture transfer (60 runs) and the four conventional arms (150 fits) are XGBoost / RF / kNN / ridge on ≤250k × ~600 tabular features — minutes each, and no GPU benefit for the same reason as in §7.4. **Four of twenty line items touch the card**: the main GNN sweep, the OpenPoly GNN scan, the polyBERT embedding pass and the polyBERT transfer arm.

**The necessity is scientific before it is computational.** H3 claims *information beats architecture*. That claim needs an architecture arm whose inductive bias comes from graph topology rather than from descriptors we hand-designed — a message-passing network. If the GPU disappears we can still run every descriptor-based arm and every Λ measurement, but H3 collapses from "information vs. architecture" to "information vs. model family within classical learners". **That is a change of claim, not a saving of time** — which is why the fallback is documented as a weakening rather than presented as an equivalent plan.

**CPU-only fallback (must be decided on D0).** If neither card is usable, drop the GNN arm and run A2/A4 on CPU: 8 tasks × 2 arms ≈ 100 fits, minutes each. Total wall-clock stays under a day. **Cost:** H3 ("information beats architecture") loses its architecture leg and must be rewritten as "information vs. model-family within classical learners". This is a real weakening, not a cosmetic one — it must be stated on the slide, not hidden.

### 7.5 What agent assistance does not compress

| Item | Why |
|---|---|
| **273 GNN runs** (48 main sweep + 225 OpenPoly scan) | 13–28 GPU-h on one 4070 including embeddings, transfer arm and retry buffer (19–36 envelope). Agents can orchestrate the sweep; they cannot make the card faster. **But note this is now a *smaller* constraint than the two rows below** |
| Label harmonisation (7 pH) | Agents draft the parser; deciding whether a Tg value is K or °C, and verifying 42k records against source, is human judgement |
| Λ/κ interpretation (3.5 pH) | This is the graded judgement |
| Audit (14 pH, **added** cost) | Agent-generated scripts most often (i) split data themselves instead of reading the official split, (ii) leak global statistics into features, (iii) reimplement metrics inconsistently. **None raise an error**; all invalidate the main table. Official-split compliance is a course red line |

---

## 8. Expected outputs

1. **Λ heatmap** — property × arm, with CIs. Answers Q1 (how large is the gap).
2. **Λ vs. tier ordering** — scatter with Spearman ρ, including the dielectric bet. Answers Q2 (what it depends on).
3. **Scaling curves, Λ vs. log n, with κ and its CI** — the single most memorable figure. Answers Q3.
4. **Information vs. architecture bar chart** — A5 gain against A3 gain.
5. **Crossover diagnostic** — R² gap (graph minus conventional) against log n, one panel per OpenPoly property, annotated with that property's label dispersion. Answers *what sets n\**, not merely *whether n\* exists*.
6. **Main table** — 8 tasks × 5 arms, mean ± std, with an honest statement of the gap to the PolyBench leaderboard.

---

## 9. Team and risks

| Role | Responsibility | pH | Share of the 96 pH |
|---|---|---|---|
| A — Data & compliance | Environments, benchmark integration, label harmonisation, task-list freeze, official splits, structure-level descriptors | 24.0 | 25.0% |
| B — Representations & GPU arms | Representation assembly and graph building, polyBERT embeddings, A3 sweep (48), A4 transfer arm, the GPU half of the OpenPoly scan (225) | 24.0 | 25.0% |
| C — Classical arms, mechanism runs & main table | A1/A2/A5, scaling (100) and the OpenPoly classical arm (225), architecture transfer (60), main table, run-record audit | 24.0 | 25.0% |
| D — Evaluation, figures & audit | Λ/κ, CIs, H1–H4, five figures, three conclusions, provenance, entry and exit audit | 24.0 | 25.0% |
| Writing & talk (repository owner) | Slides, Q&A bank, rehearsals — **not a member's assignment** | 9.0 | — (outside the 96) |

| Risk | P | Impact | Mitigation |
|---|---|---|---|
| PolyBench data inaccessible or incomplete | M | **H** | Fallback: PolyBench26 alone (8 properties × 4 official tasks still exceeds the ≥6 threshold) + OpenPoly. **But the count does not tell the whole story: the tier *labels* live only in PolyBench, so H1 must then be re-derived from property classes** (thermal / dielectric / mechanical / transport) — and we say so on the slide rather than reusing the benchmark's tiering |
| Cross-source heterogeneity makes the Λ ordering an artefact of protocol differences | M | **H** | §2.2 audit: per-source ordering with Kendall τ ≥ 0.6 before pooling, leave-one-source-out recheck, no unit conversion across measurement methods, four-tuple tagging on every value. If the ordering does not survive, it is reported per source and no combined claim is made |
| Label definitions cannot be harmonised across sources | M | **H** | Restrict to properties with unambiguous units; document every conversion; drop tasks that resist |
| No GPU available when needed | L (two cards; one suffices) | **H** if it happens | Single-4070 sizing (§7.4) needs 19–36 GPU-h, so the risk is card *availability at night*, not capability. **Reserve both cards from D0.** CPU fallback exists, with the weakened H3 stated openly |
| GNN sweep does not finish in the 8-day window | **L** (19–36 GPU-h vs. an 8-day window) | M | Pre-registered triage ladder in §7.4; the sweep is not on the critical path — human audit is |
| An OpenPoly property has fewer usable labels than the released count suggests once harmonised | L | M | Three properties chosen with headroom (443 / 295 / 168). If any drops below 120 after harmonisation, run H4 on Tg + dielectric only and report it on two properties rather than inflating n by keeping unconvertible units |
| Λ turns out statistically indistinguishable from zero everywhere | L | M | Then the repeat-unit ceiling is **closed**, not absent: the missing variance is unreachable by any chain descriptor we can build, the repeat-unit paradigm stands, and we have a clean negative result |
| Reviewers see this as two stapled-together projects | M | M | Single quantity, single model family, single statistical discipline; the claim is one measurement at one hierarchy, not two studies |

---

## 10. One-paragraph defence of the topic

> Topic 1 asks how much a crystal structure is worth once composition has done its part. We ask the same question one hierarchy up: **how much is a polymer chain worth once its repeat unit has done its part?** The quantity is the same, the protocol is the same, and the benchmark is official and open. What differs is that in polymers the missing information cannot be fully observed — so unlike crystals, we can measure not only how large the gap is, but whether more data closes it. Our prediction is that it does not.

---

## Appendix A — the 39 tasks as released (read 2026-11-11)

Names and record counts are taken verbatim from the platform's `/api/data-counts` endpoint; they sum to **42,169**, which reproduces the published total. The `Tier` column is `1/2/3` **only where the main text names that property or its family**; everything else is `pending S27` and is **not eligible for the main table** under the rule in §3.1. This is deliberately a conservative reading: it under-uses the release rather than guessing at it.

| # | Task ID (dataset name) | n | Tier | Role |
|---|---|---|---|---|
| 1 | Glass transition temperature | 8,055 | 1 | A#1 |
| 2 | Thermal decomposition temperature | 4,898 | 1 | reserve |
| 3 | Melting temperature | 3,796 | 1 | A#2 |
| 4 | bandgap chain_Egc | 3,380 | 1 | reserve |
| 5 | Intrinsic viscosity | 1,936 | pending S27 |  |
| 6 | Density | 1,699 | pending S27 | — (PB26 row 8 is a different dataset) |
| 7 | Specific volume | 1,699 | pending S27 |  |
| 8 | PCE_OPV_aug | 1,203 | 2 |  |
| 9 | Elongation at break | 1,113 | 2 |  |
| 10 | Tensile stress strength at break | 1,034 | 2 | reserve |
| 11 | Tensile modulus | 1,034 | 2 | A#3 |
| 12 | Electric conductivity | 982 | 3 |  |
| 13 | Lc phase transition temperature | 941 | pending S27 |  |
| 14 | Volume resistivity | 934 | pending S27 |  |
| 15 | Softening temperature | 763 | pending S27 |  |
| 16 | Gas permeability coefficient | 711 | 2 | A#4 |
| 17 | Water absorption | 680 | pending S27 |  |
| 18 | Heat of fusion | 588 | pending S27 |  |
| 19 | bandgap bulk_Egb | 561 | 1 |  |
| 20 | Crystallization temperature | 456 | pending S27 |  |
| 21 | Gas diffusion coefficient | 433 | 2 |  |
| 22 | Dynamic mechanical properties storage modulus | 387 | 3 | reserve |
| 23 | Dielectric Constant_EPS | 382 | 1 |  |
| 24 | Refractive Index_Nc | 382 | pending S27 |  |
| 25 | Ionization Energy_Ei | 370 | 1 |  |
| 26 | Electron Affinity_Eea | 368 | 1 |  |
| 27 | Surface tension | 346 | pending S27 |  |
| 28 | Solubility parameter | 316 | pending S27 |  |
| 29 | Cohesive energy density | 307 | pending S27 |  |
| 30 | Dynamic mechanical properties loss tangent | 300 | 3 |  |
| 31 | Dielectric loss factor | 279 | 3 | A#5 |
| 32 | Contact angle | 255 | pending S27 |  |
| 33 | Tensile stress strength at yield | 253 | 2 |  |
| 34 | Gas solubility coefficient | 249 | 2 |  |
| 35 | Dielectric loss tangent | 238 | 3 | reserve |
| 36 | Isothermal weight loss time | 228 | 3 | A#6 |
| 37 | Heat of fusion mol conversion | 213 | pending S27 |  |
| 38 | Specific heat capacity | 207 | pending S27 |  |
| 39 | Dynamic mechanical properties loss modulus | 193 | 3 |  |

Totals: **39 tasks, 42,169 records.** Named tiers: 8 in Tier 1, 8 in Tier 2, 7 in Tier 3; **16 tasks remain `pending S27`** — enough headroom that no design decision in this proposal depends on the missing table.

> **Reading note for the TA.** Tier *sizes* (14/15/10) are verified from the main text; the per-task assignment is not. Two ways to close the gap on D0, in order of cost: (i) download the 156 MB SI once and transcribe Table S27 (39 rows, ~30 min of human work); (ii) if the SI is unobtainable, recompute the tiers from the four published criteria and submit that as a derived artefact with its provenance line — never silently adopt a different tiering. Either way the main table in §3.1 does not change, because every task in it is already named in the main text.
