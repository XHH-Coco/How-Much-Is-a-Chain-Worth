# Pre-registered protocol

**Status: drafted, not yet frozen.** It becomes binding at the `d0-freeze` tag.
After that tag, any change to a task, an arm, a split or a falsification criterion
must be recorded in `DECISIONS.md` with a date, a reason and the person who signed
it off. Editing this file without an ADR is how a project quietly starts fitting its
own conclusion.

## The question

> If you compress a polymer chain into a single repeat unit, what fraction of its
> property variance becomes unexplainable — and does that fraction shrink when you
> add 10x more data?

## What is measured

| Symbol | Definition | Range | What it is not |
|---|---|---|---|
| **Λ** chain-context recovery rate | `(R2_chain - R2_unit) / (1 - R2_unit)` | [0, 1] | Not a performance ceiling. It is the fraction of the *unexplained* variance that chain information recovers |
| **κ** data-compensation rate | `dΛ / d log10(n)` on the official scaling splits | real | Not a learning-curve exponent; it is the slope of the *gap*, not of the score |

Every reported Λ carries the four-tuple `(benchmark, task_id, split_version, unit)`.
A number missing any element does not enter a figure (R3).

**On the word "ceiling".** Where the talk uses it, *ceiling* means `R2_unit`, the
share the repeat unit already explains. Λ measures the space above it. So Λ ≈ 0
means the ceiling is **closed**, not absent.

## The eight main-table tasks

Two per PolyBench interpretation tier, plus two PolyBench26 datasets. Tier labels
exist only in PolyBench; every task below has its tier named in the benchmark's main
text, so nothing here waits on the 156 MB Supporting Information.

| # | Task ID | Benchmark | n | Tier | Family | Why this one |
|---|---|---|---|---|---|---|
| 1 | Glass transition temperature | polybench | 8,055 | 1 | thermal | largest task; one of three with official repeated-split diagnostics |
| 2 | Melting temperature | polybench | 3,796 | 1 | thermal | sharper transition than Tg; a Λ gap vs row 1 is evidence about protocol spread |
| 3 | Tensile modulus | polybench | 1,034 | 2 | mechanical | the Tier-2 case where entanglement should matter physically |
| 4 | Gas permeability coefficient | polybench | 711 | 2 | transport | free-volume governed; strongest prior for a large Λ |
| 5 | Dielectric loss factor | polybench | 279 | 3 | dielectric loss | dielectric **constant** is Tier 1, dielectric **loss** is Tier 3 — one family split across tiers |
| 6 | Isothermal weight loss time | polybench | 228 | 3 | degradation | time- and temperature-dependent by construction |
| 7 | PolyMetriX Tg | polybench26 | — | — | thermal | replicates row 1 across releases; also the H2 anchor |
| 8 | homopolymer density | polybench26 | — | — | density | MD labels essentially noiseless; the contrast against row 7 is H4's axis |

The OpenPoly scan (Tg 443 / Dielectric Constant Total 295 / Tm 168) is **not** in the
main table. It exists only to answer H4, at n = 30 / 60 / 120 / 240 / all.

## The five arms

| Arm | Content | Env | GPU | Purpose |
|---|---|---|---|---|
| A1 floor | dummy mean + kNN | Core | no | how hard is the task at all |
| A2 conventional | RDKit descriptors / Morgan FP + XGBoost, RandomForest | Core | no | the required conventional baselines |
| A3 graph | official `polymer_chemprop` | GNN | **yes** | the representation variable |
| A4 language model | polyBERT official embeddings | Polybert | yes | bonus line |
| A5 chain information | repeat count as feature, wPSMILES weighting, multi-unit concatenation | Core | no | answers H3 |

A5 is a modification of A2 (and optionally A3), not a sixth model family. If the GPU
disappears, A2/A5 still run and H3 degrades from "information vs architecture" to
"information vs model family within classical learners" — a change of claim, and we
say so rather than presenting it as an equivalent plan.

## Hypotheses and what would falsify them

| ID | Claim | Home benchmark | Falsified if |
|---|---|---|---|
| H1 | Λ is ordered consistently with PolyBench's own tiers; on the architecture task block > random > alternating > homopolymer | PolyBench (+ PB26 task 4) | Spearman ρ < 0.5, or any inversion spanning ≥ 2 tiers |
| H2 | κ is **not** significantly below zero at 95% CI; a 10x increase in n does not halve Λ | PolyBench26 task 2 | κ significantly < 0 **and** Λ at the largest official n below half its value at the smallest |
| H3 | explicit chain information reduces Λ more than swapping RandomForest for a GNN does | one task, both arms | the architecture swap yields the larger reduction |
| H4 | the crossover n\* is a function of label noise, not of n | OpenPoly | n\* varies without tracking label dispersion, or no crossover is reachable within n ≤ 443 |

H4 is deliberately **not** "does an n\* exist". PolyBench26 already covers n ≥ 300
with official scaling splits, so claiming novelty there would be wrong.

## Three rules that keep four benchmarks from becoming four experiments

- **R1 — one benchmark per claim, no cross-release pooling.** Λ is dimensionless and
  therefore comparable; it is not poolable. Every ordering claim lives inside one
  release.
- **R2 — ordering needs at least two tasks per release.** One task per source is an
  anecdote; it is reported as an observation and excluded from ρ.
- **R3 — four-tuple on every Λ.**

## Measurement protocol (as implemented, not as remembered)

| Benchmark | Split | Uncertainty | Metric |
|---|---|---|---|
| PolyBench | **one fixed 70/30 split**, version-controlled indices | percentile bootstrap over test residuals + 3 model seeds | R2, MAE, RMSE |
| PolyBench26 | official 5 folds (base seed 42, val/test 10%) and official `scaling_splits` | across folds × 3 seeds | R2, MAE, RMSE |
| OpenPoly | nested scan inside the training partition | bootstrap + 3 seeds | R2, MAE, RMSE |

**There is no such thing as "5-fold CV on PolyBench."** Murcko-grouped CV exists for
hyperparameter search only. An earlier draft of the proposal claimed otherwise; the
correction is recorded in DECISIONS.md (ADR-003).

## Negative-result clause

If κ ≈ 0 fails, or the Λ ordering is violated, the hypothesis is reported as
falsified. We do not re-tune until it fits. A clean negative result is still a
result, and the risk table in the proposal already says what it would mean.
