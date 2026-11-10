# How Much Is a Chain Worth?

**Chain-information recovery in polymer property prediction, and whether more data buys it**

> If you compress a polymer chain into a single repeat unit, what fraction of its
> property variance becomes unexplainable — and does that fraction shrink when you
> add 10× more data?

`status: skeleton, pre-D0` · code MIT · no benchmark data in this repo

---

## Why this exists

A crystal is described by composition plus a unit cell. A polymer is not: molecular
weight distribution, sequence, tacticity, morphology and processing history all move
measured properties, and none of them fit in a PSMILES string. PolyBench's authors
say so in their own limitation paragraph, and they respond by grouping their 39 tasks
into three provisional interpretation tiers.

**The limitation is acknowledged and the tiering is given, but the size of the gap has
never been measured.** That is the opening. We measure two quantities:

| Symbol | What it is | Range |
|---|---|---|
| **Λ** | of the variance the repeat unit cannot explain, the fraction chain information recovers | [0, 1] |
| **κ** | `dΛ / d log₁₀ n` — does that fraction move when data grows 10× | real |

`ceiling`, where the talk uses it, means `R²_unit`: the share the repeat unit already
explains. Λ measures the space above it. So **Λ ≈ 0 means the ceiling is closed, not
absent** (see `docs/PROTOCOL.md`).

## The eight main-table tasks

Two per PolyBench tier, plus two PolyBench26 datasets. Every tier here is named in the
benchmark's own main text, so nothing waits on the 156 MB Supporting Information.

| # | Task | Benchmark | n | Tier | Family |
|---|---|---|---|---|---|
| 1 | Glass transition temperature | polybench | 8,055 | 1 | thermal |
| 2 | Melting temperature | polybench | 3,796 | 1 | thermal |
| 3 | Tensile modulus | polybench | 1,034 | 2 | mechanical |
| 4 | Gas permeability coefficient | polybench | 711 | 2 | transport |
| 5 | Dielectric loss factor | polybench | 279 | 3 | dielectric loss |
| 6 | Isothermal weight loss time | polybench | 228 | 3 | degradation |
| 7 | PolyMetriX Tg | polybench26 | — | — | thermal |
| 8 | homopolymer density | polybench26 | — | — | density |

Plus an **OpenPoly scan** at n = 30 / 60 / 120 / 240 / all over Tg (443),
Dielectric Constant Total (295) and Tm (168) — H4 only, never in the main table.
Selection rules, reserve list and the "why this one" for each row: `docs/PROTOCOL.md`
and `data/manifests/candidate_tasks.csv`.

## The five arms

| Arm | Content | Env | GPU |
|---|---|---|---|
| A1 floor | dummy mean + kNN | Core | no |
| A2 conventional | RDKit descriptors / Morgan FP + XGBoost, RandomForest | Core | no |
| A3 graph | official `polymer_chemprop` | GNN | **yes** |
| A4 language model | polyBERT official embeddings | Polybert | yes |
| A5 chain information | repeat count feature, wPSMILES weighting, multi-unit concatenation | Core | no |

A5 is a modification of A2, not a sixth family. It exists to answer H3: does
*information* beat *architecture*?

## Reproduce

Nothing here needs the benchmark data to be present, and nothing needs a package
beyond the standard library:

```bash
python scripts/check_repo.py                  # layout, configs, manifest arithmetic
python -m unittest discover -s tests -t .     # 51 tests, stdlib only
python environment/verify_env.py              # compares conda envs to pins.json
```

Once the data is in place (see `scripts/00_fetch_data.sh`) and the environments are
installed:

```bash
python scripts/01_build_manifests.py --hash   # digests every download
python scripts/02_build_splits.py             # copies official indices, never invents them
bash   scripts/03_run_main_table.sh           # one JSON per run into results/raw_runs/
python scripts/06_aggregate.py                # Λ and κ; runs never compute their own
```

**Current state:** the protocol layer, the tests and the configs are complete and
green. The arms (`src/b2/arms/`) are not implemented yet — that is P1, and it needs
RDKit, which is deliberately not installed.

## Results

Not yet. This section is filled at P2 and never before. When it is filled it will
carry Λ and κ **with bootstrap intervals**, the number of tasks that agree with H1's
direction, and the honest gap to the PolyBench leaderboard. A cell without an interval
does not go in.

## What would prove us wrong

| If | Then |
|---|---|
| κ is significantly below 0 at 95% CI, and Λ halves from the smallest to the largest official n | H2 is falsified: data *does* buy chain information |
| Λ ordering gives Spearman ρ < 0.5, or inverts across two tiers | H1 is falsified: the tiering does not track recoverable information |
| the architecture swap beats explicit chain information | H3 is falsified: representation, not information, is the lever |
| n\* moves without tracking label dispersion | H4 is falsified: n\* is not noise-driven |

We report falsification rather than re-tuning until it fits. **Λ ≈ 0 everywhere is a
result**: the ceiling is closed, the repeat-unit paradigm stands, and the field's
data-accumulation strategy has bought interpolation, not information.

## Data and licences

No benchmark data is committed. `data/raw/` is gitignored; `data/manifests/` holds
the counts and digests that prove which file we used. Four similarly named resources
exist — **OpenPoly** (741 structures / 3,985 labels / 26 properties), **OPoly26**
(Meta, 6.57M DFT), **PolyBench26** (LLNL, ~250k datapoints) and an unrelated
**PolyBench** of ~125k LLM QA pairs. Cite by DOI and scale, never by name alone.
Details and the licence status of each: `DATA_LICENSES.md` and `data/README.md`.

## Three environments

Python 3.10.14 in all three. 3.10 is past upstream EOL (2026-10-31) and we stay on it
anyway, because that is the generation the published baselines were evaluated on and
every pin in `environment/pins.json` is the **last** one that still ships a cp310
wheel. Mixed RDKit + PyTorch in one environment is the classic Windows `libgomp`
crash, hence the split. torch always comes from `download.pytorch.org`, never conda.

`pins.json` is the authority; `verify_env.py` fails on versions that are **too high**
as well as too low, because `pip install -U` under 3.10 fails silently-late.

## Repository map

| Path | What is decided there |
|---|---|
| `docs/PROTOCOL.md` | the pre-registration: tasks, arms, hypotheses, falsification criteria |
| `docs/DECISIONS.md` | ADRs — why JSON, why 3.10, why no k-fold on PolyBench |
| `docs/APPENDIX_A.md` | all 39 PolyBench tasks with counts and tier status |
| `docs/RUNBOOK.md` | D0 checklist, gates, four roles, what to do when it breaks |
| `docs/PROVENANCE.md` | the path from each claim to a file and a row |
| `splits/` | **the most important directory** — frozen indices, digest-checked |
| `configs/` | protocol as machine-readable JSON, not prose |
| `results/raw_runs/` | one JSON per run: the audit trail |
| `src/b2/analysis/lambda_kappa.py` | the only place Λ and κ are computed |

## Team

| Role | Responsibility | Share |
|---|---|---|
| A — Data & compliance | benchmark integration, harmonisation, split correctness, audit | 28% |
| B — Modelling | the five arms, GPU sweep, mechanism experiments | 32% |
| C — Analysis | Λ / κ, intervals, figures, the three conclusions | 28% |
| D — Writing & talk | slides, Q&A bank, rehearsals | 12% |

## Cite

`CITATION.cff`. Cite the benchmarks as well: PolyBench (figshare
`10.6084/m9.figshare.30917717`), PolyBench26 and OpenPoly each by their own DOI.
