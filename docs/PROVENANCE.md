# Where every number comes from

A number without a file and a row is a guess. This table is the shortest path from a
claim in the talk to the thing that supports it.

| Claim | Source | How to re-check |
|---|---|---|
| 39 tasks, 42,169 records | platform `/api/data-counts`, read 2026-11-11 | `data/manifests/polybench_tasks.csv`; `python -m unittest tests.test_manifests` |
| Tier sizes 14/15/10 | PolyBench main text | not machine-checkable; quoted, not recomputed |
| Per-task tier | main text, where named; otherwise SI Table S27 | `docs/APPENDIX_A.md`; the `tier` column |
| OpenPoly 741 structures / 3,985 labels | released CSV, counted 2026-11-11 | `data/manifests/openpoly_scan.csv` + the count script on D0 |
| OpenPoly Tg 443 / dielectric 295 / Tm 168 | same CSV, per-column non-null count | same |
| 24 of 28 OpenPoly columns below n = 200 | same CSV | same; this is the fact that killed "use the scarcest property" |
| PolyBench26 levels 300/1000/3000/10000 | official `scaling_splits` | `configs/experiments/scaling.json`; homopolymer has only 300/1000 |
| Fixed 70/30 with version-controlled indices | PolyBench main text | `splits/README.md`; enforced by `config.validate_task` |
| PolyBench does not use k-fold for evaluation | same | ADR-003; a test asserts the split kind |
| 13鈥?8 GPU-h core, 19鈥?6 envelope | proposal 搂7.4, arithmetic from run counts | recompute from `configs/experiments/*.json` |

## The three numbers that were wrong once

Recorded here because a proposal that only lists its successes is not auditable.

| Number | Was | Actually |
|---|---|---|
| OpenPoly scale | "3,985 polymer entries" | **741 structures**, 3,985 *annotations* |
| OpenPoly scarcest property | "use the properties with dozens of labels" | at n = 30 a 5-fold split leaves 6 test points; R虏 is undefined |
| PolyBench protocol | "6 tasks 脳 5 folds 脳 3 seeds" | one fixed 70/30 split; k-fold does not exist for evaluation |

All three corrections are in ADR-003 and 搂2.3 of the proposal.
