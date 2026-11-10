# Runbook — eight working days

Four people, 105 person-hours with agent assistance, 95 remaining at submission.
Roles follow the proposal: **A** data & compliance (28%), **B** modelling (32%),
**C** analysis (28%), **D** writing & talk (12%).

## Gate discipline

A phase does not start until the previous milestone is met. The milestones are
deliberately measurable, because "mostly working" is how a schedule dies.

| Phase | Dates | Milestone (do not proceed without) | Owner |
|---|---|---|---|
| **P0** | D0–D2 | **one baseline number on one task, using the official split** | A |
| **P1** | D2–D4 | five arms run end to end on one task; split-identity test green | B |
| **P2** | D4–D6 | Λ with CI exists for ≥ 4 tasks; scaling sweep launched | B + C |
| **P3** | D6–D7 | five figures; three conclusions; Q&A bank rehearsed twice | C + D |

## D0 checklist (half a day, four people)

| # | Item | Owner | Done when |
|---|---|---|---|
| 1 | `scripts/00_fetch_data.sh`, record digests with `01_build_manifests.py --hash` | A | `data/manifests/sha256.txt` has one line per file |
| 2 | Copy official split indices into `splits/`; `02_build_splits.py` exits 0 | A | every path in `configs/tasks/*.json` resolves |
| 3 | Transcribe Table S27 (39 rows, ~30 min) **or** recompute tiers from the four published criteria and record the method | A | `MANIFEST.lock.json` has `frozen: true` |
| 4 | Install the three environments from `environment/*.yml`; `verify_env.py --strict` exits 0 | B | no version mismatch in either direction |
| 5 | `python -m unittest discover -s tests -t .` green; `scripts/check_repo.py` green | C | CI green on `main` |
| 6 | Reserve both GPUs for D2–D6 nights | D | calendar entry |
| 7 | Tag `d0-freeze` and push | D | tag visible on the remote |

## Reserve both GPUs

The 19–36 GPU-h envelope is small, but it assumes the card is free when the sweep
wants it. The risk is availability at night, not capability. Book both cards at D0.

## If something breaks

| Symptom | First move |
|---|---|
| `verify_env.py` reports a version that is too high | someone ran `pip install -U`; reinstall from `requirements-*.txt`, do not accept the new number |
| a split file changed | `test_splits_identity` should have caught it; restore from git, never re-derive |
| Λ is huge (> 1) on one task | check the denominator: `R2_unit` close to 1 makes Λ explode. Report ΔR2 instead |
| a GNN run OOMs on a 4070 | batch size down before sequence length; 12 GB is enough for every task in the main table |
| an OpenPoly property drops below 120 labels after harmonisation | run H4 on Tg + dielectric only, and say so on the slide |

## What "done" means

Not "the pipeline runs". Done means: a reader can clone this repo, run three
commands, reproduce one number in the main table, and trace every other number to a
file in `results/raw_runs/`.
