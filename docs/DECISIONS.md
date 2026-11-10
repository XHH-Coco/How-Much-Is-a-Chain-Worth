# Decision log (ADR)

One entry per decision that a reader might otherwise think was arbitrary. Each is
dated. Entries are never deleted; a reversed decision gets a new entry that
supersedes the old one.

---

## ADR-001 — Configs are JSON, not YAML

**2026-11-11.** Status: accepted.

The configs under `configs/` are protocol, and they must be readable by the CI and by
`tests/` on a machine where **none of the three conda environments is installed yet**.
`json` is stdlib; `pyyaml` is not. YAML also has implicit typing, so `1e3`, `on` and
`no` can silently become a float or a boolean in a file whose only job is to be
unambiguous.

Consequence: no hydra/omegaconf. Config composition is a plain dict merge in
`src/b2/config.py`.

---

## ADR-002 — Task directories use slugs, not dataset names

**2026-11-11.** Status: accepted.

PolyBench issues no numeric task IDs; the stable identifier is the exact dataset
name, and those names contain spaces (`Gas permeability coefficient`) and slashes.
Using them as directory names produces paths that break in shell loops and in URLs.

Every config therefore carries both: `task_id` = slug (`pb_t2_permeability`) for the
filesystem, `dataset_name` = the exact string for matching against the release. The
mapping is also in `data/manifests/polybench_tasks.csv`.

---

## ADR-003 — PolyBench is a single 70/30 split, never k-fold

**2026-11-11.** Status: accepted. Supersedes an earlier draft.

The proposal's first version claimed "6 tasks × 5 folds × 3 seeds". The official
protocol is one fixed 70/30 split with version-controlled indices; Murcko-grouped CV
is used for hyperparameter search only. So there is no k-fold number to report on a
PolyBench task, and claiming one would have been a fabricated cell in the main table.

Consequences, all applied consistently: main table 6 → 8 tasks; GNN main sweep
90 → 48 runs; classical fits 120 → 96; total GNN runs 315 → 273. Person-hours did
not change, because those hours are dominated by pipeline construction, not by run
count.

---

## ADR-004 — Python 3.10, and every pin is a ceiling, not a floor

**2026-11-11.** Status: accepted.

Python 3.10 reached upstream EOL on 2026-10-31. We stay on it anyway, for one reason:
it is the generation the PolyBench / PolyBench26 baselines were evaluated on, and
comparability is the product. Every pin in `environment/pins.json` is the **last**
generation that still ships a cp310 wheel — `rdkit` 2026.3 has already dropped 3.10,
PyTorch 2.15 has already dropped 3.10.

This inverts the usual failure mode: `pip install -U rdkit` does not fail loudly, it
fails minutes later with no matching distribution, or worse, succeeds and changes a
number. `verify_env.py` therefore fails on versions that are **too high** as well as
too low.

---

## ADR-005 — Run records are committed; checkpoints are not

**2026-11-11.** Status: accepted.

`results/raw_runs/*.json` is one small record per run (1–5 KB): config hash, split
hash, seed, metrics, bootstrap interval, environment fingerprint. A full sweep is a
couple of megabytes and it is the only physical evidence linking a figure to a run.
Model checkpoints are hundreds of megabytes and fully reconstructible from
`config + seed`.

Corollary recorded in `.gitignore`: never add a blanket `*.json` or `*.csv` rule.
It would silently drop exactly the files this repository exists to preserve.

---

## ADR-006 — Three environments, and torch always comes from pip

**2026-11-11.** Status: accepted.

RDKit and PyTorch each bundle their own OpenMP runtime; on Windows, loading both in
one process is the most common cause of silent `libgomp` crashes in cheminformatics.
polyBERT additionally pins `tokenizers` and needs an isolated 7.3 GB cache.

torch is installed from `download.pytorch.org`, not from conda, even inside a conda
environment. Mixing conda's torch with pip's RDKit reintroduces the exact conflict
the split was meant to remove.

---

## ADR-007 — `raw_runs` records are write-once

**2026-11-11.** Status: accepted.

A run record is never edited. A re-run gets a new `run_id`. Editing a record to fix a
metric destroys the only audit trail we have, and it is indistinguishable from
fabrication after the fact. `06_aggregate.py` reads runs and writes separate
aggregate tables; it never touches a run record.
