# Splits 鈥?the most important directory in this repository

Code can be rewritten; the protocol cannot drift. Three different split
constructions coexist here and must never be confused:

| Directory | Benchmark | Construction | Who owns it |
|---|---|---|---|
| `polybench/` | PolyBench | **one fixed 70/30 split per task**, version-controlled indices | the benchmark |
| `polybench26/` | PolyBench26 | **official 5 folds**, base seed 42, val/test 10% each, plus `scaling_splits` | the benchmark |
| `openpoly/` | OpenPoly | internal stratified scan at n = 30/60/120/240/all | **us** 鈥?therefore needs extra scrutiny |

Two consequences that have already bitten this project once:

- **There is no such thing as "5-fold CV on PolyBench".** Murcko-grouped CV exists
  only for hyperparameter search. Any table cell claiming 5-fold on a PolyBench
  task is wrong. 螞 on PolyBench comes from one split, with uncertainty from a
  percentile bootstrap over test residuals plus 3 model seeds.
- **A split we built ourselves is evidence of nothing on its own.** Every file under
  `openpoly/` must record `generator_sha256` 鈥?the hash of the script that produced
  it 鈥?so a reviewer can rerun the generator and get the same indices.

## Rules

1. Files here are **read-only**. A test asserts that the on-disk bytes match the
   digest recorded in the manifest. Do not "fix" a split to make a number nicer.
2. Every path is `splits/<benchmark>/<slug>/<name>.json`, using a slug rather than
   the raw dataset name, because PolyBench task names contain spaces and slashes.
3. Each file carries the four-tuple used by every reported 螞:
   `(benchmark, task_id, split_version, unit)`.
