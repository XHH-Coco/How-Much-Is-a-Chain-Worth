# Results

`raw_runs/` is **committed on purpose**. One JSON per run, typically 1鈥? KB, and it
is the only physical evidence that a number in a figure came from a run that
happened. A few hundred runs is a couple of megabytes. Model checkpoints are not
committed: they are hundreds of megabytes and fully reconstructible from
`config + seed`.

Layout of a run record:

```json
{
  "run_id": "pb_t1_tg__a2_conventional__seed0",
  "task_id": "pb_t1_tg",
  "arm_id": "a2_conventional",
  "seed": 0,
  "split": {"path": "splits/polybench/pb_t1_tg/split_v1.json", "sha256": "..."},
  "config_sha256": "...",
  "metrics": {"r2": 0.0, "mae": 0.0, "rmse": 0.0},
  "bootstrap": {"r2_lo": 0.0, "r2_hi": 0.0, "n_resamples": 10000, "level": 0.95},
  "lambda": null,
  "env_fingerprint": "results/env_fingerprints/pb_t1_tg__a2_conventional__seed0.json",
  "started_at": "...", "wall_seconds": 0.0, "status": "ok"
}
```

`null` is meaningful: it says "not computed", not "zero". `lambda` is filled by
`06_aggregate.py`, never by the runner, so that a run cannot quietly assume its own
denominator.
