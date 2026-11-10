# Configs are protocol, not convenience

Every JSON file here is a pre-registered decision. Changing one after the D0 freeze
is a protocol change and must be recorded in `docs/DECISIONS.md` with a date and a
reason.

Why JSON and not YAML: these files are read by `tests/` and by CI on a machine with
**no third-party packages installed**. `json` is stdlib; `pyyaml` is not. JSON also
has no implicit typing, so `1e3`, `on`, `no` cannot silently become a float or a
boolean in a file whose whole purpose is to be unambiguous. See ADR-001.

| Directory | File count | What it fixes |
|---|---|---|
| `tasks/` | 11 | the 8 main-table tasks plus the 3 OpenPoly scan properties |
| `arms/` | 5 | A1鈥揂5: environment, features, models, GPU requirement |
| `experiments/` | 4 | main table, scaling, architecture transfer, OpenPoly scan |
