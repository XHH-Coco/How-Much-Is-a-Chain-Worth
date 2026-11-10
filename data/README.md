# Data card

**No benchmark data is stored in this repository.** `raw/ interim/ processed/
external/` are gitignored. What is stored is the *fingerprint* of each download, so
that a reviewer can verify we used the file we claim to have used.

| Directory | Contents | In git |
|---|---|---|
| `raw/` | untouched downloads, one subdirectory per release | no |
| `interim/` | harmonised labels, before feature building | no |
| `processed/` | feature matrices, graph objects | no |
| `external/` | polyBERT embeddings, pretrained weights | no |
| `manifests/` | task lists, counts, sha256 | **yes** |

## What must be true before `sha256.txt` means anything

`manifests/sha256.txt` is currently a **placeholder**. It is filled on D0 by
`scripts/01_build_manifests.py`, and `tests/test_manifests.py` only checks the
counts it can check without the payload. A manifest row without a digest is a
claim, not evidence.

## The four similarly named resources

| Name here | Actually | Scale | Not to be confused with |
|---|---|---|---|
| PolyBench | Fudan benchmark, 39 tasks | 42,169 records / 14,717 repeat units | an unrelated LLM benchmark of ~125k polymer QA pairs |
| PolyBench26 | LLNL release, 8 properties | ~250k datapoints | PolyBench (above) |
| OpenPoly | experimental table | **741 structures / 3,985 labels / 26 properties** | OPoly26 (Meta/FAIR, 6.57M DFT calculations for MLIPs) |

The OpenPoly row is the one that has already caused an error once: the 3,985 figure
is *annotations*, not structures, and a second file in the same deposit reports
Tg = 58,755, which counts literature records **before** structure matching. Always
quote 741 structures.
