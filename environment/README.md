# Environments

Three conda environments, already created on this machine:

| File | conda env | Python | GPU | Why separate |
|---|---|---|---|---|
| `b2-core.yml` | `DSC6020_Project_Core` | 3.10.14 | no | RDKit + XGBoost + all statistics |
| `b2-gnn.yml` | `DSC6020_Project_GNN` | 3.10.14 | yes | torch + PyG + chemprop |
| `b2-polybert.yml` | `DSC6020_Project_Polybert` | 3.10.14 | yes | transformers, 7.3 GB weight cache |

**Why three and not one.** RDKit and PyTorch each bundle their own OpenMP runtime;
loading both in one process on Windows is the single most common cause of silent
`libgomp` crashes in cheminformatics pipelines. polyBERT additionally pins
`tokenizers` and needs an isolated 7.3 GB cache that can be deleted without taking
the rest down.

**`pins.json` is the authority, not the yml files.** The yml files are generated
from it. `verify_env.py` compares the live environments against `pins.json` in both
directions: a version lower than pinned *or* higher than pinned both fail, because
under Python 3.10 the failure mode of `pip install -U` is a missing cp310 wheel
several minutes later, not an error here.

## Not installed yet

Per the initialisation agreement, **no packages have been installed into the three
environments**. `verify_env.py` therefore reports every package as missing and
exits non-zero. That is expected:

```
$ python environment/verify_env.py
DSC6020_Project_Core        NOT READY  (10 packages missing)
...
```

## How to install later (one-time)

```bash
# Core: pure pip, no conda solver needed
conda run -n DSC6020_Project_Core    python -m pip install -r environment/requirements-core.txt

# GNN / polyBERT: torch MUST come from the PyTorch index, not from conda
conda run -n DSC6020_Project_GNN     python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
conda run -n DSC6020_Project_GNN     python -m pip install -r environment/requirements-gnn.txt

# Then freeze and commit the lock
conda run -n DSC6020_Project_Core python -m pip freeze --all > environment/lock/core-win-64.lock.txt
```

`conda env create -f environment/*.yml` also works, but mixing conda's torch with
pip's RDKit is the failure mode this layout exists to avoid: install torch by pip.
