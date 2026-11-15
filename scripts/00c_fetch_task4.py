"""C-b-3 取件：把上游 `rlearsch/PolymerBenchmark2026` 的 task4 目录与它需要的源表取到本地。
C-b-3 fetch step: pull upstream's task4 directory and the source tables it needs.

中文：
    用途：`分工_C-b` C-b-3 第①步。钉 commit `83655bc25901a8cabe3a1e5474b7ec26f01b919f`，
          逐文件记录字节数与 sha256，格式与 `data/manifests/external_official_sources.csv` 一致。
          落点是 `data/raw/polybench26/Datasets/...`（`data/` 除 D-38 那 20 份外不入库），
          所以这些字节**不会**进版本库——进库的是清单。
    许可：task4 的训练源是 polyVERSE / VIPEA，属 **GTRC General Public Use License**（D-08），
          不是主表那个 CC BY 4.0；本脚本只取字节，不转发。
English:
    Step 1 of C-b-3. Pins the commit, records bytes and sha256 per file, and drops everything under
    data/raw/polybench26/Datasets (not committed) while the manifest is the committed artefact.
    The task4 training source is polyVERSE/VIPEA under the GTRC licence (D-08), not CC BY 4.0.
"""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
COMMIT = "83655bc25901a8cabe3a1e5474b7ec26f01b919f"
BASE = f"https://raw.githubusercontent.com/rlearsch/PolymerBenchmark2026/{COMMIT}"
RAW = REPO / "data" / "raw" / "polybench26"   # 与 A-2 的落点一致：…/polybench26/Datasets/…
MANIFEST = REPO / "data" / "manifests" / "external_task4_sources.csv"

# 取什么、为什么
# What to fetch and why
FILES = (
    # task4 本体：三个脚本 + 官方已发布成绩表 + 训练入口
    "Datasets/task4_architecture_transfer/README.md",
    "Datasets/task4_architecture_transfer/evaluate_rdkit_ood.py",
    "Datasets/task4_architecture_transfer/summarize_weighted_predictions.py",
    "Datasets/task4_architecture_transfer/task4_all_model_scores.csv",
    "Datasets/task4_architecture_transfer/train_models.sh",
    "Datasets/task4_architecture_transfer/legacy_train.csv",
    # 物化脚本与它的依赖
    "Datasets/Dataset_construction_scripts/create_task4_shared_splits.py",
    "Datasets/Dataset_construction_scripts/process_Vipea_data.py",
    "Datasets/Dataset_construction_scripts/polymer_conversions.py",
    "Datasets/Dataset_construction_scripts/PSMILES_to_wPSMILES.py",
    "Datasets/Dataset_construction_scripts/rdkit_descriptor_cache.py",
    "Datasets/Dataset_construction_scripts/task4_architecture_transfer.py",
    # 官方地板：RDKit RF 的训练入口（train_models.sh rdkit_rf 调的就是这一个）
    "Models/RDKit_RF/train_rf_scaling.py",
    "Models/RDKit_RF/train_rf.py",
    "Models/RDKit_RF/requirements.txt",
    "Models/RDKit_RF/README.md",
    # 源表：polyVERSE EA（369 行）+ VIPEA 的两份 chemprop 源表（共 17 MB）
    "Datasets/PSMILES/polyVERSE/electron_affinity/electron_affinity_data_polymers_v4.csv",
    "Datasets/Dataset_construction_scripts/files/polymer-chemprop-data/dataset-poly_chemprop.csv",
    "Datasets/Dataset_construction_scripts/files/polymer-chemprop-data/dataset.csv",
)

COLUMNS = ("path", "bytes", "sha256", "source_commit", "source_url", "role", "license")


def sha256_of(path: Path) -> str:
    """文件字节的 sha256。／ sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    """取件并写清单。／ Fetch the files and write the manifest."""
    rows: list[dict[str, str]] = []
    for relative in FILES:
        target = RAW / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE}/{relative}"
        if target.is_file():
            payload = target.read_bytes()
            status = "cached"
        else:
            payload = urllib.request.urlopen(url, timeout=120).read()
            target.write_bytes(payload)
            status = "fetched"
        role = (
            "task4 物化/评估脚本与官方成绩表"
            if "task4_architecture_transfer/" in relative
            else ("物化脚本依赖" if "Dataset_construction_scripts" in relative else "训练源表")
        )
        rows.append(
            {
                "path": relative,
                "bytes": str(len(payload)),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_commit": COMMIT,
                "source_url": url,
                "role": role,
                "license": "GTRC General Public Use License（polyVERSE / VIPEA）",
            }
        )
        print(f"{status:>8} {len(payload):>9} B  {relative}")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    total = sum(int(row["bytes"]) for row in rows)
    print(f"清单 → {MANIFEST.relative_to(REPO).as_posix()}（{len(rows)} 份、{total / 1e6:.1f} MB）")
    print(json.dumps({"commit": COMMIT, "files": len(rows), "bytes": total}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
