"""A-5 的断言测试：官方划分快照没被重算、没被改过、也没被"顺手切一刀"。
Assertions for A-5: the official split snapshots were not recomputed, edited, or "just re-cut".

中文：
    用途：把分工书 A-5 的八条验收标准里能自动化的部分变成 `pytest`。三类测试：
          ① **正向**：`verify_splits()` 在真实产物上必须返回空列表（deep 与非 deep 各一遍）；
          ② **不变量**：互斥、覆盖全量、比例、前缀嵌套、折文件结构；
          ③ **能抓住真错**：故意改坏一份划分、故意在 `src/` 里写一句 `train_test_split`，
             测试必须变红。第 ③ 类才是 D 的入口审计会用的那一类——只断言"没报错"的测试，
             拦不住 Proposal §7.5 列的头号失败模式。
    依赖：`splits/**`（A-5 产物，入库）、`configs/tasks/*.yaml`（A-4）、
          `data/interim/harmonised/**`（A-3，不入库；缺了则 deep 测试跳过）。
    配置：`configs/tasks/` 的 `split_status` —— 它决定"哪些任务必须有划分"。
    对外：pytest 用例。被谁用：CI、D 的入口审计、任何改了 `splits/` 的人。

English:
    Purpose: turn the automatable part of A-5's eight acceptance criteria into `pytest`. Three
    groups: (1) positive - `verify_splits()` returns an empty list on the real artefacts (both
    deep and shallow); (2) invariants - disjointness, full coverage, ratio, prefix nesting, fold
    structure; (3) real-failure detection - corrupt a split on purpose, or write a
    `train_test_split` into `src/`, and the tests must go red. Group 3 is what D's entry audit
    uses; a test that only asserts "no exception" cannot catch Proposal §7.5's number-one
    failure mode.
    Depends on: splits/ (A-5, committed), configs/tasks (A-4), data/interim/harmonised (A-3, not
    committed - the deep test skips when absent).
    Config: `split_status` in configs/tasks decides which tasks must have a split.
    Public API: pytest cases. Callers: CI, D's entry audit, anyone touching splits/.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from chainworth.io.tasks import load_all_task_configs
from chainworth.protocol import splits as S

CONFIGS = load_all_task_configs(S.CONFIGS_DIR)
SPLIT_TASKS = {tid: cfg for tid, cfg in CONFIGS.items() if cfg["split_status"] == "present"}

# 需要 A-3 的中间表才能重算；那份文件不入库，克隆后可能不在。
# The A-3 intermediate tables are needed for the deep recompute and are not committed.
NEEDS_INTERIM = pytest.mark.skipif(
    not all((S.REPO_ROOT / str(cfg["data_ref"])).is_file()
            for cfg in SPLIT_TASKS.values()
            if cfg["benchmark"] == "polybench"),
    reason="缺 data/interim/harmonised/*.parquet（不入库，先跑 scripts/01_harmonise.py）",
)


# ---- ① 正向：真产物上必须全过 / Positive: the real artefacts must pass ----


def test_split_status_has_no_pending_work() -> None:
    """除"整任务不可用"的以外，不应再有 pending。 / Nothing may stay pending except unavailable tasks."""
    pending = {tid: cfg for tid, cfg in CONFIGS.items() if cfg["split_status"] == "pending"}
    for task_id, config in pending.items():
        assert config["status"] == "unavailable", (
            f"{task_id} 还挂 pending 但它是可用任务：A-5 漏做了它"
        )


def test_verify_splits_passes() -> None:
    """`verify_splits()` 在入库产物上必须返回空列表。 / Must return an empty list on real artefacts."""
    problems = S.verify_splits(CONFIGS, deep=False)
    assert problems == [], "划分校验未通过：\n" + "\n".join(problems)


@NEEDS_INTERIM
def test_verify_splits_deep_passes() -> None:
    """`--deep`：拿官方 CSV 重算行号，必须与落盘的一致。 / Deep: recomputed indices must match."""
    problems = S.verify_splits(CONFIGS, deep=True)
    assert problems == [], "深度校验未通过：\n" + "\n".join(problems)


def test_snapshot_manifest_covers_every_present_task() -> None:
    """每个 present 的 PolyBench / PolyBench26 任务都要在 `splits_manifest.csv` 里有行。
    Every present task must appear in the snapshot manifest."""
    recorded = S._read_manifest(S.SPLITS_MANIFEST)
    for task_id, config in SPLIT_TASKS.items():
        if config["benchmark"] == "polybench":
            assert task_id in recorded, f"{task_id} 不在 splits_manifest.csv"
        elif config["benchmark"] == "polybench26":
            prefix = f"{task_id}/" if str(config["split_ref"]).endswith("/") else task_id
            assert any(key == prefix or key.startswith(prefix) for key in recorded), (
                f"{task_id} 不在 splits_manifest.csv"
            )


# ---- ② 不变量 / Invariants ----


@pytest.mark.parametrize("task_id", sorted(t for t, c in SPLIT_TASKS.items() if c["benchmark"] == "polybench"))
def test_polybench_split_invariants(task_id: str) -> None:
    """互斥、覆盖全量、无重复、比例 30%±0.5pp、协议是固定 70/30。
    Disjoint, covers everything, no duplicates, 30% +/- 0.5pp, protocol is the fixed 70/30."""
    config = SPLIT_TASKS[task_id]
    split = json.loads((S.REPO_ROOT / str(config["split_ref"])).read_text(encoding="utf-8"))
    test_index = split["test_index"]
    train_index = split["train_index"]
    n_total = int(config["n_labels"])

    assert len(set(test_index) & set(train_index)) == 0, f"{task_id}：train/test 有交集"
    assert len(set(test_index)) == len(test_index), f"{task_id}：test 有重复行号"
    assert len(set(train_index)) == len(train_index), f"{task_id}：train 有重复行号"
    assert sorted(test_index + train_index) == list(range(n_total)), f"{task_id}：没有覆盖 0..{n_total - 1}"
    assert abs(len(test_index) / n_total - 0.30) <= 0.005, f"{task_id}：测试比例不是约 30%"
    assert split["protocol"] == "official_70_30", f"{task_id}：协议不是 official_70_30"
    assert "kfold" not in split["protocol"].lower(), f"{task_id}：PolyBench 不许出现 k 折"


@pytest.mark.parametrize(
    "task_id,directory",
    [
        ("pb26_polymetrix_tg", "pb26_polymetrix_tg/seed_42"),
        ("pb26_homopolymer_density", "pb26_homopolymer_density/seed_42"),
        ("pb26_scaling_cp/md5000_alternating_cp", "pb26_scaling_cp/md5000_alternating_cp/seed_42"),
        ("pb26_scaling_cp/md5000_homopolymer_cp", "pb26_scaling_cp/md5000_homopolymer_cp/seed_42"),
    ],
)
def test_polybench26_folds_are_a_partition(task_id: str, directory: str) -> None:
    """每折的 test+val+train_pool 必须恰好覆盖全量一次，且档位不超池子。
    Each fold's test+val+train_pool must cover every row exactly once, and tiers must fit the pool."""
    root = S.SPLITS_ROOT / "polybench26" / directory
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    n_rows = int(manifest["n_rows"])
    for fold_meta in manifest["folds"]:
        arrays = S.load_fold(root, int(fold_meta["fold"]))
        merged = np.concatenate([arrays["test"], arrays["val"], arrays["train_pool"]])
        assert len(merged) == n_rows, f"{task_id} 第 {fold_meta['fold']} 折：三段相加不等于全量"
        assert sorted(merged.tolist()) == list(range(n_rows)), f"{task_id}：不是恰好覆盖 0..{n_rows - 1}"
        for tier in manifest["train_sizes"]:
            assert tier <= len(arrays["train_pool"]), f"{task_id}：档位 {tier} 超过池子"


def test_nested_prefix_is_a_prefix_and_nests() -> None:
    """档位是前缀：300 ⊂ 1000 ⊂ 3000，且不重排。 / Tiers are prefixes and nest without reshuffling."""
    pool = list(range(5000, 0, -1))  # 故意倒序，确保函数没有偷偷排序 / deliberately descending
    small = S.nested_prefix(pool, 300)
    big = S.nested_prefix(pool, 1000)
    assert small == pool[:300], "nested_prefix 不只是取前 n 个"
    assert big[:300] == small, "档位之间不是嵌套前缀"
    with pytest.raises(S.SplitError):
        S.nested_prefix(pool, 6000)


def test_load_fold_rejects_missing_file(tmp_path: Path) -> None:
    """缺折文件必须抛 FileNotFoundError，而不是返回空数组。 / A missing fold must raise."""
    with pytest.raises(FileNotFoundError):
        S.load_fold(tmp_path, 0)


# ---- ③ 能抓住真错 / Real-failure detection ----


def test_tampered_split_is_detected(tmp_path: Path) -> None:
    """删掉一个测试行号，校验必须报错（不是"跑得通就行"）。
    Drop one test index and the check must complain - "it runs" is not the bar."""
    config = dict(SPLIT_TASKS["pb_t1_tg"])
    source = S.REPO_ROOT / str(config["split_ref"])
    tampered = tmp_path / "official_70_30.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["test_index"] = payload["test_index"][:-1]
    tampered.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    problems: list[str] = []
    S._verify_polybench({**config, "split_ref": str(tampered)}, problems, deep=False)
    assert problems, "删了一个测试行号却没报错——这条测试没在防任何东西"


def test_swapped_protocol_is_detected(tmp_path: Path) -> None:
    """把协议改成 k 折，校验必须报错（PolyBench 上不许 k 折）。
    Switching the protocol to k-fold must be rejected - PolyBench has no k-fold."""
    config = dict(SPLIT_TASKS["pb_t1_tm"])
    source = S.REPO_ROOT / str(config["split_ref"])
    tampered = tmp_path / "official_70_30.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["protocol"] = "official_5fold"
    tampered.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    problems: list[str] = []
    S._verify_polybench({**config, "split_ref": str(tampered)}, problems, deep=False)
    assert problems, "把 70/30 改成 5 折却没报错"


def test_no_self_cut_split_in_src() -> None:
    """`src/` 里不许出现 `train_test_split` / `KFold`：那是 Proposal §7.5 的头号红线。
    `src/` must not contain `train_test_split` or `KFold` - Proposal §7.5's first red line.

    中文：
        用途：D 复核入口时会**故意在 `src/` 里注入一次 `train_test_split`**，看这条测试红不红。
              所以这里不能只扫"我们记得的那几个文件"，要扫整个 `src/`，且不放过
              `StratifiedKFold` / `GroupKFold` 这类同族名字。
    English:
        Purpose: D's audit injects a `train_test_split` into `src/` and expects this test to go
                 red, so the scan must cover the whole tree and the sibling names too.
    """
    offenders = S.find_self_cut_calls()
    assert not offenders, (
        "src/ 里出现了自己切数据的调用（评估必须用官方划分）：\n" + "\n".join(offenders)
    )


def test_self_cut_scan_catches_an_injected_call(tmp_path: Path) -> None:
    """注入一次真调用，扫描必须报出来——这条测试本身证明上一条不是空转。
    Inject a real call and the scan must report it: this test proves the previous one is not a
    no-op. It mirrors what D's entry audit does."""
    fake = tmp_path / "injected.py"
    fake.write_text(
        "from sklearn.model_selection import train_test_split\n"
        "def bad(x, y):\n"
        "    return train_test_split(x, y, test_size=0.3)\n",
        encoding="utf-8",
    )
    offenders = S.find_self_cut_calls(tmp_path)
    assert offenders and "train_test_split" in offenders[0]


def test_openpoly_has_no_home_made_split() -> None:
    """OpenPoly 目录里只允许结论文件，不许出现自己切的划分。
    OpenPoly may hold only the conclusion file, never a home-made split."""
    directory = S.SPLITS_ROOT / "openpoly"
    extra = [p.name for p in directory.iterdir() if p.suffix == ".json" and p.name != "NOT_AVAILABLE.json"]
    assert not extra, f"splits/openpoly 里出现了划分文件：{extra}（D-22：官方没发布，不许自己切）"
    note = json.loads((directory / "NOT_AVAILABLE.json").read_text(encoding="utf-8"))
    assert note["status"] == "blocked_upstream"
    assert note["decision"] == "D-22"


def test_polybench26_copies_are_byte_identical() -> None:
    """仓库里的索引拷贝必须与 `data/raw` 下的官方副本逐字节相同（data/raw 不在则跳过）。
    The committed copies must equal the official ones byte for byte (skipped without data/raw)."""
    pairs = [
        (
            S.SPLITS_ROOT / "polybench26/pb26_polymetrix_tg/seed_42",
            S.PB26_ROOT / "Datasets/scaling_indices/PolyMetriX/Tg/Tg/seed_42",
        ),
        (
            S.SPLITS_ROOT / "polybench26/pb26_scaling_cp/md5000_alternating_cp/seed_42",
            S.PB26_ROOT / "Datasets/scaling_indices/MD_5000/Cp/alternating_Cp/seed_42",
        ),
    ]
    for target, source in pairs:
        if not source.is_dir():
            pytest.skip(f"缺官方副本 {source}（data/raw 不入库）")
        for path in sorted(target.iterdir()):
            assert S.sha256_of(path) == S.sha256_of(source / path.name), (
                f"{path.name} 与 data/raw 下的官方副本不一致——有人改了冻结物"
            )


def test_canonical_tables_match_official_digest() -> None:
    """canonical 表的 sha256 必须等于官方 manifest 的 `canonical_sha256`（验收 ⑥）。
    The canonical tables must match the official `canonical_sha256` (criterion 6)."""
    if not S.PB26_SCRIPTS.is_dir():
        pytest.skip("缺 data/raw/polybench26（不入库）")
    expected = S._canonical_sha256_map(CONFIGS)
    assert expected, "官方 manifest 里一个 canonical_sha256 都没读到"
    for source, digest in expected.items():
        path = S.PB26_CANONICAL_ROOT / source
        if not path.is_file():
            pytest.skip(f"canonical 表未物化：{source}（先跑 scripts/03_splits.py --source canonical）")
        assert S.lf_normalized_sha256(path) == digest, f"{source} 与官方 canonical_sha256 不等"


def test_verify_flags_a_removed_artifact(tmp_path: Path) -> None:
    """把一份划分"搬走"，`verify_splits()` 必须报出来（而不是静静跳过）。
    Move one split away and `verify_splits()` must report it instead of silently skipping."""
    config = dict(SPLIT_TASKS["pb_t3_iwlt"])
    problems: list[str] = []
    S._verify_polybench({**config, "split_ref": str(tmp_path / "nope.json")}, problems, deep=False)
    assert any("不存在" in problem for problem in problems)
