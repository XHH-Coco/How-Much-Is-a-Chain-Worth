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

# OpenPoly 是"规则复现"而不是"官方快照"，单独一组（D-24）。
# OpenPoly is a rule reproduction rather than an official snapshot, so it gets its own group.
OPENPOLY_TASKS = {
    tid: cfg for tid, cfg in CONFIGS.items() if cfg["split_status"] == "rule_replicated"
}

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


def test_snapshot_manifest_covers_every_split_task() -> None:
    """每个有划分的任务（present 或 rule_replicated）都要在 `splits_manifest.csv` 里有行。
    Every task that has a split - present or rule_replicated - must appear in the snapshot
    manifest.

    中文：用 `_read_manifest_rows()` 而不是按 task_id 建键的那版——OpenPoly 一条任务有 10 行，
          按 task_id 建键会把其中 9 行挤掉，测试反而看不见缺口。
    English: use the row-list reader; keying by task_id would collapse OpenPoly's ten rows per
    task into one and hide exactly the gap this test is meant to find.
    """
    recorded = [row["artifact"] for row in S._read_manifest_rows(S.SPLITS_MANIFEST)]

    def covered(artifact: str, reference: str) -> bool:
        """产物与配置路径在**目录层级上相交**就算登记过：PolyBench 是同一个文件，
        PolyBench26 记的是 `split_ref` 的父目录（`seed_42`），OpenPoly 记的是目录里的子文件。
        A hit means the two paths intersect in the directory tree: PolyBench records the same
        file, PolyBench26 records the parent (`seed_42`), OpenPoly records children of the dir.
        """
        artifact, reference = artifact.rstrip("/"), reference.rstrip("/")
        return (
            artifact == reference
            or artifact.startswith(reference + "/")
            or reference.startswith(artifact + "/")
        )

    for task_id, config in {**SPLIT_TASKS, **OPENPOLY_TASKS}.items():
        reference = str(config["split_ref"])
        assert any(covered(artifact, reference) for artifact in recorded), (
            f"{task_id} 的划分产物不在 splits_manifest.csv（split_ref={reference}）"
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


def test_openpoly_protocol_note_quotes_the_paper() -> None:
    """规则说明必须引论文原文，且如实声明"作者没给种子"。
    The protocol card must quote the paper and admit the authors published no seed."""
    note = json.loads(S.OPENPOLY_PROTOCOL_NOTE.read_text(encoding="utf-8"))
    assert note["status"] == "rule_replicated"
    assert note["decision"] == "D-24"
    quotes = " ".join(note["rule"]["quotes"])
    assert "8:1:1" in quotes and "10 equal-width bins" in quotes, "规则引文缺关键句"
    assert note["rule"]["seed_published"] is False


def test_openpoly_does_not_pretend_to_be_official() -> None:
    """OpenPoly 目录里不许冒出 `*_official*.json`——那是冒充官方划分（D-24 之后仍防）。
    No `*_official*.json` may appear: that would be passing a reproduction off as official."""
    directory = S.SPLITS_ROOT / "openpoly"
    pretend = sorted(path.name for path in directory.rglob("*_official*.json"))
    assert not pretend, f"splits/openpoly 里出现了冒充官方的文件：{pretend}"
    for task_id, config in OPENPOLY_TASKS.items():
        assert not (S.REPO_ROOT / str(config["split_ref"]) / f"{task_id}_official.json").exists()


@pytest.mark.parametrize("task_id", sorted(OPENPOLY_TASKS))
def test_openpoly_splits_follow_the_published_rule(task_id: str) -> None:
    """每条任务的全部种子都必须守规则：8:1:1、10 箱、互斥、覆盖全量、行号可回溯。
    Every seed of every task must obey the rule: 8:1:1, ten bins, disjoint, exhaustive, traceable."""
    directory = S.REPO_ROOT / str(OPENPOLY_TASKS[task_id]["split_ref"])
    for seed in S.OPENPOLY_SEEDS:
        payload = json.loads((directory / f"rule_replicated_seed{seed}.json").read_text(encoding="utf-8"))
        problems: list[str] = []
        S._check_openpoly_split(payload, problems, task_id=task_id, seed=seed, deep=False)
        assert problems == [], "\n".join(problems)


@pytest.mark.parametrize("task_id", sorted(OPENPOLY_TASKS))
def test_openpoly_splits_are_reproducible(task_id: str) -> None:
    """同一种子在发布表上重算，行号必须逐位相同——这是"复现"而不是"自创"的凭据。
    Recomputing a seed on the published table must give identical indices: that is what makes
    this a reproduction rather than an invention."""
    config = OPENPOLY_TASKS[task_id]
    if not (S.REPO_ROOT / str(config["data_ref"])).is_file():
        pytest.skip("缺 data/raw/openpoly（不入库）")
    directory = S.REPO_ROOT / str(config["split_ref"])
    for seed in (0, 9):
        again = S.build_openpoly_split(config, seed=seed)
        on_disk = json.loads((directory / f"rule_replicated_seed{seed}.json").read_text(encoding="utf-8"))
        for name in ("train", "val", "test"):
            assert again[f"{name}_index"] == on_disk[f"{name}_index"], f"{task_id} seed {seed} 的 {name} 重算不一致"


@pytest.mark.parametrize("task_id", sorted(OPENPOLY_TASKS))
def test_openpoly_seeds_actually_differ(task_id: str) -> None:
    """不同种子必须给出不同的测试集，否则"10 份"是假的（比如代码忽略了 seed）。
    Different seeds must give different test sets, otherwise the "ten splits" are fake."""
    directory = S.REPO_ROOT / str(OPENPOLY_TASKS[task_id]["split_ref"])
    seen = set()
    for seed in S.OPENPOLY_SEEDS:
        payload = json.loads((directory / f"rule_replicated_seed{seed}.json").read_text(encoding="utf-8"))
        seen.add(tuple(payload["test_index"]))
    assert len(seen) == len(S.OPENPOLY_SEEDS), f"{task_id}：{len(seen)} 个不同的测试集，应有 {len(S.OPENPOLY_SEEDS)} 个"


def test_openpoly_check_catches_a_broken_partition() -> None:
    """结构被改坏（行号重复）必须当场报出来——证明浅层校验不是空转。
    A structurally broken partition (duplicate index) must be reported on the spot: this proves
    the shallow check is not a no-op."""
    task_id = sorted(OPENPOLY_TASKS)[0]
    directory = S.REPO_ROOT / str(OPENPOLY_TASKS[task_id]["split_ref"])
    payload = json.loads((directory / "rule_replicated_seed0.json").read_text(encoding="utf-8"))
    payload["train_index"] = sorted(payload["train_index"] + [payload["test_index"][0]])
    problems: list[str] = []
    S._check_openpoly_split(payload, problems, task_id=task_id, seed=0, deep=False)
    assert any("重叠" in problem or "合计" in problem for problem in problems), problems


def test_openpoly_shallow_check_cannot_see_a_one_row_swap() -> None:
    """**记录一条已知局限**：把一个样本从测试集挪进训练集，三份仍互斥、仍覆盖全量，
    比例只动 1/443 ≈ 0.2pp，浅层校验看不出——只有按种子重算（`deep=True`）才抓得到。
    这条测试不是在测代码，是在防止下一个人误以为浅层校验足够。
    Records a known limitation: moving one row from test to train keeps the three sets disjoint
    and exhaustive and moves the ratios by 1/443 ~ 0.2pp, so the shallow check cannot see it; only
    the `deep` recomputation can. This test exists to stop the next person believing shallow is
    enough."""
    task_id = sorted(OPENPOLY_TASKS)[0]
    directory = S.REPO_ROOT / str(OPENPOLY_TASKS[task_id]["split_ref"])
    payload = json.loads((directory / "rule_replicated_seed0.json").read_text(encoding="utf-8"))
    stolen = payload["test_index"][0]
    payload["test_index"] = payload["test_index"][1:]
    payload["train_index"] = sorted(payload["train_index"] + [stolen])
    payload["n_test"] -= 1
    payload["n_train"] += 1
    problems: list[str] = []
    S._check_openpoly_split(payload, problems, task_id=task_id, seed=0, deep=False)
    assert problems == [], "浅层校验居然抓到了——那说明这条局限已不存在，请删掉这条测试"


def test_openpoly_deep_check_catches_a_one_row_swap() -> None:
    """同样的交换，`deep=True` 必须抓到——这才是 `--verify --deep` 存在的理由。
    The same swap must be caught by `deep=True`; that is why `--verify --deep` exists."""
    task_id = sorted(OPENPOLY_TASKS)[0]
    config = OPENPOLY_TASKS[task_id]
    if not (S.REPO_ROOT / str(config["data_ref"])).is_file():
        pytest.skip("缺 data/raw/openpoly（不入库），无法重算")
    directory = S.REPO_ROOT / str(config["split_ref"])
    payload = json.loads((directory / "rule_replicated_seed0.json").read_text(encoding="utf-8"))
    stolen = payload["test_index"][0]
    payload["test_index"] = payload["test_index"][1:]
    payload["train_index"] = sorted(payload["train_index"] + [stolen])
    payload["n_test"] -= 1
    payload["n_train"] += 1
    problems: list[str] = []
    S._check_openpoly_split(
        payload, problems, task_id=task_id, seed=0, deep=True, config=config
    )
    assert any("不一致" in problem for problem in problems), problems


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
