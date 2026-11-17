"""D-8 对账的守门：全绿只是起点，**故意弄坏时必须点出 run_id** 才算闸门。
D-8 provenance gates: all-green is the starting point; the gate only counts when a deliberately
broken tree names the offending run_id.

中文：
    用途：分工书 D-8 的三条验收逐条落成断言：
          ① 故意删掉一个预测文件，对账表必须点出那个 `run_id`（不许静默跳过）；
          ② 通过率 100% 才允许出图 → `require_clean()` 在仍有 `fail` 时抛 `ProvenanceError`；
          ③ 任何缺口都留痕 → `exempt` 必须带登记过的理由，不许出现"空理由豁免"。
    为什么反向用例占多数：C-c 验收的教训（D-62）——只跑绿的检查等于没写。这里每一条"正向"
          断言旁边都有一条"把它弄坏，必须红"。
    数据来源：`results/raw_runs/`（真仓库）、`tmp_path`（反向用例，全在临时目录里造）。
    副作用：临时目录内写盘；真仓库只读（`test_real_repo_passes_the_four_way_audit` 不落盘）。
    被谁用：CI、D 的交接自检。

English:
    Purpose: the three D-8 acceptance criteria as assertions - a deleted prediction must be named
    by run_id, `require_clean()` must raise while any failure remains, and every exemption must
    carry a registered reason. Most cases are reverse cases, following the C-c lesson (D-62):
    a check that only ever passes proves nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from chainworth.analysis import provenance as P

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# 造一个最小的"仓库"/ Build a minimal fake tree
# --------------------------------------------------------------------------- #
def _write_prediction(root: Path, run_id: str, row_ids: list[int], n_test: int | None = None) -> Path:
    """在临时仓库里放一份预测 parquet（`row_id, y_pred`）。
    Drop one prediction parquet (`row_id, y_pred`) into the fake repo.
    """
    directory = root / "results" / "aggregate" / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.parquet"
    pd.DataFrame({"row_id": row_ids, "y_pred": [0.0] * len(row_ids)}).to_parquet(path, index=False)
    return path


def _record(
    root: Path,
    run_id: str = "task__a2_conventional__fold__seed0",
    n_test: int = 3,
    split_relative: str = "splits/polybench/task/official_70_30.json",
    split_bytes: bytes | None = b'{"a": 1}',
) -> dict:
    """造一条 run 记录，并把它的划分文件与指纹文件按需放好。
    Build one record and lay down the split/fingerprint files it points at.
    """
    split_path = root / split_relative
    split_path.parent.mkdir(parents=True, exist_ok=True)
    if split_bytes is not None and not split_path.exists():
        split_path.write_bytes(split_bytes)
    digest = P._sha256_prefix(split_path) if split_path.exists() else "0" * 12

    fingerprint = root / "results" / "env_fingerprints" / "env_core.json"
    fingerprint.parent.mkdir(parents=True, exist_ok=True)
    fingerprint.write_text("{}", encoding="utf-8")

    return {
        "run_id": run_id,
        "n_test": n_test,
        "split_version": f"{digest}@{split_relative}",
        "env_fingerprint": "results/env_fingerprints/env_core.json",
        "predictions": "",
    }


def _status(rows, check: str) -> list[str]:
    """取某一类检查的状态列表。/ The statuses of one check."""
    return [row.status for row in rows if row.check == check]


# --------------------------------------------------------------------------- #
# ① 真仓库：必须全绿，且豁免都带登记理由
# --------------------------------------------------------------------------- #
def test_real_repo_passes_the_four_way_audit() -> None:
    """真仓库的四向对账：一条 `fail` 都不许有；每个 `exempt` 都必须带登记过的理由。
    The real tree must have zero failures, and every exemption must carry a registered reason.

    中文：这条在**干净 clone** 上也必须绿：`splits/polybench26/arch/` 按 D-57 不入库，
          15 条 arch 记录会落进 `exempt`（缺件），而缺件不是记录错。所以判据是"零 fail"，
          不是"零 exempt"——把两者混成一个红，就会重演 C-c 把"环境没物化"当成"记录坏了"。
    English: green even on a clean clone: the arch splits are intentionally uncommitted (D-57), so
    fifteen records land in `exempt`. The criterion is zero failures, not zero exemptions.

    中文（2026-11-17 修）：原先这里把 922 写死，链臂一加 3 条记录就假红——而记录条数本来就该
          由 `raw_runs/` 现场数出来（D-62："期望条数只有一处出处"）。现在改成两条**与机器无关**
          的判据：①每条记录都必须进过逐条检查（防静默跳过）；②指纹与预测行数这两条**不碰物化**
          的检查必须全员 ok，划分那条允许"存在即核哈希"与"缺件豁免"互补着凑满。
    English (fixed 2026-11-17): the count is now taken from `raw_runs/` instead of being hardcoded,
    and the coverage criterion is: every record was audited, the two checks that do not touch
    materialisation are all `ok`, and the split check is covered by `ok` plus "missing" exemptions.
    """
    rows = P.collect()
    summary = P.summarise(rows)

    assert summary["n_fail"] == 0, P.render(summary, limit=10)
    assert summary["pass_rate"] == 1.0

    n_records = len(P.load_records(P.REPO_ROOT / "results" / "raw_runs"))
    assert n_records > 900, f"记录只剩 {n_records} 条，先看看是不是 raw_runs 读漏了"
    # ① 覆盖：逐条检查必须覆盖全部记录。这里只取 `check_runs` 的输出——`collect()` 的结果里
    #    还混着汇总表的行，那些行的 run_id 来自表里引用的 run，会把覆盖数顶到记录数之上。
    audited = {row.run_id for row in P.check_runs(root=P.REPO_ROOT)}
    assert len(audited) == n_records, f"{n_records - len(audited)} 条记录没进逐条检查"
    # ② 这两条只读记录与 predictions/，与物化无关 → 不许有 exempt。
    for check in ("env_fingerprint_present", "prediction_rows_equal_n_test"):
        assert summary["by_check"][check][P.STATUS_OK] == n_records, check
    # ③ 划分：在场就核哈希（ok），不在场只许是物化缺件（exempt），两者之和 = 全部记录。
    #    本机物化齐全时 `split_file_present` 一行都不产生，所以取缺省 0。
    split_ok = summary["by_check"]["split_digest_matches"][P.STATUS_OK]
    split_missing = summary["by_check"].get("split_file_present", {}).get(P.STATUS_EXEMPT, 0)
    assert split_ok + split_missing == n_records, (split_ok, split_missing, n_records)

    for row in rows:
        if row.status != P.STATUS_EXEMPT:
            continue
        # 豁免的理由必须来自登记表，不许是"我跳过了"。
        assert any(
            row.detail.startswith(reason) or reason in row.detail for reason in P.EXEMPT_REASONS
        ), f"未登记的豁免：{row.check} {row.run_id} {row.detail}"


# --------------------------------------------------------------------------- #
# ①' 反向：删一份预测，必须点名那个 run_id
# --------------------------------------------------------------------------- #
def test_a_missing_prediction_is_named_not_skipped(tmp_path: Path) -> None:
    """删掉预测文件后，对账表必须出现那条 run_id 的 `fail`——不许静默跳过。
    With the parquet deleted, the table must carry a failure naming that run_id.

    中文：这是 D-8 验收①的原句。判据落在 `run_id` 上，不是"失败数 +1"——
          后者在几百条里看不出是谁坏了。
    """
    root = tmp_path
    record = _record(root)
    rows = P.check_runs([record], root=root)

    assert _status(rows, "prediction_present") == [P.STATUS_FAIL]
    failures = [row for row in rows if row.status == P.STATUS_FAIL]
    assert [row.run_id for row in failures] == [record["run_id"]]
    assert "不存在" in failures[0].detail


def test_prediction_row_count_must_equal_n_test(tmp_path: Path) -> None:
    """预测行数与记录的 `n_test` 不等 → 红。这是"算 R² 的点"与"落盘的点"不是同一批。
    Rows != `n_test` is a failure: the points behind R^2 and the points on disk differ.
    """
    root = tmp_path
    record = _record(root, n_test=5)
    _write_prediction(root, record["run_id"], row_ids=[0, 1, 2])

    rows = P.check_runs([record], root=root)
    assert _status(rows, "prediction_rows_equal_n_test") == [P.STATUS_FAIL]
    assert "5" in next(row.detail for row in rows if row.check == "prediction_rows_equal_n_test")


def test_duplicate_row_ids_in_a_prediction_fail(tmp_path: Path) -> None:
    """`row_id` 重复 → 红：bootstrap 会把同一个点重复计。
    Duplicate `row_id` fails; bootstrap would count one point twice.
    """
    root = tmp_path
    record = _record(root, n_test=3)
    _write_prediction(root, record["run_id"], row_ids=[0, 0, 1])

    rows = P.check_runs([record], root=root)
    assert _status(rows, "prediction_row_ids_unique") == [P.STATUS_FAIL]


# --------------------------------------------------------------------------- #
# ② 反向：划分数被换 → fail；未物化 → exempt（两者不是一个红）
# --------------------------------------------------------------------------- #
def test_a_tampered_split_digest_fails(tmp_path: Path) -> None:
    """记录里的摘要与文件字节对不上 → `fail`，并说出两边各是什么。
    A digest mismatch is a failure that names both digests.
    """
    root = tmp_path
    record = _record(root, split_bytes=b'{"a": 1}')
    # 事后换掉文件内容：手写版本号不会变，字节哈希会变。
    (root / "splits/polybench/task/official_70_30.json").write_bytes(b'{"a": 2}')
    _write_prediction(root, record["run_id"], row_ids=[0, 1, 2])

    rows = P.check_runs([record], root=root)
    assert _status(rows, "split_digest_matches") == [P.STATUS_FAIL]
    detail = next(row.detail for row in rows if row.check == "split_digest_matches")
    assert record["split_version"].split("@")[0] in detail


def test_unmaterialised_arch_split_is_exempt_with_a_rebuild_hint(tmp_path: Path) -> None:
    """D-57 的那份划分没物化 → `exempt` 且打印重建命令；判成 `fail` 会把缺件误读成记录坏。
    An un-materialised D-57 split is an exemption carrying the rebuild command; calling it a
    failure would read a missing file as a broken record.
    """
    root = tmp_path
    record = _record(
        root,
        split_relative="splits/polybench26/arch/seed_42/fold_00.npz",
        split_bytes=None,
    )
    _write_prediction(root, record["run_id"], row_ids=[0, 1, 2])

    rows = P.check_runs([record], root=root)
    assert _status(rows, "split_file_present") == [P.STATUS_EXEMPT]
    detail = next(row.detail for row in rows if row.check == "split_file_present")
    assert "arch --setup" in detail


def test_materialised_arch_split_must_match_when_present(tmp_path: Path) -> None:
    """物化**存在**但对不上 → 必须 `fail`，不许顺着上面的豁免溜过去。
    When the D-57 split *is* present but mismatched, it must fail - the exemption must not leak.

    中文：这条是"防放水"的反向测试。没有它，`exempt` 那条分支会把"本机物化了但文件是旧的"
          也一起吞掉——正是 C-c 里 `_resolve_fold()` 把两件事混成一个红的镜像问题。
    English: the anti-leak test. Without it, the exemption branch would also swallow "materialised
    locally but stale", which is the mirror image of the C-c `_resolve_fold()` problem.
    """
    root = tmp_path
    record = _record(
        root,
        split_relative="splits/polybench26/arch/seed_42/fold_00.npz",
        split_bytes=b"not-the-real-npz",
    )
    _write_prediction(root, record["run_id"], row_ids=[0, 1, 2])
    (root / "splits/polybench26/arch/seed_42/fold_00.npz").write_bytes(b"tampered")

    rows = P.check_runs([record], root=root)
    assert _status(rows, "split_digest_matches") == [P.STATUS_FAIL]


# --------------------------------------------------------------------------- #
# ③ 汇总表：未登记的表、空 run_id、RF 豁免
# --------------------------------------------------------------------------- #
def _aggregate_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "results" / "aggregate"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def test_unregistered_aggregate_table_fails(tmp_path: Path) -> None:
    """目录里出现一张没登记的表 → `fail`：新数字不许静默获得"已对账"身份。
    An unregistered table fails: a new number must not silently acquire an audited identity.
    """
    directory = _aggregate_dir(tmp_path)
    (directory / "brand_new_table.csv").write_text("task,r2\nx,0.9\n", encoding="utf-8")

    rows = P.check_aggregates(["known"], directory)
    assert _status(rows, "aggregate_table_registered") == [P.STATUS_FAIL]
    assert "brand_new_table.csv" in next(
        row.detail for row in rows if row.check == "aggregate_table_registered"
    )


def test_derived_table_is_exempt_with_its_reason(tmp_path: Path) -> None:
    """登记过的派生表 → `exempt`，且理由来自 `DERIVED_TABLES`，不是空话。
    A registered derived table is exempt, and its reason comes from `DERIVED_TABLES`.
    """
    directory = _aggregate_dir(tmp_path)
    (directory / "spearman.csv").write_text("task,arm\nx,a2\n", encoding="utf-8")

    rows = P.check_aggregates(["known"], directory)
    assert _status(rows, "aggregate_table_registered") == [P.STATUS_EXEMPT]
    assert P.DERIVED_TABLES["spearman.csv"] in next(
        row.detail for row in rows if row.check == "aggregate_table_registered"
    )


def test_a_dangling_run_id_in_an_aggregate_table_fails(tmp_path: Path) -> None:
    """表里引用了不存在的 run_id → `fail`，点名那个 id 与行号。
    A run id that does not resolve fails, naming the id and the row.
    """
    directory = _aggregate_dir(tmp_path)
    (directory / "arch_transfer.csv").write_text(
        "run_id,task\ntask__a2__f__s0,x\nmissing__a2__f__s1,y\n", encoding="utf-8"
    )

    rows = P.check_aggregates(["task__a2__f__s0"], directory)
    failures = [row for row in rows if row.status == P.STATUS_FAIL]
    assert len(failures) == 1
    assert failures[0].run_id == "missing__a2__f__s1"
    assert "arch_transfer.csv:3" in failures[0].detail


def test_multi_run_table_checks_every_named_column(tmp_path: Path) -> None:
    """一行引用两条 run 的表（`lambda_ci.csv`）：`unit_run` 与 `chain_run` 都要能反查。
    A row referencing two runs must resolve both columns.

    中文：这条是"新增表不许静默获得已对账身份"的另一半。`lambda_ci.csv` 首版没登记时，
          这条闸门当场红了（实测）；登记成"派生表豁免"也能让它变绿，但那样它 642 个引用
          一个都不核。**登记的正确方式是"多列反查"，不是"豁免"。**
    English:
        When the first version of `lambda_ci.csv` was unregistered the gate went red (measured);
        registering it as a derived table would also turn it green while checking none of its 642
        references. The right registration is multi-column resolution, not exemption.
    """
    directory = _aggregate_dir(tmp_path)
    (directory / "lambda_ci.csv").write_text(
        "task,arm,unit_run,chain_run\n"
        "t,a3_graph,u__a2__f__s0,c__a3__f__s0\n"
        "t,a3_graph,ghost__a2__f__s1,c__a3__f__s1\n",
        encoding="utf-8",
    )

    rows = P.check_aggregates(
        ["u__a2__f__s0", "c__a3__f__s0", "c__a3__f__s1"], directory
    )
    failures = [row for row in rows if row.status == P.STATUS_FAIL]
    assert len(failures) == 1
    assert failures[0].run_id == "ghost__a2__f__s1"
    assert "unit_run" in failures[0].detail
    assert not [row for row in rows if row.status == P.STATUS_EXEMPT]


def test_rf_rows_are_exempt_but_an_empty_xgboost_run_id_fails(tmp_path: Path) -> None:
    """`scaling_cpu.csv` 的 RF 行按 D-53 豁免；同一张表里 xgboost 行空 run_id 必须红。
    RF rows are exempt per D-53; an xgboost row with an empty run id must fail.

    中文：这条防的是"把忘了写混进豁免"——豁免按 `learner` 判定，不是一个笼统的"空就跳过"。
    English: the exemption keys on `learner`, not on "the cell is empty", so a forgotten id cannot
    hide inside it.
    """
    directory = _aggregate_dir(tmp_path)
    (directory / "scaling_cpu.csv").write_text(
        "task,learner,run_id\npb26_scaling_cp,rf,\npb26_scaling_cp,xgboost,\npb26_scaling_cp,xgboost,ok__a2__f__s0\n",
        encoding="utf-8",
    )

    rows = P.check_aggregates(["ok__a2__f__s0"], directory)
    statuses = _status(rows, "aggregate_run_id_resolves")
    assert statuses.count(P.STATUS_EXEMPT) == 1
    assert statuses.count(P.STATUS_FAIL) == 1
    # 能反查的那个 id 不许出现在失败里：豁免与失败都按行判，不按整表判。
    assert all(
        row.run_id != "ok__a2__f__s0" for row in rows if row.status == P.STATUS_FAIL
    )


# --------------------------------------------------------------------------- #
# ② 出图闸门：require_clean 必须抛
# --------------------------------------------------------------------------- #
def test_require_clean_raises_while_a_failure_remains(tmp_path: Path) -> None:
    """`require_clean()` 在仍有 `fail` 时抛 `ProvenanceError`，且**已经把表写出来了**。
    `require_clean()` raises while a failure remains - after writing the table.

    中文：顺序很重要：先落盘再抛，否则出错时连"哪里坏了"都没有凭据。
    English: write first, then raise; otherwise the failure leaves no evidence behind.
    """
    root = tmp_path
    raw = root / "results" / "raw_runs"
    raw.mkdir(parents=True, exist_ok=True)
    record = _record(root)  # 故意不写预测文件
    (raw / f"{record['run_id']}.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8"
    )
    table = _aggregate_dir(root) / "provenance_check.csv"

    with pytest.raises(P.ProvenanceError) as excinfo:
        P.require_clean(path=table, root=root)

    assert record["run_id"] in str(excinfo.value)
    assert table.is_file(), "抛异常前必须先落盘"
