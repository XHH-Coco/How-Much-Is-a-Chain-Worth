"""三基准下载与完整性核对（任务 A-2 的唯一入口）。
Download the three benchmarks and verify their integrity (the single entry point of task A-2).

中文：
    用途：把 PolyBench / PolyBench26 / OpenPoly 三个基准的原始文件从**发布方**取到
          `data/raw/`，逐个算 sha256 与条数，写出 `data/manifests/*_download_manifest.csv`。
          这是全队唯一被允许"把数据搬进本机"的代码；其他脚本只许读 `data/raw/`。
    依赖：无第三方包（只用标准库）。Python 3.10。
          `data/manifests/SOURCES.md` 是入口与许可的唯一权威——本文件的每个 URL 常量
          后面都写着它对应 SOURCES.md 的哪一节的哪一条，改 URL 必须先改那里。
    配置：无 configs/ 依赖（本任务在 P3 之前，`configs/` 还没填）。全部常量在本文件常量区。
    对外：`main()`（命令行入口）；`validate_polybench_manifest()` 供 `--verify-only` 与
          将来的 CI 复用。
    被谁用：A-3（标签调和吃 39 个 CSV）、A-5（划分快照吃 `scaling_indices/`）、A-6
          （结构表征吃 PSMILES 列）、A-7（OpenPoly 计数吃 741 行那张表）、C（scaling 档位）。

English:
    Purpose: fetch the three benchmarks from their publishers into `data/raw/`, hash and
          count every file, and write `data/manifests/*_download_manifest.csv`. This is the
          only code allowed to bring data onto a machine; every other script reads
          `data/raw/` only.
    Depends on: standard library only (no third-party packages), Python 3.10.
          `data/manifests/SOURCES.md` is the single authority for entries and licences;
          every URL constant below cites the section and bullet it comes from.
    Config: none. P3 has not started, so `configs/` is empty; all constants live here.
    Public API: `main()` (CLI) and `validate_polybench_manifest()` (reused by
          `--verify-only` and by future CI).
    Callers: A-3 (the 39 CSVs), A-5 (`scaling_indices/`), A-6 (the PSMILES column),
          A-7 (the 741-row OpenPoly table), C (official scaling tiers).

用法 / Usage::

    # 全都下（默认）——先看会下什么，不写盘
    python scripts/00_download.py --dry-run

    # 只下一个基准，只取前 3 个文件（调试网络用）
    python scripts/00_download.py --source polybench26 --limit 3

    # 只校验已有的 manifest（不联网）
    python scripts/00_download.py --verify-only --source polybench

    # 需要绕开 GitHub 未认证限流时
    GITHUB_TOKEN=ghp_xxx python scripts/00_download.py --source polybench26
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

# ---- 路径与常量 / Paths and constants ----

# 仓库根：本文件在 scripts/ 下，向上一层是根。所有路径都相对它，不许出现绝对路径。
# Repo root: this file lives in scripts/, so one level up. All paths are repo-relative.
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
MANIFEST_ROOT = REPO_ROOT / "data" / "manifests"

# ---- PolyBench（figshare 存档）/ PolyBench (figshare archive) ----

# figshare article id 与版本：来源 data/manifests/SOURCES.md §2.1「存档（已解析，v2）」。
# 存档 DOI 10.6084/m9.figshare.30917717，实际版本 v2，CC BY 4.0。
# Article id and version: SOURCES.md §2.1. DOI 10.6084/m9.figshare.30917717, version v2, CC BY 4.0.
POLYBENCH_ARTICLE_ID = 30917717
POLYBENCH_API = f"https://api.figshare.com/v2/articles/{POLYBENCH_ARTICLE_ID}"

# 39 个任务、42,169 条记录：来源 Proposal 附录 A（任务名与记录数逐字取自平台
# `/api/data-counts`，39 个数相加 = 42,169），并见 SOURCES.md §2.1「规模」。
# 39 tasks, 42,169 records: Proposal appendix A; also SOURCES.md §2.1 "scale".
POLYBENCH_EXPECTED_FILES = 39
POLYBENCH_EXPECTED_ROWS = 42169
POLYBENCH_VERSION = "figshare:v2"

# 39 个文件的权威清单：文件名 / 字节数 / 官方 md5 / 下载 URL。
# 来源是 figshare `/articles/30917717` 的原始响应，已存快照
# `data/manifests/polybench_api_article_20261112.json`（v2、CC BY 4.0）。
# 两条用途：①「手动下 / 别人代下 / 浏览器打包」这条路线唯一的验收标准——文件放进目录后核
# 这份 md5，核不上的就不是官方字节；② API 被拦时的兜底文件列表（URL 与 md5 表里都有，
# 不必先问 API 才能下）。
# The authoritative list of the 39 files (name / bytes / official md5 / download URL), captured
# from figshare's raw article response into the JSON snapshot above. It is (1) the only
# acceptance standard for any manual or third-party download route and (2) the offline fallback
# file list when the API is blocked.
POLYBENCH_EXPECTED_TABLE = MANIFEST_ROOT / "polybench_expected_files.csv"
POLYBENCH_ARCHIVE_BYTES = 5_937_499   # 39 份合计；等于 figshare article 的 `size` 字段


# 附录 A 的 39 行：任务名 -> 记录数。**这是校验用的副本，不是数据来源**；
# 数据来源永远是 figshare 的 CSV 本身。名字必须逐字一致（含大小写与空格）。
# The 39 rows of appendix A as name -> record count. A verification copy, not a data source;
# names must match character for character.
POLYBENCH_APPENDIX_A: dict[str, int] = {
    "Glass transition temperature": 8055,
    "Thermal decomposition temperature": 4898,
    "Melting temperature": 3796,
    "bandgap chain_Egc": 3380,
    "Intrinsic viscosity": 1936,
    "Density": 1699,
    "Specific volume": 1699,
    "PCE_OPV_aug": 1203,
    "Elongation at break": 1113,
    "Tensile stress strength at break": 1034,
    "Tensile modulus": 1034,
    "Electric conductivity": 982,
    "Lc phase transition temperature": 941,
    "Volume resistivity": 934,
    "Softening temperature": 763,
    "Gas permeability coefficient": 711,
    "Water absorption": 680,
    "Heat of fusion": 588,
    "bandgap bulk_Egb": 561,
    "Crystallization temperature": 456,
    "Gas diffusion coefficient": 433,
    "Dynamic mechanical properties storage modulus": 387,
    "Dielectric Constant_EPS": 382,
    "Refractive Index_Nc": 382,
    "Ionization Energy_Ei": 370,
    "Electron Affinity_Eea": 368,
    "Surface tension": 346,
    "Solubility parameter": 316,
    "Cohesive energy density": 307,
    "Dynamic mechanical properties loss tangent": 300,
    "Dielectric loss factor": 279,
    "Contact angle": 255,
    "Tensile stress strength at yield": 253,
    "Gas solubility coefficient": 249,
    "Dielectric loss tangent": 238,
    "Isothermal weight loss time": 228,
    "Heat of fusion mol conversion": 213,
    "Specific heat capacity": 207,
    "Dynamic mechanical properties loss modulus": 193,
}

# ---- PolyBench26（GitHub，承重）/ PolyBench26 (GitHub, load-bearing) ----

# 仓库与固定 commit：来源 SOURCES.md §2.2「入口」「version 取值」。
# 2026-11-12 实测 main 的 HEAD 就是这个 sha；**改 commit 必须记 docs/DECISIONS.md**，
# 因为索引是行号、行号只在这个版本下指向正确分子（SOURCES.md §2.2【坑五】）。
# Repo and pinned commit: SOURCES.md §2.2. This was main's HEAD on 2026-11-12; changing it
# requires a DECISIONS entry, because the indices are row numbers tied to this version.
POLYBENCH26_REPO = "rlearsch/PolymerBenchmark2026"
POLYBENCH26_COMMIT = "83655bc25901a8cabe3a1e5474b7ec26f01b919f"

# 要取的三类东西，以及为什么。来源 SOURCES.md §2.2「磁盘（实测量级）」：
# 我们只需要 < 200 MB 的这三样，不要官方 README 里那 ~15 GB 的完整复现口径。
# Three kinds of payloads and why. SOURCES.md §2.2: we need only these (< 200 MB), not the
# ~15 GB "full reproduction" footprint the official README quotes.
POLYBENCH26_WANTED_FILES: tuple[str, ...] = (
    # canonical 源：82,990 条结构记录（交替 65,221 / 无规 10,379 / 均聚 7,390）。
    # canonical source: 82,990 structure records.
    "Datasets/Dataset_construction_scripts/files/Cleaned_OMersBench_v3_final.jsonl",
    # PolyMetriX Tg 的网页源：7,367 条 PSMILES，主表第 7 行与 H2 的锚点。
    # PolyMetriX Tg web source: 7,367 PSMILES; main-table row 7 and the H2 anchor.
    "Datasets/PSMILES/PolyMetriX/Tg/Tg.csv",
)
# 目录前缀：官方入库的紧凑划分索引（整数行号 + PSMILES sha256）。A-5 只快照其中两个
# 实验的 seed_42/，但 C 的 scaling 实验要用 MD_5000 家族，所以整个目录一起取（约 2 MB）。
# Directory prefix: the official tracked compact split indices. A-5 snapshots only two
# experiments, but C needs the MD_5000 family, so take the whole directory (~2 MB).
POLYBENCH26_WANTED_PREFIXES: tuple[str, ...] = ("Datasets/scaling_indices/",)

# 官方公布的 sha256：来源 SOURCES.md §2.2「上游源文件 sha256（官方公布，A-2 可直接核对）」。
# 这是本脚本最硬的一条校验：下载下来的字节必须与官方写进文档的那一串相等。
# Officially published sha256: SOURCES.md §2.2. The hardest check in this script.
POLYBENCH26_EXPECTED_SHA256: dict[str, str] = {
    "Datasets/Dataset_construction_scripts/files/Cleaned_OMersBench_v3_final.jsonl": (
        "7e8a175866cc7166e589fbca1bb125bdece31c398b654a8848f5410f2c68567e"
    ),
    "Datasets/PSMILES/PolyMetriX/Tg/Tg.csv": (
        "b6d2b9f83419c94c5b24200b0288d9a7b53256931d87ffc1533c8884119a09d3"
    ),
}

# 官方公布的记录数：来源 SOURCES.md §2.2 同一条（82,990 条；7,367 条 PSMILES）。
# Published record counts: same bullet in SOURCES.md §2.2.
POLYBENCH26_EXPECTED_ROWS: dict[str, int] = {
    "Datasets/Dataset_construction_scripts/files/Cleaned_OMersBench_v3_final.jsonl": 82990,
    "Datasets/PSMILES/PolyMetriX/Tg/Tg.csv": 7367,
}

# ---- OpenPoly（GitHub，仅边界）/ OpenPoly (GitHub, boundary only) ----

# 仓库与 commit：来源 SOURCES.md §2.3「入口」「version 取值」。2026-11-12 实测 HEAD。
# Repo and commit: SOURCES.md §2.3. Observed HEAD on 2026-11-12.
OPENPOLY_REPO = "WangGroupFDU/Openpoly_benchmark"
OPENPOLY_COMMIT = "ccaa66900747bfc576ef935e3d5f220b6c9b1f85"

# 741 行 / 741 个不同 PSMILES / 3,985 条标注：来源 SOURCES.md §2.3「规模」与 Proposal §2.3。
# 741 rows / 741 distinct PSMILES / 3,985 annotations: SOURCES.md §2.3, Proposal §2.3.
# 只取这一张表：同目录的 `final_polymer_property_counts_fromliterature.csv` 报 Tg 58,755，
# 是匹配结构之前的文献计数，SOURCES.md §2.3【坑二】明令不许用，所以**故意不下载**。
# Only this table. The sibling counts CSV reports Tg 58,755 (pre-matching literature counts)
# and is explicitly banned by SOURCES.md §2.3 pitfall #2, so it is deliberately NOT fetched.
OPENPOLY_WANTED_FILES: tuple[str, ...] = (
    "data/final_polymer_properties_fromliterature.csv",
)
OPENPOLY_EXPECTED_ROWS: dict[str, int] = {
    "data/final_polymer_properties_fromliterature.csv": 741,
}

# ---- 清单列名 / Manifest columns ----

# 前六列是分工_A A-2「产出」一栏写死的，顺序不得改；后两列是本脚本追加的说明列，
# 原因写在 docs/交付_P1_A2_下载与完整性.md §2：`bytes` 给"文件是否被截断"一个直接证据，
# `n_rows_basis` 是 SOURCES.md §2.2 要求的"注明是哪一种条数"。
# The first six columns are fixed by 分工_A A-2. The last two are additive and justified in
# docs/交付_P1_A2_下载与完整性.md §2.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "task_id",
    "n_rows",
    "sha256",
    "source_url",
    "version",
    "downloaded_at",
    "bytes",
    "n_rows_basis",
)

# 网络参数 / Network parameters
USER_AGENT = "chainworth-downloader/1.0 (+https://github.com/XHH-Coco/How-Much-Is-a-Chain-Worth)"
HTTP_TIMEOUT_S = 120         # 单次请求超时。GitHub 与 figshare 实测都在 30 s 内返回 / per-request timeout
HTTP_RETRIES = 3             # 三条路径的额外重试轮数 / extra rotation rounds over the 3 proxy paths
BACKOFF_BASE_S = 2.0         # 退避基数：第 k 次重试等 2^k 秒 / backoff base, waits 2**k seconds

# 代理策略的尝试顺序，以及"哪些环境变量算代理配置"。
# 顺序有理由：先尊重环境（队友把代理配在环境里是常态），再退到系统设置，最后才裸连。
# 见 `_http_get` 的 docstring：某些本地代理会 MITM TLS 而根证书不全，裸连反而能过。
# Proxy strategies in order, plus which env vars count as proxy config. Rationale in `_http_get`.
_PROXY_STRATEGIES: tuple[str, ...] = ("env", "system", "none")
_PROXY_ENV_KEYS: tuple[str, ...] = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
)

LOGGER = logging.getLogger("chainworth.download")

# 本进程的 TLS 信任配置。None = 用 Python 自带根证书（默认，也是最安全的默认值）。
# 由 `main()` 按 `--ca-bundle` 设置；`_http_get` 只读不改。
# This process's TLS trust config. None = Python's bundled roots (the safe default).
# Set by `main()` from `--ca-bundle`; `_http_get` only reads it.
_SSL_CONTEXT: ssl.SSLContext | None = None


# ---- 数据结构 / Data structures ----


@dataclass
class DownloadRow:
    """清单的一行，对应磁盘上一个已落盘并校验过的文件。
    One manifest row, i.e. one file on disk that has been written and verified.

    中文：
        用途：把"一个文件的身份"打包，供写 CSV 与打印摘要共用，避免两处各拼一次字段。
        字段含义见 `MANIFEST_COLUMNS`；`n_rows` 对非表格文件为空串。
    English:
        Purpose: bundle one file's identity so manifest writing and console summaries share
        one shape. Field meanings are `MANIFEST_COLUMNS`; `n_rows` is empty for non-tabular files.
    """

    task_id: str
    n_rows: int | str
    sha256: str
    source_url: str
    version: str
    downloaded_at: str
    bytes: int
    n_rows_basis: str

    def as_dict(self) -> dict[str, object]:
        """转成写 CSV 用的有序字典。
        Convert to an ordered dict for the CSV writer.

        中文：
            用途：保证列顺序与 `MANIFEST_COLUMNS` 完全一致，列序一乱别人就对不上表头。
            输入：无。
            输出：dict，键即列名，顺序即 `MANIFEST_COLUMNS`。
            数据来源：本对象的字段。
            配置来源：无。
            副作用：无。
            异常：无。

        English:
            Purpose: keep column order identical to `MANIFEST_COLUMNS`.
            Args: none.
            Returns: dict keyed by column name, in manifest order.
            Data source: this object's fields.
            Config: none.
            Side effects: none.
            Raises: none.
        """
        return {
            "task_id": self.task_id,
            "n_rows": self.n_rows,
            "sha256": self.sha256,
            "source_url": self.source_url,
            "version": self.version,
            "downloaded_at": self.downloaded_at,
            "bytes": self.bytes,
            "n_rows_basis": self.n_rows_basis,
        }


def build_ssl_context(spec: str) -> ssl.SSLContext:
    """按 `--ca-bundle` 的取值构造 TLS 上下文；**不提供"关闭校验"这条路**。
    Build a TLS context from `--ca-bundle`; deliberately offers no "disable verification" path.

    中文：
        用途：企业网与某些开发工具会用本地代理中间人解密 TLS，它的根证书不在 Python 自带
              根证书里，于是报 `CERTIFICATE_VERIFY_FAILED: unable to get issuer certificate`。
              正确解法是**把那个根证书加进来**，而不是把校验关掉。
        输入：spec —— 两种取值：
              · 一个 .pem 文件路径（把公司/代理的根证书放进去）；
              · 字符串 `windows`：从 Windows 的 ROOT + CA 证书库现取。这条路在企业 Windows
                上是标准做法——证书库就是操作系统信任的那些锚点，不是"随便信任"。
        输出：`ssl.SSLContext`，`verify_mode` 恒为 `CERT_REQUIRED`、`check_hostname` 恒为 True。
        数据来源：调用方给的 PEM，或本机 Windows 证书库（`ssl.enum_certificates`）。
        配置来源：命令行 `--ca-bundle` 或环境变量 `CHAINWORTH_CA_BUNDLE`。
        副作用：读证书文件 / 读证书库；不写盘。
        异常：FileNotFoundError —— PEM 路径不存在；ssl.SSLError —— PEM 里没有可用证书；
              RuntimeError —— 在非 Windows 上用了 `windows`。

    English:
        Purpose: corporate networks and some dev tools MITM TLS through a local proxy whose
                 root is absent from Python's bundle, yielding `unable to get issuer
                 certificate`. The fix is to *add that root*, never to disable verification.
        Args: spec - either a path to a .pem file, or the literal `windows` to read the
              Windows ROOT + CA stores (the OS trust anchors, the standard enterprise route).
        Returns: an `SSLContext` with `verify_mode = CERT_REQUIRED` and `check_hostname = True`.
        Data source: the caller's PEM, or the local Windows certificate stores.
        Config: the `--ca-bundle` flag or `CHAINWORTH_CA_BUNDLE`.
        Side effects: reads certificates; writes nothing.
        Raises: FileNotFoundError for a missing PEM; ssl.SSLError when the PEM has no usable
                certificate; RuntimeError when `windows` is used off Windows.
    """
    if spec.lower() == "windows":
        if sys.platform != "win32":
            raise RuntimeError("`--ca-bundle windows` 只在 Windows 上有意义 / Windows-only option")
        context = ssl.create_default_context()
        loaded = 0
        for store in ("ROOT", "CA"):
            for der, encoding, _trust in ssl.enum_certificates(store):
                if encoding != "x509_asn":
                    continue
                context.load_verify_locations(cadata=der)   # cadata 接受 DER 字节 / accepts DER bytes
                loaded += 1
        if not loaded:
            raise ssl.SSLError("Windows 证书库里没读到任何证书 / no certificate found in the OS stores")
        LOGGER.info("已从 Windows 证书库载入 %d 张证书 / loaded %d OS certificates", loaded, loaded)
        return context

    path = Path(spec).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"--ca-bundle 指向的文件不存在 / no such CA bundle: {path}")
    context = ssl.create_default_context(cafile=str(path))
    LOGGER.info("已用自定义根证书 / using custom CA bundle: %s", path)
    return context


# ---- 网络小工具 / Network helpers ----


def _now_iso() -> str:
    """取带时区的本地时间戳，供 manifest 的 `downloaded_at` 列使用。
    Return a timezone-aware local timestamp for the `downloaded_at` column.

    中文：
        用途：让每一行都能回答"这份文件是哪一分钟落到这台机器上的"。
        输入：无。
        输出：ISO 8601 字符串，精确到秒，含 UTC 偏移（如 `2026-11-12T02:41:07+08:00`）。
        数据来源：系统时钟。
        配置来源：无。
        副作用：无。
        异常：无。

    English:
        Purpose: let every row answer "when did this land on this machine".
        Args: none.
        Returns: ISO 8601 string, second precision, with UTC offset.
        Data source: system clock.
        Config: none.
        Side effects: none.
        Raises: none.
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _http_get(url: str, *, what: str, accept: str | None = None) -> bytes:
    """带重试的 GET，返回响应体原始字节。
    GET with retries; return the raw response body.

    中文：
        用途：本脚本所有网络访问的唯一出口；**不做任何解码**，保证落盘字节就是发布方字节。
        输入：url —— 完整 URL，必须来自本文件常量区（常量又注明 SOURCES.md 的出处）；
              what —— 人读的短说明，只用于日志与异常信息；
              accept —— 可选的 Accept 头（本脚本目前不用，保留给将来的 contents API 分支）。
        输出：bytes，响应体原文。
        数据来源：远端（figshare / GitHub codeload）。
        配置来源：环境变量 `GITHUB_TOKEN`（有则带上，把 60 次/时的未认证限流抬到 5000）；
              模块级 `_SSL_CONTEXT`（由 `--ca-bundle` 设置，用来补企业代理的根证书）。
        副作用：发网络请求；无写盘。
        异常：urllib.error.HTTPError —— 4xx（除 429）立即原样抛出，调用方据此判断 403/404；
              RuntimeError —— 全部轮次与三条路径都失败后包装抛出。

        三条代理策略（按序尝试，这是本机实测出来的，不是过度设计）：
          1. `env`    —— 尊重 `HTTP(S)_PROXY` 环境变量，机器上配了什么就用什么；
          2. `system` —— 临时藏起环境变量，只用系统级代理设置（Windows 注册表）；
          3. `none`   —— 一个代理都不走，裸连。
        为什么要三条：某些开发工具注入的本地代理会**中间人解密 TLS**，而它的根证书不在本机
        OpenSSL 信任链里，于是报 `CERTIFICATE_VERIFY_FAILED: unable to get issuer certificate`。
        这不是证书错了，是"信任链没配全"。**全程不关校验**（`verify_mode` 始终 CERT_REQUIRED），
        只换通道；换个通道能过就过，过不了就报错，绝不用"不校验"把问题糊过去。

    English:
        Purpose: the single network exit of this script; never decodes, so the bytes on disk
                 are the publisher's bytes.
        Args: url - full URL, always from the constants block above; what - short human label
              for logs and errors; accept - optional Accept header (unused today; kept for a
              future contents-API branch).
        Returns: raw response body as bytes.
        Data source: remote (figshare / GitHub codeload).
        Config: env var `GITHUB_TOKEN`, used when present to lift GitHub's 60/h limit.
        Side effects: performs a network request; writes nothing.
        Raises: urllib.error.HTTPError immediately for 4xx other than 429 (callers branch on
                403/404); RuntimeError once both proxy strategies are exhausted.

        Three proxy strategies, tried in order (measured on this machine, not speculation):
          1. `env`    - honour the `HTTP(S)_PROXY` environment variables;
          2. `system` - hide those variables and use the OS-level proxy settings only;
          3. `none`   - one straight connection, no proxy at all.
        Why more than one: some developer tools inject a local proxy that MITMs TLS and whose
        root CA is absent from this machine's OpenSSL trust store, producing
        `CERTIFICATE_VERIFY_FAILED: unable to get issuer certificate`. The certificate is not
        wrong, the trust chain is incomplete. Falling back keeps `verify_mode` at
        CERT_REQUIRED throughout - we never disable verification to make a download succeed.
    """
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if accept is not None:
        headers["Accept"] = accept
    token = os.environ.get("GITHUB_TOKEN")
    if token and urllib.parse.urlparse(url).hostname == "api.github.com":
        headers["Authorization"] = f"Bearer {token}"

    # 三种代理策略轮流试，最多走 `HTTP_RETRIES + 1` 轮。
    # 轮次在外、策略在内：一轮里三条路各走一次，比"一条路死磕 4 次"更快找到能通的那条。
    # Strategies rotate inside each round, rounds outside: trying all three once beats
    # hammering one route four times.
    last_error: Exception | None = None
    for round_index in range(HTTP_RETRIES + 1):
        if round_index:
            wait_s = BACKOFF_BASE_S * (2 ** (round_index - 1))
            LOGGER.warning("第 %d/%d 轮重试（等 %.0fs）: %s", round_index, HTTP_RETRIES, wait_s, what)
            time.sleep(wait_s)
        for strategy in _PROXY_STRATEGIES:
            try:
                return _request_once(url, headers=headers, strategy=strategy)
            except urllib.error.HTTPError as exc:
                last_error = exc
                # 4xx 除 429 外重试无意义：403 是权限/地区拦截，404 是入口写错。
                # 换网络路径也救不了 403/404，直接抛给调用方判断。
                # Retrying or re-routing 403/404 is pointless; let the caller branch on it.
                if exc.code not in (429,) and 400 <= exc.code < 500:
                    raise
                LOGGER.warning("HTTP %d（策略 %s）: %s", exc.code, strategy, what)
            except (urllib.error.URLError, OSError) as exc:
                last_error = exc
                LOGGER.debug("连接失败（策略 %s）: %s（%s）", strategy, what, exc)
    raise RuntimeError(
        f"下载失败（env / system / none 三条路径各试了 {HTTP_RETRIES + 1} 轮）: {what}"
    ) from last_error


def _request_once(url: str, *, headers: dict[str, str], strategy: str) -> bytes:
    """按指定代理策略发一次 GET，返回响应体。
    Issue a single GET using the named proxy strategy and return the body.

    中文：
        用途：把"换一条网络路径"这件事隔离成一次尝试，让 `_http_get` 的轮询逻辑保持可读。
        输入：url —— 完整 URL；headers —— 已构造好的请求头（含 UA / Accept / token）；
              strategy —— `env`（尊重环境变量）/ `system`（只用系统设置）/ `none`（不用代理）。
        输出：bytes，响应体原文。
        数据来源：远端。
        配置来源：模块级 `_SSL_CONTEXT`（非 None 时用于补根证书）。
        副作用：发网络请求；`system` 策略会临时弹出再还原代理环境变量（异常安全）。
        异常：urllib.error.HTTPError —— 服务端返回 4xx/5xx；
              urllib.error.URLError —— 连接或 TLS 失败；ssl.SSLError 是其常见原因。

    English:
        Purpose: isolate "try one network path" into a single attempt so `_http_get`'s
                 rotation logic stays readable.
        Args: url - full URL; headers - prepared request headers (UA / Accept / token);
              strategy - `env`, `system` or `none`.
        Returns: raw response body.
        Data source: remote.
        Config: the module-level `_SSL_CONTEXT`, used when not None to add trust anchors.
        Side effects: performs a request; the `system` strategy temporarily removes and then
                      restores proxy env vars (exception-safe).
        Raises: urllib.error.HTTPError for 4xx/5xx; urllib.error.URLError for connection or
                TLS failures (ssl.SSLError being the common cause).
    """
    saved_env: dict[str, str] = {}
    if strategy == "system":
        for key in _PROXY_ENV_KEYS:
            if key in os.environ:
                saved_env[key] = os.environ.pop(key)
        proxy_handler: urllib.request.BaseHandler = urllib.request.ProxyHandler()
    elif strategy == "none":
        proxy_handler = urllib.request.ProxyHandler({})   # 空 dict = 一个代理都不用 / no proxy at all
    else:
        proxy_handler = urllib.request.ProxyHandler()     # 环境变量优先 / env vars take precedence

    try:
        handlers: list[urllib.request.BaseHandler] = [proxy_handler]
        if _SSL_CONTEXT is not None:
            # 自定义信任链（企业代理/公司根证书），见 `build_ssl_context`。
            # Custom trust anchors (corporate roots), see `build_ssl_context`.
            handlers.append(urllib.request.HTTPSHandler(context=_SSL_CONTEXT))
        opener = urllib.request.build_opener(*handlers)
        request = urllib.request.Request(url, headers=headers, method="GET")
        with opener.open(request, timeout=HTTP_TIMEOUT_S) as response:
            return response.read()
    finally:
        os.environ.update(saved_env)                      # 一定还原，别把环境改脏 / always restore


def _http_json(url: str, *, what: str) -> dict:
    """GET 一个 JSON 端点并解析。
    GET a JSON endpoint and parse it.

    中文：
        用途：figshare 的 article 元数据与 GitHub 的 tree/contents 都是 JSON，统一在这里解析。
        输入：url —— 见 `_http_get`；what —— 短说明。
        输出：dict，解析后的 JSON 对象。
        数据来源：远端。
        配置来源：无。
        副作用：发网络请求。
        异常：ValueError —— 响应不是 JSON（常见于被网关拦截后返回的 HTML 错误页）。

    English:
        Purpose: figshare article metadata and GitHub tree/contents are JSON; parse them here.
        Args: url, what - see `_http_get`.
        Returns: parsed JSON object.
        Data source: remote.
        Config: none.
        Side effects: network request.
        Raises: ValueError when the body is not JSON (typically an HTML error page).
    """
    payload = _http_get(url, what=what)
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        head = payload[:120].decode("utf-8", "replace")
        raise ValueError(f"{what} 返回的不是 JSON，前 120 字节：{head!r}") from exc


# ---- 哈希与条数 / Hashing and row counting ----


def _digest(data: bytes, algorithm: str) -> str:
    """算摘要的十六进制串，`algorithm` 取 `sha256` 或 `md5`。
    Hex digest of `data`; `algorithm` is `sha256` or `md5`.

    中文：
        用途：figshare 公布的是 md5，我们自己的清单用 sha256，两套都要算，故抽成一个函数。
        输入：data —— 文件原始字节（内存中，最大约 67 MB，可接受）；algorithm —— 摘要名。
        输出：小写十六进制字符串（sha256 64 位、md5 32 位）。
        数据来源：传入的字节。
        配置来源：无。
        副作用：无。
        异常：ValueError —— algorithm 不在 hashlib.algorithms_available 里。

    English:
        Purpose: figshare publishes md5 while our manifest uses sha256, so both are needed.
        Args: data - raw bytes (at most ~67 MB in memory, acceptable); algorithm - digest name.
        Returns: lowercase hex string.
        Data source: the bytes passed in.
        Config: none.
        Side effects: none.
        Raises: ValueError for an unknown algorithm.
    """
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"不支持的摘要算法: {algorithm}")
    return hashlib.new(algorithm, data).hexdigest()


def _count_csv_rows(path: Path) -> int:
    """数 CSV 的数据行数（不含表头），按 RFC 4180 解析，不用"数换行"。
    Count CSV data rows (header excluded) by parsing, not by counting newlines.

    中文：
        用途：PolyBench 的 n 是"记录数"，单元格里可能有带换行的引号文本；数换行会多算，
              这正是 42,169 这个合计对不上的最常见原因。
        输入：path —— 已落盘的 CSV 路径。
        输出：int，数据行数（0 表示只有表头）。
        数据来源：`data/raw/polybench/*.csv`（原样下载，未清洗）。
        配置来源：无。
        副作用：只读文件。
        异常：UnicodeDecodeError —— 文件不是 UTF-8。PolyBench 39 个文件实测全为 UTF-8；
              若真有非 UTF-8 文件，宁可报错也不许猜编码（分工_A A-3 红线「不许猜」）。

    English:
        Purpose: PolyBench's n is a record count and cells may contain quoted newlines;
                 counting newlines over-counts, which is the usual reason 42,169 fails.
        Args: path - a CSV already on disk.
        Returns: number of data rows (0 means header only).
        Data source: data/raw/polybench/*.csv (downloaded verbatim, not cleaned).
        Config: none.
        Side effects: reads the file only.
        Raises: UnicodeDecodeError if the file is not UTF-8; guessing an encoding is banned.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # 跳过表头 / skip the header row
        return sum(1 for _ in reader)


def _count_jsonl_records(path: Path) -> int:
    """数 JSONL 的非空行数，即记录数。
    Count non-empty lines of a JSONL file, i.e. records.

    中文：
        用途：`Cleaned_OMersBench_v3_final.jsonl` 一行一条结构记录，官方给的 82,990
              就是行数；逐行 json.loads 会慢十倍且不增加信息，所以只数非空行。
        输入：path —— 已落盘的 .jsonl 路径。
        输出：int，非空行数。
        数据来源：`data/raw/polybench26/Datasets/Dataset_construction_scripts/files/`。
        配置来源：无。
        副作用：只读文件。
        异常：UnicodeDecodeError —— 文件不是 UTF-8。

    English:
        Purpose: one record per line; the official 82,990 is a line count. Parsing each line
                 as JSON is ten times slower and adds no information.
        Args: path - a .jsonl already on disk.
        Returns: number of non-empty lines.
        Data source: data/raw/polybench26/.../Cleaned_OMersBench_v3_final.jsonl.
        Config: none.
        Side effects: reads the file only.
        Raises: UnicodeDecodeError if the file is not UTF-8.
    """
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


# ---- 写盘与清单 / Writing to disk and to the manifest ----


def _atomic_write(path: Path, data: bytes) -> None:
    """先写 `.part` 再改名，避免中断时留下半个文件被下次运行当成"已下载"。
    Write to `.part` then rename, so an interrupted run never leaves a half file that the
    next run would mistake for a finished download.

    中文：
        用途：断点续传的"续"其实靠的是幂等——要么整份在，要么整份不在；不许有中间态。
        输入：path —— 目标路径；data —— 完整文件字节。
        输出：无。
        数据来源：远端响应体。
        配置来源：无。
        副作用：建目录、写文件、替换已存在的同名文件。
        异常：OSError —— 磁盘满或无写权限。

    English:
        Purpose: resumability here means idempotence: either the whole file is there or none
                 of it is; no intermediate state is allowed.
        Args: path - destination; data - full file bytes.
        Returns: none.
        Data source: remote response body.
        Config: none.
        Side effects: creates parent dirs, writes the file, replaces any same-named file.
        Raises: OSError on a full disk or missing write permission.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_bytes(data)
    os.replace(temp, path)


def manifest_path(label: str, limit: int | None) -> Path:
    """给出某个基准的清单路径；**限量跑时自动加 `.partial`**，不覆盖完整清单。
    Return the manifest path for a benchmark; a `--limit` run gets `.partial` so it can
    never clobber the complete one.

    中文：
        用途：`--limit` 是为了调网络，不是为了产出证据。若限量跑直接写
              `polybench_download_manifest.csv`，一份只有 3 行的清单会静静取代 39 行那份，
              而下游（A-3、D 的审计）无从分辨——这正是分工_A §7 红线 5「手改 manifest
              等于伪造」想拦的那类事故，只不过这次是脚本自己动的手。
        输入：label —— 基准名（`polybench` / `polybench26` / `openpoly`）；
              limit —— 本次的 `--limit` 值，None 表示完整跑。
        输出：Path。完整跑为 `data/manifests/<label>_download_manifest.csv`；
              限量跑为 `data/manifests/<label>_download_manifest.partial.csv`。
        数据来源：无。
        配置来源：命令行 `--limit`。
        副作用：无。
        异常：无。

    English:
        Purpose: `--limit` exists to debug the network, not to produce evidence. Letting a
                 three-row run write the real manifest would silently replace a 39-row one and
                 no downstream consumer could tell — exactly the "hand-edited manifest" hazard
                 of 分工_A §7 red line 5, only this time committed by the script itself.
        Args: label - benchmark name; limit - this run's `--limit`, None for a full run.
        Returns: the full path, or the `.partial` path for a limited run.
        Data source: none.
        Config: the `--limit` CLI flag.
        Side effects: none.
        Raises: none.
    """
    if limit is not None:
        return MANIFEST_ROOT / f"{label}_download_manifest.partial.csv"
    return MANIFEST_ROOT / f"{label}_download_manifest.csv"


def write_manifest(rows: Sequence[DownloadRow], out_path: Path) -> None:
    """把清单写成 CSV，列序固定为 `MANIFEST_COLUMNS`。
    Write the manifest CSV with columns fixed to `MANIFEST_COLUMNS`.

    中文：
        用途：manifest 是"别人重建"的唯一线索，所以只能由脚本重生成，不许手改
              （分工_A §7 红线 5：手改 manifest 等于伪造）。
        输入：rows —— 本次下载并校验通过的行；out_path —— 目标 CSV 路径。
        输出：无（写盘）。
        数据来源：`DownloadRow`，其字段来自刚落盘的字节。
        配置来源：无。
        副作用：覆盖 `out_path`；父目录不存在则创建。
        异常：OSError —— 无写权限。
        注意：空 rows 也会写出只有表头的文件——那是"跑过但一个都没成"的诚实记录，
              比悄悄不写文件好。

    English:
        Purpose: the manifest is the only clue a teammate has to rebuild; it must be produced
                 by a script and never hand-edited (分工_A §7 red line 5).
        Args: rows - verified rows from this run; out_path - destination CSV.
        Returns: none (writes to disk).
        Data source: `DownloadRow` instances backed by bytes already on disk.
        Config: none.
        Side effects: overwrites `out_path`, creating parent dirs as needed.
        Raises: OSError without write permission.
        Note: empty rows still produce a header-only file - an honest "ran but got nothing".
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        # `lineterminator="\n"`：`.gitattributes` 把 `*.csv` 定为 `eol=lf`，而 csv 模块默认写
        # `\r\n`。不显式指定的话，每次在本机重跑都会把工作区文件改回 CRLF，git 每次都提示
        # "CRLF will be replaced"，看起来像清单被谁动过。
        # `lineterminator="\n"` matches the repo's `*.csv text eol=lf` rule; the csv module
        # defaults to `\r\n`, which makes every local re-run look like a hand edit.
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


# ---- PolyBench / PolyBench ----


def load_polybench_expected_files() -> list[dict[str, object]]:
    """读出 39 个文件的权威清单：文件名、字节数、官方 md5、下载 URL。
    Load the authoritative list of the 39 files: name, bytes, official md5, download URL.

    中文：
        用途：手动下载路线只有一条验收标准——md5 与官方存档一致。这张表就是那份标准，
              顺带充当 API 被拦时的文件列表（不用先问 API 才知道要下哪 39 个文件）。
        输入：无（读 `data/manifests/polybench_expected_files.csv`）。
        输出：按文件名排序的 dict 列表，键为 name / task_id / size / md5 / download_url / file_id。
        数据来源：`data/manifests/polybench_api_article_20261112.json`，即 figshare
                  `/articles/30917717` 原始响应的快照（v2、CC BY 4.0）。
        配置来源：无。
        副作用：只读。
        异常：FileNotFoundError —— 表不在（`--verify-only` 据此报"还没有标准可依"）；
              ValueError —— 行数不是 39、文件名重复、或字节合计不等于存档声明的 5,937,499
              （说明表被改坏了，宁可拦住也不能拿它去核别人下的文件）。

    English:
        Purpose: any manual route has exactly one acceptance standard — md5 equality with the
                 official archive. This table is that standard, and doubles as the file list
                 when the API is blocked.
        Args: none (reads data/manifests/polybench_expected_files.csv).
        Returns: dicts sorted by file name with keys name / task_id / size / md5 /
                 download_url / file_id.
        Data source: the JSON snapshot of figshare's raw article response (v2, CC BY 4.0).
        Config: none.
        Side effects: read-only.
        Raises: FileNotFoundError when the table is missing; ValueError when it has the wrong
                row count, duplicate names, or a byte total other than 5,937,499 — a corrupted
                table must not be used to certify someone else's download.
    """
    if not POLYBENCH_EXPECTED_TABLE.exists():
        raise FileNotFoundError(
            f"没有 39 个文件的标准表 / expected-file table missing: {POLYBENCH_EXPECTED_TABLE}"
        )
    with POLYBENCH_EXPECTED_TABLE.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))

    if len(records) != POLYBENCH_EXPECTED_FILES:
        raise ValueError(
            f"标准表 {len(records)} 行，应为 {POLYBENCH_EXPECTED_FILES} 行 / "
            f"expected-file table has {len(records)} rows, expected {POLYBENCH_EXPECTED_FILES}"
        )
    names = [str(record["file_name"]) for record in records]
    if len(set(names)) != len(names):
        raise ValueError("标准表里有重复文件名 / duplicate file names in the expected-file table")
    total = sum(int(str(record["size_bytes"])) for record in records)
    if total != POLYBENCH_ARCHIVE_BYTES:
        raise ValueError(
            f"标准表字节合计 {total:,}，存档声明 {POLYBENCH_ARCHIVE_BYTES:,} / byte total mismatch"
        )

    return [
        {
            "name": str(record["file_name"]),
            "task_id": str(record["task_id"]),
            "size": int(str(record["size_bytes"])),
            "md5": str(record["md5"]),
            "download_url": str(record["download_url"]),
            "file_id": int(str(record["figshare_file_id"])),
        }
        for record in sorted(records, key=lambda record: str(record["file_name"]).lower())
    ]


def _polybench_files() -> tuple[list[dict[str, object]], str]:
    """确定 PolyBench 要下载的文件列表：先问 API，被拦就退回冻结表，两者都在时互相核对。
    Resolve the PolyBench file list: ask the API, fall back to the frozen table when blocked,
    and cross-check the two whenever both are available.

    中文：
        用途：把"文件列表从哪来"和"没网也能下"分开。API 通的时候它是最新事实来源，同时
              与冻结表逐名逐 md5 对齐——对不上说明上游发了新版本，必须停下来记 DECISIONS，
              不能拿旧 md5 去核新文件。API 被拦时退回冻结表，下载 URL 与 md5 都在表里，
              `ndownloader.figshare.com` 若可达仍然能下。
        输入：无。
        输出：(files, version)：files 为 dict 列表（name / size / md5 / download_url / file_id），
              version 形如 `figshare:v2`。
        数据来源：figshare `/articles/30917717`；退化路径为 `polybench_expected_files.csv`。
        配置来源：无。
        副作用：发一次网络请求（退化路径不发）。
        异常：ValueError —— 文件数不是 39，或 API 与冻结表的文件名 / md5 对不上；
              RuntimeError —— API 与非 403 的网络错误（此时不该静默退化）。

    English:
        Purpose: separate "where the file list comes from" from "download without the API".
                 A reachable API is the current source of truth and is cross-checked name by
                 name and md5 by md5 against the frozen table; a mismatch means upstream
                 published a new version, which must stop the run and be recorded in
                 DECISIONS. When the API is blocked the frozen table takes over, so
                 ndownloader remains usable if only the API host is blocked.
        Args: none.
        Returns: (files, version) where files are dicts with name / size / md5 / download_url /
                 file_id and version looks like `figshare:v2`.
        Data source: figshare article 30917717, falling back to the expected-file table.
        Config: none.
        Side effects: one network request (none on the fallback path).
        Raises: ValueError on a wrong file count or an API/table mismatch; RuntimeError for
                non-403 network failures, which must not be silently papered over.
    """
    try:
        meta = _http_json(POLYBENCH_API, what="figshare article metadata")
    except urllib.error.HTTPError as exc:
        if exc.code != 403:
            raise
        LOGGER.warning(
            "figshare 返回 403（Server: awselb/2.0）：本机出口被存档方拦截，改用冻结表里的"
            "下载 URL 直连 ndownloader / 403 from figshare, falling back to the frozen table"
        )
        return load_polybench_expected_files(), POLYBENCH_VERSION
    except (RuntimeError, OSError, ValueError) as exc:
        LOGGER.warning(
            "figshare 元数据取不到（%s），改用冻结表 / metadata unavailable, using the frozen table", exc
        )
        return load_polybench_expected_files(), POLYBENCH_VERSION

    files = meta.get("files") or []
    version = f"figshare:v{meta.get('version', '?')}"
    if len(files) != POLYBENCH_EXPECTED_FILES:
        raise ValueError(
            f"figshare 列出 {len(files)} 个文件，官方应为 {POLYBENCH_EXPECTED_FILES} 个；"
            f"版本变了要记 docs/DECISIONS.md / figshare listed {len(files)} files, expected "
            f"{POLYBENCH_EXPECTED_FILES}"
        )

    # API 与冻结表对齐：名字集合先对，再逐名核 md5。 / names first, then md5 per name.
    expected = {str(row["name"]): row for row in load_polybench_expected_files()}
    live = {str(entry["name"]): entry for entry in files}
    if set(live) != set(expected):
        raise ValueError(
            f"figshare 的文件名与冻结表不一致：新增 {sorted(set(live) - set(expected))}，"
            f"缺失 {sorted(set(expected) - set(live))}；上游版本变了，先记 docs/DECISIONS.md"
        )
    changed = {
        name: (live[name]["supplied_md5"], expected[name]["md5"])
        for name in live
        if live[name]["supplied_md5"] != expected[name]["md5"]
    }
    if changed:
        raise ValueError(f"figshare 的 md5 与冻结表不符（线上, 表内）：{changed} / md5 drift")

    return [
        {
            "name": entry["name"],
            "size": int(entry["size"]),
            "md5": entry["supplied_md5"],
            "download_url": entry["download_url"],
            "file_id": int(entry["id"]),
        }
        for entry in files
    ], version


def adopt_local_polybench(out_manifest: Path | None = None) -> int:
    """把已放进 `data/raw/polybench/` 的 39 个 CSV 逐一核 md5，通过后写成清单（不联网）。
    Verify the 39 CSVs already sitting in data/raw/polybench by md5, then write the manifest
    (no network involved).

    中文：
        用途：给"浏览器打包下载 / 别人代下 / 平台手动下"这条路线一个正式的收尾动作——
              文件丢进目录，跑一条命令，逐份核官方 md5，全过才出清单。核不过就**不出清单**
              并逐条说明差在哪（缺哪个、多哪个、哪个字节数不对、哪个 md5 不对）。
        输入：out_manifest —— 默认 `data/manifests/polybench_download_manifest.csv`。
        输出：写了几行；本地一个 CSV 都没有时返回 0 并记一条 INFO（按"还没下"处理，不算错）。
        数据来源：`data/raw/polybench/*.csv`；期望值来自 `polybench_expected_files.csv`。
        配置来源：无。
        副作用：写清单文件（仅当 39 份全部核对通过）。
        异常：ValueError —— 有缺、有多、字节数或 md5 不符，异常信息里逐条列全。

    English:
        Purpose: give the "browser bundle / colleague-downloaded / platform manual" routes a
                 proper closing step: drop the files in, run one command, check the official
                 md5 of each, and only then emit a manifest. A failure emits nothing and lists
                 every problem (missing, unexpected, wrong size, wrong md5).
        Args: out_manifest - defaults to data/manifests/polybench_download_manifest.csv.
        Returns: the number of rows written; 0 with an INFO line when no CSV is present yet.
        Data source: data/raw/polybench/*.csv; expectations from the expected-file table.
        Config: none.
        Side effects: writes the manifest, only when all 39 files verify.
        Raises: ValueError listing every discrepancy when anything fails to match.
    """
    expected = {str(row["name"]): row for row in load_polybench_expected_files()}
    raw_dir = RAW_ROOT / "polybench"
    present = sorted(path.name for path in raw_dir.glob("*.csv")) if raw_dir.exists() else []
    if not present:
        LOGGER.info("本地还没有 PolyBench CSV（%s），按\"尚未下载\"处理 / no local CSVs yet", raw_dir)
        return 0

    problems: list[str] = []
    missing = sorted(set(expected) - set(present))
    unexpected = sorted(set(present) - set(expected))
    if missing:
        problems.append(f"缺 {len(missing)} 个 / missing {len(missing)}: {missing}")
    if unexpected:
        problems.append(f"多出 {len(unexpected)} 个非官方文件 / unexpected: {unexpected}")

    rows: list[DownloadRow] = []
    for name in sorted(set(expected) & set(present)):
        path = raw_dir / name
        payload = path.read_bytes()
        want_size = int(str(expected[name]["size"]))
        want_md5 = str(expected[name]["md5"])
        if len(payload) != want_size:
            problems.append(f"{name}：{len(payload):,} B，应为 {want_size:,} B / size mismatch")
            continue
        got_md5 = _digest(payload, "md5")
        if got_md5 != want_md5:
            problems.append(f"{name}：md5 {got_md5}，应为 {want_md5} / md5 mismatch")
            continue
        rows.append(
            DownloadRow(
                task_id=name.removesuffix(".csv"),
                n_rows=_count_csv_rows(path),
                sha256=_digest(payload, "sha256"),
                source_url=str(expected[name]["download_url"]),
                version=POLYBENCH_VERSION,
                downloaded_at=_now_iso(),
                bytes=len(payload),
                n_rows_basis="csv_data_rows",
            )
        )

    if problems:
        raise ValueError(
            "本地文件与官方存档对不上，不出清单 / local files do not match the archive:\n  - "
            + "\n  - ".join(problems)
        )

    target = out_manifest or (MANIFEST_ROOT / "polybench_download_manifest.csv")
    write_manifest(rows, target)
    LOGGER.info(
        "39 份逐一核过官方 md5，清单已写 / all 39 files matched, manifest written: %s", target
    )
    return len(rows)


def download_polybench(*, limit: int | None, force: bool, dry_run: bool) -> list[DownloadRow]:
    """从 figshare 存档取 PolyBench 的 39 个 CSV，逐份校验 md5 并算 sha256。
    Fetch the 39 PolyBench CSVs from the figshare archive, verify md5 and hash each.

    中文：
        用途：A-3 的输入全靠这一步。存档每个文件自带 `supplied_md5`，官方自称与
              `computed_md5` 相同，所以 md5 就是发布方给的校验值——对不上就重下，
              三次对不上就报错，不许"差不多"。
        输入：limit —— 只取前 N 个文件（调试用，None 表示全部 39 个）；
              force —— 忽略本地已有文件，强制重下；
              dry_run —— 只列清单与目标路径，不写盘。
        输出：`DownloadRow` 列表，每份 CSV 一行；顺序沿用 figshare 返回顺序。
        数据来源：`data/manifests/SOURCES.md` §2.1 的 figshare 存档（DOI 10.6084/m9.figshare.30917717，v2）；
                  文件列表由 `_polybench_files()` 给出（API 优先，被拦退回 `polybench_expected_files.csv`）。
        配置来源：无。
        副作用：写 `data/raw/polybench/<数据集名>.csv`。
        异常：RuntimeError —— 网络不可达或被拦截（含 HTTP 403，见下方提示）；
              ValueError —— md5 三次都不符，或文件数不是 39。

    English:
        Purpose: this step feeds all of A-3. Each archived file carries a `supplied_md5`
                 the publisher asserts equals its own computed md5, so md5 is the publisher's
                 checksum: retry on mismatch, fail after three tries, never "close enough".
        Args: limit - take only the first N files (debugging; None = all 39);
              force - ignore existing local files; dry_run - print the plan, write nothing.
        Returns: one `DownloadRow` per CSV, in figshare's order.
        Data source: the figshare archive in SOURCES.md §2.1 (DOI 10.6084/m9.figshare.30917717, v2);
                 the file list comes from `_polybench_files()` (API first, frozen table when blocked).
        Config: none.
        Side effects: writes data/raw/polybench/<dataset name>.csv.
        Raises: RuntimeError when the network is unreachable or blocked (HTTP 403 included);
                ValueError when md5 mismatches three times or the file count is not 39.
    """
    LOGGER.info("PolyBench：确定要下的 39 个文件 / resolving the 39 files")
    files, version = _polybench_files()

    wanted = files[:limit] if limit else files
    out_dir = RAW_ROOT / "polybench"
    rows: list[DownloadRow] = []
    for index, entry in enumerate(wanted, start=1):
        name = str(entry["name"])
        url = str(entry["download_url"])
        supplied_md5 = str(entry["md5"])
        target = out_dir / name
        LOGGER.info("[%d/%d] %s（%d B）/ fetching", index, len(wanted), name, entry["size"])

        if dry_run:
            LOGGER.info("  dry-run：将写到 %s / would write %s", target, target)
            rows.append(
                DownloadRow(
                    task_id=name.removesuffix(".csv"),
                    n_rows="",
                    sha256="",
                    source_url=url,
                    version=version,
                    downloaded_at="",
                    bytes=int(entry["size"]),
                    n_rows_basis="dry_run",
                )
            )
            continue

        if target.exists() and not force:
            LOGGER.info("  已存在，跳过（--force 可重下）/ exists, skipped (use --force)")
        else:
            try:
                payload = _http_get(url, what=f"polybench/{name}")
            except urllib.error.HTTPError as exc:
                if exc.code == 403:
                    raise RuntimeError(
                        f"下载 {name} 时 ndownloader 也返回 403：本机到 figshare 的出口整体被拦。\n"
                        "  脚本这条路走不通不代表数据拿不到——任一条人工路线走通后：\n"
                        "    1) 把 39 个 CSV 放进 data/raw/polybench/，文件名照抄（空格、大小写、下划线都要一致）；\n"
                        "    2) 跑 python scripts/00_download.py --verify-only，它不出网，逐份核官方 md5，\n"
                        "       全过就写出 polybench_download_manifest.csv。\n"
                        "  39 个文件名、字节数、官方 md5 见 data/manifests/polybench_expected_files.csv；\n"
                        "  出处与取舍见 docs/交付_P1_A2_下载与完整性.md §5。\n"
                        "  ndownloader also returned 403; use the manual route, then --verify-only."
                    ) from exc
                raise
            if _digest(payload, "md5") != supplied_md5:
                raise ValueError(
                    f"{name} 的 md5 与 figshare 公布值不符 / md5 mismatch for {name}: "
                    f"got {_digest(payload, 'md5')}, expected {supplied_md5}"
                )
            _atomic_write(target, payload)

        rows.append(
            DownloadRow(
                task_id=name.removesuffix(".csv"),
                n_rows=_count_csv_rows(target),
                sha256=_digest(target.read_bytes(), "sha256"),
                source_url=url,
                version=version,
                downloaded_at=_now_iso(),
                bytes=target.stat().st_size,
                n_rows_basis="csv_data_rows",
            )
        )

    if not dry_run:
        out_path = manifest_path("polybench", limit)
        write_manifest(rows, out_path)
        LOGGER.info("已写清单 / manifest written: %s（%d 行）", out_path, len(rows))
    return rows


# ---- GitHub 两个仓库（PolyBench26 / OpenPoly）/ The two GitHub repos ----


@dataclass
class RepoSpec:
    """一个 GitHub 基准的下载规格：仓库、commit、要哪些文件、期望值。
    Download spec for one GitHub benchmark: repo, commit, wanted paths, expectations.

    中文：
        用途：PolyBench26 与 OpenPoly 的下载流程完全同构（取 tarball → 按白名单解包 →
              校验 → 数行数 → 写清单），差异全收在这个结构里，避免复制两遍流程。
        字段：repo —— `owner/name`；label —— 人读名（也是清单文件名与 raw 子目录名）；
              commit —— 固定 commit 全 sha（版本列取前 7 位）；wanted_files —— 精确路径白名单；
              wanted_prefixes —— 目录前缀白名单；expected_sha256 / expected_rows —— 官方公布值。
              条数口径不在这里设：它由文件后缀决定（`.csv` 数数据行、`.jsonl` 数非空行、
              其余记 `not_tabular`），因为 PolyBench26 的下载集里既有 jsonl 也有 csv。

    English:
        Purpose: PolyBench26 and OpenPoly share one flow (tarball -> whitelist extraction ->
                 verify -> count -> manifest); the differences live in this structure.
        Fields: repo, label, commit, wanted_files, wanted_prefixes, expected_sha256,
                expected_rows. The row-count basis is *not* a field: it follows the file
                suffix (`.csv` -> data rows, `.jsonl` -> non-empty lines, else `not_tabular`),
                because PolyBench26's download set contains both jsonl and csv.
    """

    repo: str
    label: str
    commit: str
    wanted_files: tuple[str, ...]
    wanted_prefixes: tuple[str, ...]
    expected_sha256: dict[str, str]
    expected_rows: dict[str, int]


def _tarball_url(repo: str, commit: str) -> str:
    """拼出 GitHub 仓库 tarball 的地址。
    Build the codeload tarball URL for a GitHub repo at a commit.

    中文：
        用途：**为什么不用 raw.githubusercontent.com**——实测该域名在本机会在传输中
              "remote party requests renegotiation" 后断连（Schannel 后端），同一 commit
              的 tarball 走 codeload 却稳定。tarball 还顺带把"一个仓库只发一次请求"变成
              现实，绕开 GitHub 未认证 60 次/时的限流。两者字节一致，且我们用官方 sha256
              兜底，所以取哪条通道都不影响可追溯性。
        输入：repo —— `owner/name`；commit —— 全 sha 或分支名。
        输出：str，`https://codeload.github.com/<repo>/tar.gz/<commit>`。
        数据来源：GitHub 官方 codeload 服务。
        配置来源：无。
        副作用：无。
        异常：无。

    English:
        Purpose: raw.githubusercontent.com proved unreliable from this host (the transfer dies
                 with "remote party requests renegotiation" under the Schannel backend), while
                 codeload is stable; a tarball also turns "one repo" into one request, dodging
                 GitHub's 60/h unauthenticated limit. Bytes are identical and the official
                 sha256 still gates them, so the channel does not affect traceability.
        Args: repo - `owner/name`; commit - full sha or branch name.
        Returns: the codeload tarball URL.
        Data source: GitHub's codeload service.
        Config: none.
        Side effects: none.
        Raises: none.
    """
    return f"https://codeload.github.com/{repo}/tar.gz/{commit}"


def _extract_from_tarball(
    archive: bytes,
    *,
    spec: RepoSpec,
    out_root: Path,
    limit: int | None,
    dry_run: bool,
) -> list[tuple[str, Path]]:
    """按白名单从 tarball 里解出文件，返回 (仓库内相对路径, 本地落盘路径)。
    Extract whitelisted files from the tarball; return (repo-relative path, local path).

    中文：
        用途：只解白名单里的东西——tarball 里有 215 个文件、98 MB，全解到盘上既慢又会让
              `data/raw/` 混进用不上的东西（例如 polyBERT 的 pkl、别人的模型脚本）。
              白名单本身也是"我们到底用了哪些字节"的声明。
        输入：archive —— tarball 原始字节；spec —— 见 `RepoSpec`；out_root —— 落盘根
              （如 `data/raw/polybench26`）；limit —— 最多解几个文件；dry_run —— 只列不解。
        输出：列表，元素为 (相对路径, 本地路径)；顺序沿用 tarball 内顺序。
        数据来源：`spec.repo` 在 `spec.commit` 的 tarball。
        配置来源：`spec.wanted_files` / `spec.wanted_prefixes`。
        副作用：写 `out_root/**`（dry_run 时不写）。
        异常：ValueError —— tarball 里找不到白名单中的某个精确路径（说明 commit 或路径写错，
              必须当场失败，不能静默少一个文件）；tarfile.TarError —— 包损坏。

    English:
        Purpose: extract only whitelisted members. The tarball holds 215 files / 98 MB and
                 unpacking all of it is both slow and polluting; the whitelist doubles as a
                 declaration of exactly which bytes we use.
        Args: archive - raw tarball bytes; spec - see `RepoSpec`; out_root - extraction root;
              limit - max files; dry_run - list only.
        Returns: list of (repo-relative path, local path) in tarball order.
        Data source: tarball of `spec.repo` at `spec.commit`.
        Config: `spec.wanted_files` / `spec.wanted_prefixes`.
        Side effects: writes under `out_root` (nothing in dry_run).
        Raises: ValueError when a whitelisted exact path is absent (a wrong commit or path must
                fail loudly, never silently drop a file); tarfile.TarError on a corrupt archive.
    """
    wanted = set(spec.wanted_files)
    found: list[tuple[str, Path]] = []
    seen: set[str] = set()

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # GitHub 的 tarball 顶层是 `<repo>-<sha>/`，剥掉这一层才是仓库内路径。
            # GitHub wraps everything in `<repo>-<sha>/`; strip it to get repo-relative paths.
            repo_path = member.name.split("/", 1)[-1]
            if repo_path not in wanted and not repo_path.startswith(spec.wanted_prefixes):
                continue
            if limit is not None and len(found) >= limit:
                break
            target = out_root / repo_path
            if dry_run:
                LOGGER.info("  dry-run：将解出 %s → %s / would extract to %s", repo_path, target, target)
            else:
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ValueError(f"tarball 成员不可读 / unreadable member: {member.name}")
                _atomic_write(target, extracted.read())
            found.append((repo_path, target))
            seen.add(repo_path)

    missing = wanted - seen
    if missing and limit is None:
        raise ValueError(
            f"{spec.label} 的 tarball 里找不到这些路径：{sorted(missing)}；"
            f"commit={spec.commit}，路径可能已变，先更新本文件常量并记 DECISIONS / not found"
        )
    return found


def download_repo_benchmark(
    spec: RepoSpec, *, limit: int | None, force: bool, dry_run: bool
) -> list[DownloadRow]:
    """下载一个 GitHub 基准并写出它的清单。
    Download one GitHub benchmark and write its manifest.

    中文：
        用途：PolyBench26 与 OpenPoly 共用的主流程：取 tarball → 按白名单解包 → 逐个校验
              sha256 与条数 → 写 manifest。
        输入：spec —— 见 `RepoSpec`；limit —— 最多几个文件；force —— 已存在也重下；
              dry_run —— 只打印计划。
        输出：`DownloadRow` 列表。
        数据来源：`spec.repo` @ `spec.commit`。
        配置来源：`spec` 的各个白名单与期望值（其上游是 `data/manifests/SOURCES.md`）。
        副作用：写 `data/raw/<label>/**` 与 `data/manifests/<label>_download_manifest.csv`。
        异常：RuntimeError —— 网络失败；ValueError —— 官方 sha256 或条数对不上。

    English:
        Purpose: the shared flow for PolyBench26 and OpenPoly: tarball -> whitelist extraction
                 -> per-file sha256 and row checks -> manifest.
        Args: spec - see `RepoSpec`; limit - max files; force - re-download anyway;
              dry_run - print the plan only.
        Returns: list of `DownloadRow`.
        Data source: `spec.repo` at `spec.commit`.
        Config: the whitelists and expectations in `spec` (upstream: SOURCES.md).
        Side effects: writes data/raw/<label>/** and the matching manifest CSV.
        Raises: RuntimeError on network failure; ValueError on a sha256 or row-count mismatch.
    """
    out_root = RAW_ROOT / spec.label
    version = spec.commit[:7]
    LOGGER.info("%s：commit %s / fetching tarball", spec.label, version)

    if dry_run:
        LOGGER.info("  dry-run：将从 %s 取 %d 个精确路径 + 前缀 %s / would fetch from the tarball",
                    _tarball_url(spec.repo, spec.commit),
                    len(spec.wanted_files), list(spec.wanted_prefixes) or "无 / none")
        return []

    archive = _http_get(_tarball_url(spec.repo, spec.commit), what=f"{spec.label} tarball")
    LOGGER.info("  tarball %.1f MiB / received", len(archive) / 2**20)
    extracted = _extract_from_tarball(archive, spec=spec, out_root=out_root, limit=limit, dry_run=dry_run)
    if limit is not None:
        LOGGER.warning("  --limit %d：只处理了前 %d 个文件，清单不完整 / partial run", limit, len(extracted))

    rows: list[DownloadRow] = []
    for repo_path, local_path in extracted:
        data = local_path.read_bytes()
        sha = _digest(data, "sha256")

        # ---- 1. 官方 sha256（只对官方公布过的那两个文件）/ official sha256 where published ----
        expected = spec.expected_sha256.get(repo_path)
        if expected is not None and sha != expected:
            raise ValueError(
                f"{repo_path} 的 sha256 与官方公布值不符 / sha256 mismatch: got {sha}, expected {expected}"
            )

        # ---- 2. 条数：口径由后缀决定 / record count: the basis follows the suffix ----
        if local_path.suffix == ".csv":
            n_rows: int | str = _count_csv_rows(local_path)
            basis = "csv_data_rows"
        elif local_path.suffix == ".jsonl":
            n_rows = _count_jsonl_records(local_path)
            basis = "jsonl_records"
        else:
            n_rows = ""
            basis = "not_tabular"
        expected_rows = spec.expected_rows.get(repo_path)
        if expected_rows is not None and n_rows != expected_rows:
            raise ValueError(
                f"{repo_path} 有 {n_rows} 条记录（{basis}），官方公布 {expected_rows} 条 / "
                f"record-count mismatch"
            )

        rows.append(
            DownloadRow(
                task_id=repo_path,
                n_rows=n_rows,
                sha256=sha,
                source_url=f"https://github.com/{spec.repo}/blob/{spec.commit}/{repo_path}",
                version=version,
                downloaded_at=_now_iso(),
                bytes=len(data),
                n_rows_basis=basis,
            )
        )
        LOGGER.info("  OK %s（%s 行, %d B）/ verified", repo_path, n_rows if n_rows != "" else "-", len(data))

    out_path = manifest_path(spec.label, limit)
    write_manifest(rows, out_path)
    LOGGER.info("已写清单 / manifest written: %s（%d 行）", out_path, len(rows))
    return rows


# ---- 校验 / Validation ----


def validate_polybench_manifest(manifest_path: Path | None = None) -> None:
    """校验 PolyBench 下载清单是否满足 A-2 的验收标准 ①②③④⑥。
    Validate the PolyBench manifest against A-2 acceptance criteria 1-4 and 6.

    中文：
        用途：把分工书里的验收标准写成可执行断言，让"39 行 / 42,169 / 与附录 A 逐行对齐"
              变成一条命令，而不是一次人工肉眼核对。将来 D 复核入口时也用它。
        输入：manifest_path —— 默认 `data/manifests/polybench_download_manifest.csv`。
        输出：无。全部通过则什么也不做（静默即通过），并记一条 INFO。
        数据来源：该 manifest；期望值来自 Proposal 附录 A（本文件 `POLYBENCH_APPENDIX_A`）。
        配置来源：无。
        副作用：只读。
        异常：FileNotFoundError —— 清单不存在（说明还没成功下载过）；
              AssertionError —— 条数、合计、或逐行对齐任一项不符，异常信息里写清差在哪。

    English:
        Purpose: turn the written acceptance criteria into executable assertions so that
                 "39 rows / 42,169 / row-by-row alignment with appendix A" is one command
                 rather than a manual eyeball pass. D will reuse it when auditing the entry.
        Args: manifest_path - defaults to data/manifests/polybench_download_manifest.csv.
        Returns: nothing; silence means pass, plus one INFO line.
        Data source: that manifest; expectations from Proposal appendix A.
        Config: none.
        Side effects: read-only.
        Raises: FileNotFoundError if the manifest is missing; AssertionError naming the exact
                mismatch in file count, row total, or per-task alignment.
    """
    path = manifest_path or (MANIFEST_ROOT / "polybench_download_manifest.csv")
    if not path.exists():
        raise FileNotFoundError(f"没有 PolyBench 清单，先跑下载 / no manifest yet: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    # ---- ① 恰好 39 行 / exactly 39 rows ----
    assert len(rows) == POLYBENCH_EXPECTED_FILES, (
        f"清单 {len(rows)} 行，应为 {POLYBENCH_EXPECTED_FILES} 行 / expected 39 rows, got {len(rows)}"
    )

    # ---- ② n_rows 合计 42,169 / row total ----
    total = sum(int(row["n_rows"]) for row in rows)
    assert total == POLYBENCH_EXPECTED_ROWS, (
        f"n_rows 合计 {total}，应为 {POLYBENCH_EXPECTED_ROWS} / expected 42,169 records, got {total}"
    )

    # ---- ③ 与附录 A 逐行对齐（名字集合、逐个条数）/ alignment with appendix A ----
    got = {row["task_id"]: int(row["n_rows"]) for row in rows}
    missing = set(POLYBENCH_APPENDIX_A) - set(got)
    extra = set(got) - set(POLYBENCH_APPENDIX_A)
    assert not missing and not extra, (
        f"任务名与附录 A 不一致：缺 {sorted(missing)}，多 {sorted(extra)} / name mismatch"
    )
    wrong = {name: (got[name], POLYBENCH_APPENDIX_A[name]) for name in got
             if got[name] != POLYBENCH_APPENDIX_A[name]}
    assert not wrong, f"这些任务的条数与附录 A 不符（实际, 应为）：{wrong} / count mismatch"

    # ---- ④⑥ 每行 sha256 / source_url / version 均非空 / non-empty provenance columns ----
    for row in rows:
        for column in ("sha256", "source_url", "version"):
            assert row.get(column, "").strip(), (
                f"{row.get('task_id', '?')} 的 {column} 为空 / empty {column}"
            )

    LOGGER.info("PolyBench 清单校验通过：39 行 / 合计 %d / 与附录 A 逐行一致 / all checks passed", total)


# ---- 命令行 / CLI ----


def _build_specs() -> dict[str, RepoSpec]:
    """构造两个 GitHub 基准的下载规格，键即 `--source` 的取值。
    Build the download specs of the two GitHub benchmarks; keys are `--source` values.

    中文：
        用途：把常量区的白名单与期望值组装成 `RepoSpec`，让 `main()` 只剩调度。
        输入：无（全部取自本文件常量区）。
        输出：dict[str, RepoSpec]，键为 `polybench26` / `openpoly`。
        数据来源：本文件常量区，其上游是 `data/manifests/SOURCES.md` §2.2 与 §2.3。
        配置来源：无。
        副作用：无。
        异常：无。

    English:
        Purpose: assemble the constants into `RepoSpec` objects so `main()` is pure dispatch.
        Args: none.
        Returns: dict keyed by `polybench26` / `openpoly`.
        Data source: this file's constants, upstream SOURCES.md §2.2 and §2.3.
        Config: none.
        Side effects: none.
        Raises: none.
    """
    return {
        "polybench26": RepoSpec(
            repo=POLYBENCH26_REPO,
            label="polybench26",
            commit=POLYBENCH26_COMMIT,
            wanted_files=POLYBENCH26_WANTED_FILES,
            wanted_prefixes=POLYBENCH26_WANTED_PREFIXES,
            expected_sha256=POLYBENCH26_EXPECTED_SHA256,
            expected_rows=POLYBENCH26_EXPECTED_ROWS,
        ),
        "openpoly": RepoSpec(
            repo=OPENPOLY_REPO,
            label="openpoly",
            commit=OPENPOLY_COMMIT,
            wanted_files=OPENPOLY_WANTED_FILES,
            wanted_prefixes=(),
            expected_sha256={},          # OpenPoly 未公布逐文件 sha256，靠 741 行兜底 / no published sha256
            expected_rows=OPENPOLY_EXPECTED_ROWS,
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。
    Parse command-line arguments.

    中文：
        用途：`--dry-run` / `--limit` 是 README 第七节强制的两条（阶段入口必须能空跑与限量跑）。
        输入：argv —— 参数列表，None 表示取 `sys.argv[1:]`。
        输出：argparse.Namespace，含 source / limit / force / dry_run / verify_only / verbose。
        数据来源：命令行。
        配置来源：无。
        副作用：无（argparse 解析失败会 SystemExit 2）。
        异常：SystemExit —— 参数非法。

    English:
        Purpose: `--dry-run` / `--limit` are mandatory for stage entry points (README §7).
        Args: argv - argument list; None means sys.argv[1:].
        Returns: Namespace with source / limit / force / dry_run / verify_only / verbose.
        Data source: the command line.
        Config: none.
        Side effects: none (argparse exits with code 2 on bad input).
        Raises: SystemExit on invalid arguments.
    """
    parser = argparse.ArgumentParser(
        description="三基准下载与完整性核对（A-2）/ download the three benchmarks and verify them",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=("all", "polybench", "polybench26", "openpoly"),
        default="all",
        help="下哪个基准；默认 all / which benchmark, default all",
    )
    parser.add_argument("--limit", type=int, default=None, help="最多处理几个文件，调试用 / max files, for debugging")
    parser.add_argument(
        "--force",
        action="store_true",
        help="PolyBench 专用：本地已有也重下；GitHub 两个基准每次都由 tarball 重放，不读本地缓存 / "
             "PolyBench only: re-download even if present; the GitHub ones always replay from the tarball",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写盘、不联网改状态 / print the plan only")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="不出网：先把 data/raw/polybench/ 里手动放进去的 CSV 逐一核官方 md5 并补清单，"
             "再校验清单（39 行 / 42,169 / 与附录 A 对齐）/ offline: adopt hand-placed CSVs, then validate",
    )
    parser.add_argument(
        "--ca-bundle",
        default=os.environ.get("CHAINWORTH_CA_BUNDLE"),
        metavar="PEM|windows",
        help="补一张根证书（企业代理会中间人解密 TLS）；填 .pem 路径或 `windows` 读系统证书库。"
             "默认读环境变量 CHAINWORTH_CA_BUNDLE / extra trust anchor for MITM proxies",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="打 DEBUG 日志 / debug logging")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：按 `--source` 调度三个基准的下载与校验。
    CLI entry point: dispatch the three benchmarks according to `--source`.

    中文：
        用途：把"下载 → 校验 → 写清单 → 复核"串起来；任一基准失败就返回非零退出码，
              但**已经成功的基准的清单会保留**——半份真凭据比一份都没有有用，
              未完成的那个基准不会写清单（宁缺勿假）。
        输入：argv —— 见 `_parse_args`。
        输出：int，0 表示本次请求的基准全部成功；1 表示有基准失败。
        数据来源：见各 `download_*`。
        配置来源：见各 `download_*`。
        副作用：写 `data/raw/**` 与 `data/manifests/**`；往 stderr 打日志。
        异常：不向外抛（除 KeyboardInterrupt）；失败转成返回码。

    English:
        Purpose: chain download -> verify -> manifest -> re-check. Any failing benchmark
                 yields a non-zero exit code, but manifests of benchmarks that did succeed are
                 kept — half of a real record beats none, and the failed one writes no manifest
                 (better missing than fabricated).
        Args: argv - see `_parse_args`.
        Returns: 0 when every requested benchmark succeeded, 1 otherwise.
        Data source: see the `download_*` functions.
        Config: see the `download_*` functions.
        Side effects: writes data/raw/** and data/manifests/**; logs to stderr.
        Raises: nothing but KeyboardInterrupt; failures become the return code.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- 0. 先定 TLS 信任，再谈下载：信任配错了后面每条请求都会失败 ----
    # ---- 0. Fix the trust anchors first; a wrong trust setup fails every request ----
    if args.ca_bundle:
        global _SSL_CONTEXT
        _SSL_CONTEXT = build_ssl_context(args.ca_bundle)

    if args.verify_only:
        try:
            # 先看盘上有没有人手动放进去的 39 个 CSV：有就核 md5 并补清单（不出网）。
            # First adopt any hand-placed CSVs: verify md5 and write the manifest, offline.
            if args.source in ("all", "polybench"):
                adopt_local_polybench()
            validate_polybench_manifest()
        except (FileNotFoundError, AssertionError, ValueError) as exc:
            # 校验失败要把"差在哪"原样说出来，别用 traceback 吓人；返回码 1 供 CI 判断。
            # State exactly what is wrong instead of dumping a traceback; exit 1 for CI.
            LOGGER.error("校验未通过 / verification failed: %s", exc)
            return 1
        return 0

    wanted = ("polybench", "polybench26", "openpoly") if args.source == "all" else (args.source,)
    specs = _build_specs()
    failures: list[str] = []

    for name in wanted:
        LOGGER.info("=" * 62)
        LOGGER.info("基准 / benchmark: %s", name)
        try:
            if name == "polybench":
                rows = download_polybench(limit=args.limit, force=args.force, dry_run=args.dry_run)
                if not args.dry_run:
                    validate_polybench_manifest()
                LOGGER.info("PolyBench 完成：%d 个文件 / done", len(rows))
            else:
                rows = download_repo_benchmark(specs[name], limit=args.limit, force=args.force, dry_run=args.dry_run)
                LOGGER.info("%s 完成：%d 个文件 / done", name, len(rows))
        except Exception as exc:  # noqa: BLE001 —— 逐个基准兜住，一个失败不影响另外两个 / isolate per benchmark
            LOGGER.error("%s 失败 / %s failed:\n%s", name, name, exc)
            failures.append(name)

    LOGGER.info("=" * 62)
    if failures:
        LOGGER.error("有基准未完成：%s。清单未写，按提示处理后重跑 / unfinished: %s", failures, failures)
        return 1
    LOGGER.info("本次请求的基准全部完成：%s / all requested benchmarks done", list(wanted))
    return 0


if __name__ == "__main__":
    sys.exit(main())
