#!/usr/bin/env python3
"""Collect and explain an on-site FlagGems/Triton/MLIR/SME compilation dump."""

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path


TEXT_SUFFIXES = {
    ".mlir",
    ".ll",
    ".llir",
    ".ir",
    ".ttir",
    ".ttsharedir",
    ".s",
    ".asm",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".py",
    ".json",
    ".txt",
    ".ref",
}
BINARY_SUFFIXES = {".o", ".obj", ".so", ".a"}
NUMBERED_STAGE_RE = re.compile(r"^(\d+)_.*\.(?:mlir|ll|ir)$", re.IGNORECASE)
MLIR_OP_RE = re.compile(
    r'"([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_.]+)"'
    r"|(?<![A-Za-z0-9_.])([a-z][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]*)"
)
FEATURE_PATTERNS = {
    "triton": re.compile(r"\btt\.|\bttg\.|\btriton\b", re.IGNORECASE),
    "linalg": re.compile(r"\blinalg\.(?:matmul|batch_matmul|generic)\b"),
    "affine_loop": re.compile(r"\baffine\.(?:for|parallel)\b"),
    "scf_loop": re.compile(r"\bscf\.(?:for|parallel|forall)\b"),
    "vector_contract": re.compile(r"\bvector\.contract\b"),
    "vector_outerproduct": re.compile(r"\bvector\.outerproduct\b"),
    "arm_sme_mlir": re.compile(
        r"\barm_sme\.(?:intr\.|outerproduct\b|tile_|get_tile|zero\b|move\b|load\b|store\b)"
    ),
    "arm_sme_attrs": re.compile(
        r"\b(?:arm_new_za|arm_locally_streaming|aarch64_new_za)\b"
        r"|\barm_sme\.(?:demo|schedule|noalias|fp_reassociation|bmm)\b"
    ),
    "llvm_sme": re.compile(r"llvm\.aarch64\.sme|aarch64_new_za"),
    "llvm_sve": re.compile(r"llvm\.aarch64\.sve"),
    "mopa": re.compile(r"\b(?:f|s|u|us|su)?mopa\b", re.IGNORECASE),
    "smstart": re.compile(r"\bsmstart\b", re.IGNORECASE),
    "smstop": re.compile(r"\bsmstop\b", re.IGNORECASE),
    "za": re.compile(r"\bza(?:\d+)?\b|tile_id", re.IGNORECASE),
    "load_store": re.compile(
        r"\b(?:memref|llvm|vector|tt)\.(?:load|store|transfer_read|transfer_write)\b"
        r"|^\s*(?:%[^=]+=\s*)?load\b|^\s*store\b",
        re.MULTILINE,
    ),
    "prefetch": re.compile(r"\bprfm\b|llvm\.prefetch|\bprefetch\b", re.IGNORECASE),
    "c_interface": re.compile(r"_mlir_ciface|emit_c_interface"),
}
ENV_KEYS = (
    "PATH",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "CC",
    "CXX",
    "TRITON_",
    "MLIR_",
    "LLVM_",
    "FLAGGEMS_",
)


def run_command(argv, cwd=None, timeout=30):
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "argv": argv,
            "exit_code": completed.returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "output": completed.stdout,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        output = getattr(exc, "stdout", "") or ""
        return {
            "argv": argv,
            "exit_code": None,
            "elapsed_seconds": time.perf_counter() - started,
            "output": output + f"\n[collector error] {exc}\n",
        }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_rel(path, root):
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def read_text_limited(path, max_bytes):
    try:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError:
        return "", True
    truncated = len(raw) > max_bytes
    raw = raw[:max_bytes]
    if b"\x00" in raw[:4096]:
        return "", truncated
    return raw.decode("utf-8", errors="replace"), truncated


def infer_role(path):
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name == "00_input.mlir":
        return "MLIR初始输入", "high"
    if "after_transform_interpreter" in name:
        return "Transform Dialect执行后的MLIR", "high"
    if "erase_schedule" in name or "earase_schedule" in name:
        return "删除Transform调度描述后的MLIR", "high"
    if "convert_to_llvm" in name:
        return "LLVM降级边界MLIR", "high"
    if "canonicalize" in name:
        return "规范化后的MLIR", "high"
    if "strip_debug" in name:
        return "删除调试信息后的MLIR", "high"
    if "legalize_float8" in name:
        return "Float8合法化后的MLIR", "high"
    if name in {"tt.mlir"} or suffix == ".ttir":
        return "Triton高层TTIR", "medium"
    if name in {"ttshared.mlir"} or suffix == ".ttsharedir":
        return "Triton共享布局/数据移动IR", "medium"
    if name in {"ll.mlir"}:
        return "LLVM方言MLIR", "medium"
    if name in {"ll.ir"} or suffix in {".ll", ".llir"}:
        return "LLVM IR", "medium"
    if suffix in {".o", ".obj"}:
        return "目标文件", "high"
    if suffix == ".so":
        return "Linux共享库", "high"
    if name == "main.cxx":
        return "运行时C++驱动", "high"
    if name.endswith(".json"):
        return "编译缓存元数据", "medium"
    if name.endswith(".ref"):
        return "共享库/缓存引用记录", "low"
    if "objdump" in name or "disasm" in name:
        return "已有反汇编文本", "medium"
    if name.startswith("test_") and suffix == ".py":
        return "pytest测试源码", "high"
    return "未分类产物", "low"


def collect_environment(output_dir):
    commands = [
        ("uname", ["uname", "-a"]),
        ("lscpu", ["lscpu"]),
        ("cpuinfo", ["sh", "-c", "cat /proc/cpuinfo 2>/dev/null"]),
        ("memory", ["sh", "-c", "cat /proc/meminfo 2>/dev/null"]),
        ("cache_sysfs", [
            "sh",
            "-c",
            "for f in /sys/devices/system/cpu/cpu0/cache/index*/{level,type,size,coherency_line_size,ways_of_associativity,number_of_sets}; do test -r \"$f\" && printf '%s=' \"$f\" && cat \"$f\"; done",
        ]),
        ("perf", ["sh", "-c", "perf --version 2>&1; printf 'perf_event_paranoid='; cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null"]),
    ]
    for tool in ("python3", "pytest", "clang", "gcc", "mlir-opt", "llc", "llvm-objdump", "objdump"):
        path = shutil.which(tool)
        if path:
            version_args = [path, "--version"]
            if tool == "python3":
                version_args = [path, "--version"]
            commands.append((f"tool_{tool}", version_args))

    package_versions = {}
    for package in ("torch", "triton", "flag_gems", "flaggems", "pytest"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None

    captured = {}
    text_parts = [
        f"collector_python={sys.executable}",
        f"python_version={platform.python_version()}",
        f"platform={platform.platform()}",
        "",
        "[packages]",
        json.dumps(package_versions, ensure_ascii=False, indent=2),
        "",
        "[selected environment]",
    ]
    selected_env = {
        key: value
        for key, value in sorted(os.environ.items())
        if key in ENV_KEYS or any(key.startswith(prefix) for prefix in ENV_KEYS if prefix.endswith("_"))
    }
    text_parts.append(json.dumps(selected_env, ensure_ascii=False, indent=2))
    for label, argv in commands:
        result = run_command(argv)
        captured[label] = result
        text_parts.extend(
            [
                "",
                f"[{label}] exit={result['exit_code']}",
                result["output"].rstrip(),
            ]
        )

    payload = {
        "collector_python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions,
        "environment": selected_env,
        "commands": captured,
    }
    (output_dir / "00_environment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "00_environment.txt").write_text(
        "\n".join(text_parts) + "\n",
        encoding="utf-8",
    )
    return payload


def execute_test(command, workdir, output_dir):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started_wall = dt.datetime.now(dt.timezone.utc)
    started = time.perf_counter()
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    payload = {
        "command": command,
        "argv_interpretation": shlex.split(command),
        "workdir": str(workdir),
        "started_utc": started_wall.isoformat(),
        "elapsed_seconds": elapsed,
        "exit_code": completed.returncode,
        "child_user_seconds": after.ru_utime - before.ru_utime,
        "child_system_seconds": after.ru_stime - before.ru_stime,
        "child_maxrss_raw": after.ru_maxrss,
    }
    (output_dir / "01_test_command.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "01_test_command.log").write_text(
        f"$ cd {workdir}\n$ {command}\n\n{completed.stdout}",
        encoding="utf-8",
    )
    return payload


def record_test_without_execution(command, workdir, output_dir):
    payload = {
        "command": command,
        "workdir": str(workdir) if workdir else None,
        "executed_by_collector": False,
    }
    (output_dir / "01_test_command.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "01_test_command.log").write_text(
        "测试命令仅记录，未由现场工具执行。\n"
        + (f"$ cd {workdir}\n" if workdir else "")
        + (f"$ {command}\n" if command else "未提供测试命令。\n"),
        encoding="utf-8",
    )
    return payload


def build_inventory(dump_root, output_dir):
    rows = []
    for path in sorted(dump_root.rglob("*")):
        if not path.is_file():
            continue
        if output_dir == path.parent or output_dir in path.parents:
            continue
        role, confidence = infer_role(path)
        try:
            stat = path.stat()
            digest = sha256_file(path)
        except OSError:
            continue
        rows.append(
            {
                "path": safe_rel(path, dump_root),
                "bytes": stat.st_size,
                "sha256": digest,
                "suffix": path.suffix.lower(),
                "role": role,
                "role_confidence": confidence,
            }
        )
    json_path = output_dir / "02_artifact_inventory.json"
    csv_path = output_dir / "02_artifact_inventory.csv"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else [
            "path", "bytes", "sha256", "suffix", "role", "role_confidence"
        ])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def analyze_text_file(path, dump_root, max_bytes):
    text, truncated = read_text_limited(path, max_bytes)
    feature_counts = {
        key: len(pattern.findall(text)) for key, pattern in FEATURE_PATTERNS.items()
    }
    operations = Counter()
    if path.suffix.lower() == ".mlir":
        for match in MLIR_OP_RE.finditer(text):
            op = match.group(1) or match.group(2)
            if op and not op.startswith(("arm_sme.schedule.", "arm_sme.demo.")):
                operations[op] += 1
    return {
        "path": safe_rel(path, dump_root),
        "bytes_scanned": len(text.encode("utf-8")),
        "truncated": truncated,
        "lines_scanned": text.count("\n") + (1 if text else 0),
        "feature_counts": feature_counts,
        "operation_counts": dict(operations.most_common()),
    }


def stage_sort_key(path):
    match = NUMBERED_STAGE_RE.match(path.name)
    if match:
        return int(match.group(1)), str(path)
    suffix_rank = {
        ".ttir": 20,
        ".ttsharedir": 30,
        ".llir": 40,
    }
    return suffix_rank.get(path.suffix.lower(), 90), str(path)


def delta_counts(before, after):
    keys = set(before) | set(after)
    added = {}
    removed = {}
    for key in sorted(keys):
        delta = after.get(key, 0) - before.get(key, 0)
        if delta > 0:
            added[key] = delta
        elif delta < 0:
            removed[key] = -delta
    return added, removed


def analyze_ir_chain(dump_root, output_dir, max_bytes):
    candidates = [
        path
        for path in dump_root.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    ]
    analyses = {
        safe_rel(path, dump_root): analyze_text_file(path, dump_root, max_bytes)
        for path in candidates
    }
    numbered = sorted(
        [path for path in candidates if NUMBERED_STAGE_RE.match(path.name)],
        key=stage_sort_key,
    )
    cache_chain = sorted(
        [
            path
            for path in candidates
            if path.suffix.lower() in {".ttir", ".ttsharedir", ".llir"}
        ],
        key=stage_sort_key,
    )
    transitions = []
    for chain_name, chain in (
        ("numbered_mlir_pipeline", numbered),
        ("triton_cache_pipeline", cache_chain),
    ):
        for before_path, after_path in zip(chain, chain[1:]):
            before = analyses[safe_rel(before_path, dump_root)]
            after = analyses[safe_rel(after_path, dump_root)]
            feature_added, feature_removed = delta_counts(
                before["feature_counts"], after["feature_counts"]
            )
            op_added, op_removed = delta_counts(
                before["operation_counts"], after["operation_counts"]
            )
            transitions.append(
                {
                    "chain": chain_name,
                    "from": before["path"],
                    "to": after["path"],
                    "feature_added": feature_added,
                    "feature_removed": feature_removed,
                    "top_ops_added": dict(
                        sorted(op_added.items(), key=lambda item: -item[1])[:12]
                    ),
                    "top_ops_removed": dict(
                        sorted(op_removed.items(), key=lambda item: -item[1])[:12]
                    ),
                }
            )

    first_seen = {}
    ordered = numbered + [path for path in cache_chain if path not in numbered]
    for path in ordered:
        row = analyses[safe_rel(path, dump_root)]
        for feature, count in row["feature_counts"].items():
            if count and feature not in first_seen:
                first_seen[feature] = {"path": row["path"], "count": count}

    payload = {
        "numbered_chain": [safe_rel(path, dump_root) for path in numbered],
        "cache_chain": [safe_rel(path, dump_root) for path in cache_chain],
        "files": analyses,
        "first_seen": first_seen,
        "all_text_evidence": {},
        "transitions": transitions,
    }
    for feature in FEATURE_PATTERNS:
        hits = []
        for row in analyses.values():
            count = row["feature_counts"].get(feature, 0)
            if count:
                hits.append({"path": row["path"], "count": count})
        payload["all_text_evidence"][feature] = hits[:20]
    (output_dir / "03_ir_chain_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# IR 相邻层变化",
        "",
        "这里的“新增/消失”是文本与操作统计变化，不自动等价于严格语义变化。",
        "",
        "## 首次出现",
        "",
        "| 特征 | 首次出现文件 | 当层计数 |",
        "|---|---|---:|",
    ]
    for feature, row in sorted(first_seen.items()):
        lines.append(f"| `{feature}` | `{row['path']}` | {row['count']} |")
    lines.extend(
        [
            "",
            "## 相邻层差分",
            "",
            "| 链 | 前一层 | 后一层 | 新增特征 | 消失特征 |",
            "|---|---|---|---|---|",
        ]
    )
    for row in transitions:
        added = ", ".join(
            f"{key}+{value}" for key, value in row["feature_added"].items()
        ) or "-"
        removed = ", ".join(
            f"{key}-{value}" for key, value in row["feature_removed"].items()
        ) or "-"
        lines.append(
            f"| `{row['chain']}` | `{row['from']}` | `{row['to']}` | "
            f"{added} | {removed} |"
        )
    (output_dir / "04_ir_transitions.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return payload


def disassemble_binaries(dump_root, output_dir, inventory):
    binary_dir = output_dir / "06_binary"
    binary_dir.mkdir(parents=True, exist_ok=True)
    llvm_objdump = shutil.which("llvm-objdump")
    gnu_objdump = shutil.which("objdump")
    readelf = shutil.which("readelf")
    nm = shutil.which("nm")
    file_tool = shutil.which("file")
    results = []
    evidence_lines = []
    binary_rows = [
        row for row in inventory if row["suffix"] in BINARY_SUFFIXES
    ][:32]
    for index, row in enumerate(binary_rows):
        source = dump_root / row["path"]
        stem = f"{index:02d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', source.name)}"
        commands = []
        if file_tool:
            commands.append(("file", [file_tool, str(source)]))
        if readelf:
            commands.extend(
                [
                    ("readelf_header", [readelf, "-h", str(source)]),
                    ("readelf_symbols", [readelf, "-Ws", str(source)]),
                ]
            )
            if source.suffix.lower() == ".so":
                commands.append(("readelf_dynamic", [readelf, "-d", str(source)]))
        if nm:
            commands.append(("nm", [nm, "-a", str(source)]))
        if llvm_objdump:
            commands.append(
                (
                    "disassembly",
                    [llvm_objdump, "--disassemble", "--mattr=+sme", str(source)],
                )
            )
        elif gnu_objdump:
            commands.append(("disassembly", [gnu_objdump, "-d", str(source)]))

        item = {"path": row["path"], "commands": {}, "instruction_counts": {}}
        disassembly = ""
        for label, argv in commands:
            result = run_command(argv, timeout=120)
            item["commands"][label] = {
                "argv": argv,
                "exit_code": result["exit_code"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
            (binary_dir / f"{stem}.{label}.txt").write_text(
                result["output"],
                encoding="utf-8",
            )
            if label == "disassembly":
                disassembly = result["output"]
        for feature in ("fmopa", "smopa", "umopa", "smstart", "smstop", "ptrue", "ld1w", "st1w"):
            item["instruction_counts"][feature] = len(
                re.findall(rf"\b{feature}\b", disassembly, re.IGNORECASE)
            )
        if any(item["instruction_counts"].values()):
            evidence_lines.append(
                f"{row['path']}: "
                + ", ".join(
                    f"{key}={value}"
                    for key, value in item["instruction_counts"].items()
                    if value
                )
            )
        results.append(item)
    (binary_dir / "binary_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return results, evidence_lines


def invoke_generic_analyzer(dump_root, output_dir):
    repo_root = Path(__file__).resolve().parents[2]
    analyzer = (
        repo_root
        / "01_sme_region_discovery"
        / "tools"
        / "analyze_kernel_dump.py"
    )
    if not analyzer.is_file():
        payload = {
            "available": False,
            "reason": "公开现场工具未捆绑可选的高层语义分析器。",
        }
        (output_dir / "07_generic_analysis.log").write_text(
            payload["reason"] + "\n",
            encoding="utf-8",
        )
        (output_dir / "07_generic_analysis_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return payload
    with tempfile.TemporaryDirectory(prefix="sme2-onsite-generic-") as temp:
        temp_dir = Path(temp)
        md_path = temp_dir / "generic.md"
        json_path = temp_dir / "generic.json"
        result = run_command(
            [
                sys.executable,
                str(analyzer),
                str(dump_root),
                "--output",
                str(md_path),
                "--json-output",
                str(json_path),
            ],
            timeout=300,
        )
        payload = None
        if result["exit_code"] == 0 and json_path.is_file():
            full = json.loads(json_path.read_text(encoding="utf-8"))
            safe_keys = (
                "text_files_scanned",
                "kernel_type",
                "type_confidence",
                "furthest_visible_stage",
                "type_scores",
                "feature_counts",
                "structural_metrics",
                "chain_coverage",
                "paper_inspired_checklist",
                "decision_trace",
                "recommendations",
                "risks",
            )
            payload = {key: full.get(key) for key in safe_keys}
    (output_dir / "07_generic_analysis.log").write_text(
        result["output"],
        encoding="utf-8",
    )
    if payload is not None:
        (output_dir / "07_generic_analysis_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload


def write_sme_evidence(output_dir, chain, binary_evidence):
    lines = [
        "# SME 证据摘要",
        "",
        "## IR中特征首次出现",
        "",
    ]
    for feature in (
        "arm_sme_mlir",
        "arm_sme_attrs",
        "llvm_sme",
        "llvm_sve",
        "mopa",
        "smstart",
        "smstop",
        "za",
    ):
        row = chain["first_seen"].get(feature)
        if row:
            lines.append(
                f"- `{feature}`：首次见于 `{row['path']}`，该层计数 `{row['count']}`。"
            )
        else:
            lines.append(f"- `{feature}`：未在可读文本中发现。")
    lines.extend(["", "## 二进制静态反汇编", ""])
    if binary_evidence:
        lines.extend(f"- {line}" for line in binary_evidence)
    else:
        lines.append("- 未获得 SME/SVE 指令证据；检查 objdump 工具和目标文件路径。")
    lines.extend(["", "## dump中已有的反汇编文本", ""])
    existing = []
    for feature in ("mopa", "smstart", "smstop"):
        for row in chain["all_text_evidence"].get(feature, []):
            if "objdump" in row["path"].lower() or "disasm" in row["path"].lower():
                existing.append(
                    f"`{row['path']}`：`{feature}`文本计数 `{row['count']}`"
                )
    if existing:
        lines.extend(f"- {row}。" for row in existing)
        lines.append("- 这些文本由dump提供，仍应尽量从原始`.o/.so`重新反汇编交叉验证。")
    else:
        lines.append("- 未发现已有反汇编文本中的SME特征。")
    (output_dir / "05_sme_evidence.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_report(
    output_dir,
    case_name,
    dump_root,
    environment,
    test_result,
    inventory,
    chain,
    binaries,
    generic,
):
    package_versions = environment["packages"]
    numbered_count = len(chain["numbered_chain"])
    binary_with_sme = [
        row for row in binaries
        if sum(
            row["instruction_counts"].get(key, 0)
            for key in ("fmopa", "smopa", "umopa", "smstart", "smstop")
        )
    ]
    first_arm_sme = chain["first_seen"].get("arm_sme_mlir")
    first_llvm_sme = chain["first_seen"].get("llvm_sme")
    first_mopa = chain["first_seen"].get("mopa")
    direct_facts = [
        f"dump中共发现 `{len(inventory)}` 个文件。",
        f"发现 `{numbered_count}` 个带编号的MLIR编译阶段文件。",
        f"发现 `{len(binaries)}` 个可尝试分析的目标文件/共享库。",
        f"有 `{len(binary_with_sme)}` 个二进制包含SME相关静态指令证据。",
    ]
    if test_result.get("exit_code") is not None:
        direct_facts.append(
            f"现场测试退出码为 `{test_result['exit_code']}`，墙钟时间 "
            f"`{test_result.get('elapsed_seconds', 0):.6f}` 秒。"
        )

    def first_text(row):
        return f"`{row['path']}`" if row else "未发现"

    rows = [
        "# Linux现场编译链分析报告",
        "",
        f"- 用例：`{case_name}`",
        f"- dump：`{dump_root}`",
        f"- 生成时间：`{dt.datetime.now().astimezone().isoformat()}`",
        "",
        "## 1. 直接证据",
        "",
    ]
    rows.extend(f"- {fact}" for fact in direct_facts)
    rows.extend(
        [
            "",
            "## 2. 环境",
            "",
            f"- Python：`{environment['python_version']}`，路径 `{environment['collector_python']}`。",
            f"- PyTorch：`{package_versions.get('torch')}`。",
            f"- Triton：`{package_versions.get('triton')}`。",
            f"- pytest：`{package_versions.get('pytest')}`。",
            "- CPU、Cache、编译器和 perf 详细输出见 `00_environment.txt`。",
            "",
            "## 3. 测试命令",
            "",
            f"- 命令：`{test_result.get('command') or '未提供'}`。",
            f"- 工作目录：`{test_result.get('workdir') or '未提供'}`。",
            f"- 是否由工具执行：`{test_result.get('executed_by_collector', True)}`。",
            f"- 退出码：`{test_result.get('exit_code', '未执行')}`。",
            "- 完整输出见 `01_test_command.log`。",
            "",
            "## 4. 编译链里程碑",
            "",
            f"- ArmSME MLIR首次出现：{first_text(first_arm_sme)}。",
            f"- ArmSME属性/标记首次出现：{first_text(chain['first_seen'].get('arm_sme_attrs'))}。",
            f"- LLVM SME intrinsic（LLVM SME内建操作）首次出现：{first_text(first_llvm_sme)}。",
            f"- MOPA/FMOPA文本证据首次出现：{first_text(first_mopa)}。",
            "- 相邻阶段具体增删统计见 `04_ir_transitions.md`。",
            "",
            "## 5. 对文件角色的解释",
            "",
            "| 文件 | 推断角色 | 置信度 |",
            "|---|---|---|",
        ]
    )
    for row in inventory[:80]:
        rows.append(
            f"| `{row['path']}` | {row['role']} | `{row['role_confidence']}` |"
        )
    if len(inventory) > 80:
        rows.append(f"| ... | 其余{len(inventory) - 80}项见CSV | - |")
    rows.extend(
        [
            "",
            "## 6. 二进制证据",
            "",
        ]
    )
    if binary_with_sme:
        for row in binary_with_sme:
            counts = ", ".join(
                f"{key}={value}"
                for key, value in row["instruction_counts"].items()
                if value
            )
            rows.append(f"- `{row['path']}`：{counts}。")
    else:
        rows.append("- 当前未获得二进制SME指令证据，不能仅凭MLIR断言运行时执行了SME。")
    rows.extend(
        [
            "",
            "## 7. 启发式分析",
            "",
        ]
    )
    if generic and generic.get("available", True):
        rows.extend(
            [
                f"- 算子类别候选：`{generic.get('kernel_type')}`。",
                f"- 置信度：`{generic.get('type_confidence')}`。",
                f"- 最远可见阶段：`{generic.get('furthest_visible_stage')}`。",
                f"- 编译链覆盖率：`{generic.get('chain_coverage', {}).get('coverage_score')}%`。",
            ]
        )
        for gap in generic.get("chain_coverage", {}).get("gaps", []):
            rows.append(f"- 缺口：{gap}")
    else:
        rows.append("- 可选高层语义分析器未捆绑或未成功运行，查看 `07_generic_analysis.log`。")
    rows.extend(
        [
            "",
            "## 8. 现场必须确认的问题",
            "",
            "1. pytest插件或环境变量中，哪个选项打开了这些dump？",
            "2. 每个编号MLIR文件对应的准确Pass名称和命令行是什么？",
            "3. `kernel.o`与cache中的`.obj`是否为同一个目标文件？用SHA256确认。",
            "4. `_cpu_kernel_launcher.so`只负责调用，还是也包含SME计算内核？",
            "5. 计时是否包含JIT编译、缓存命中、输入生成和正确性检查？",
            "6. 测试重复运行时是否复用cache？冷启动和热启动必须分开记录。",
            "7. 目标CPU是否允许读取PMU，能否采集cycles、instructions和cache-misses？",
            "",
            "## 9. 结论边界",
            "",
            "- 文件名对应的生产阶段属于推断，必须向对方确认准确Pass。",
            "- IR中存在`arm_sme`只能证明编译表示到达该层，机器码必须由反汇编确认。",
            "- 二进制存在`fmopa`只能证明静态代码包含该指令，动态是否执行仍需运行或性能计数器。",
            "- 本报告不复制原始源码和IR；完整文件通过相对路径和SHA256在现场定位。",
            "",
            "## 10. 输出索引",
            "",
            "- `00_environment.*`：Linux、CPU、Cache、工具和Python包。",
            "- `01_test_command.*`：测试命令、退出码、时间和完整输出。",
            "- `02_artifact_inventory.*`：文件角色、大小和SHA256。",
            "- `03_ir_chain_analysis.json`：逐文件操作与特征统计。",
            "- `04_ir_transitions.md`：相邻层变化。",
            "- `05_sme_evidence.txt`：SME证据摘要。",
            "- `06_binary/`：file/readelf/nm/objdump输出。",
            "- `07_generic_analysis_summary.json`：不含源码片段的通用识别摘要。",
        ]
    )
    report = output_dir / "ONSITE_REPORT_CN.md"
    report.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return report


def write_manifest_and_package(output_dir, package):
    manifest_rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            manifest_rows.append(
                f"{sha256_file(path)}  {path.relative_to(output_dir)}"
            )
    (output_dir / "MANIFEST.sha256").write_text(
        "\n".join(manifest_rows) + "\n",
        encoding="utf-8",
    )
    if not package:
        return None
    archive = output_dir.parent / f"{output_dir.name}_report_bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output_dir, arcname=output_dir.name)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--case-name", default="onsite_case")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--test-command")
    parser.add_argument("--test-workdir", type=Path)
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Actually execute --test-command before analysis.",
    )
    parser.add_argument(
        "--max-text-mb",
        type=int,
        default=64,
        help="Maximum bytes scanned from each text/IR file.",
    )
    parser.add_argument(
        "--no-package",
        action="store_true",
        help="Do not create a report-only tar.gz bundle.",
    )
    args = parser.parse_args()

    if args.run_test and not args.test_command:
        parser.error("--run-test requires --test-command")
    if args.run_test and not args.test_workdir:
        parser.error("--run-test requires --test-workdir")
    if args.test_workdir and not args.test_workdir.is_dir():
        parser.error(f"test workdir not found: {args.test_workdir}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output.resolve()
        if args.output
        else (Path.cwd() / "onsite_results" / f"{args.case_name}-{stamp}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = collect_environment(output_dir)
    if args.run_test:
        test_result = execute_test(
            args.test_command, args.test_workdir.resolve(), output_dir
        )
    else:
        test_result = record_test_without_execution(
            args.test_command, args.test_workdir, output_dir
        )

    dump_root = args.dump.resolve()
    if not dump_root.is_dir():
        raise SystemExit(
            f"dump directory not found after test: {dump_root}\n"
            "请确认dump环境变量和测试工作目录。"
        )

    inventory = build_inventory(dump_root, output_dir)
    chain = analyze_ir_chain(
        dump_root, output_dir, args.max_text_mb * 1024 * 1024
    )
    binaries, binary_evidence = disassemble_binaries(
        dump_root, output_dir, inventory
    )
    write_sme_evidence(output_dir, chain, binary_evidence)
    generic = invoke_generic_analyzer(dump_root, output_dir)
    report = write_report(
        output_dir,
        args.case_name,
        dump_root,
        environment,
        test_result,
        inventory,
        chain,
        binaries,
        generic,
    )
    archive = write_manifest_and_package(output_dir, not args.no_package)
    print(report)
    if archive:
        print(archive)


if __name__ == "__main__":
    main()
