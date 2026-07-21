#!/usr/bin/env python3
"""Print a short Chinese evidence card from an onsite trace directory."""

import argparse
import json
import re
import sys
from pathlib import Path


STAGE_ORDER = (
    "00_input.mlir",
    "01_after_transform_interpreter.mlir",
    "02_after_earase_schedule.mlir",
    "02_after_erase_schedule.mlir",
    "03_after_convert_to_llvm.mlir",
    "04_after_canonicalize.mlir",
    "05_after_strip_debug.mlir",
    "06_after_legalize_float8.mlir",
    "tt.mlir",
    "ttshared.mlir",
    "ll.mlir",
    "ll.ir",
    "bmm_kernel.ttir",
    "bmm_kernel.ttsharedir",
    "bmm_kernel.llir",
    "kernel.o",
    "bmm_kernel.obj",
    "_cpu_kernel_launcher.so",
)
COMPILER_RE = re.compile(
    r"(?:clang|gcc|g\+\+|cc1|\bld\b|lld|mlir|llvm|triton|compiler|launcher)",
    re.IGNORECASE,
)


def load_json(path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return default


def find_latest(root):
    if root.is_dir() and any(
        (root / name).is_file()
        for name in ("PYTEST_PIPELINE_REPORT_CN.md", "ONSITE_REPORT_CN.md")
    ):
        return root.resolve()
    pipeline_candidates = []
    analyzer_candidates = []
    if root.is_dir():
        for report in root.rglob("PYTEST_PIPELINE_REPORT_CN.md"):
            try:
                pipeline_candidates.append(
                    (report.stat().st_mtime_ns, report.parent)
                )
            except OSError:
                continue
        if not pipeline_candidates:
            for report in root.rglob("ONSITE_REPORT_CN.md"):
                try:
                    analyzer_candidates.append(
                        (report.stat().st_mtime_ns, report.parent)
                    )
                except OSError:
                    continue
    candidates = pipeline_candidates or analyzer_candidates
    return (
        max(candidates, key=lambda item: item[0])[1].resolve()
        if candidates
        else None
    )


def status(exit_code):
    if exit_code is None:
        return "未知"
    return "成功" if exit_code == 0 else "失败({})".format(exit_code)


def basename(path):
    return Path(path).name if path else "未知"


def choose_dispatch_state(events):
    states = [row for row in events if row.get("event") == "flaggems_state"]
    for stage in ("after_use_gems_enter", "after_collection"):
        matches = [row for row in states if row.get("stage") == stage]
        if matches:
            return matches[-1]
    return states[-1] if states else None


def process_candidates(processes):
    rows = []
    for item in processes:
        text = "{} {}".format(
            item.get("executable") or "", item.get("command") or ""
        )
        if COMPILER_RE.search(text):
            rows.append(item)
    return rows


def producer_for_path(path, writes):
    matches = [row for row in writes if row.get("path") == path]
    if not matches:
        same_name = [
            row for row in writes
            if basename(row.get("path")) == basename(path)
        ]
        matches = same_name
    return matches[0] if matches else None


def producer_name(writer, processes_by_pid):
    if not writer:
        return None
    pid = writer.get("pid")
    process = processes_by_pid.get(pid, {})
    executable = process.get("executable")
    if executable:
        return basename(executable)
    raw = writer.get("producer_exec") or ""
    match = re.search(r'execve\("([^"]+)"', raw)
    return basename(match.group(1)) if match else "进程名未捕获"


def collect_stage_rows(events, inventory, writes, processes_by_pid):
    first_events = {}
    for item in sorted(events, key=lambda row: row.get("seconds_from_start", 0.0)):
        name = basename(item.get("path"))
        if name in STAGE_ORDER and name not in first_events:
            first_events[name] = item
    inventory_paths = {}
    for item in inventory:
        name = basename(item.get("path"))
        if name in STAGE_ORDER and name not in inventory_paths:
            inventory_paths[name] = item.get("path")

    rows = []
    seen = set()
    for name in STAGE_ORDER:
        if name in seen:
            continue
        seen.add(name)
        event = first_events.get(name)
        if event:
            path = event.get("path")
            writer = producer_for_path(path, writes)
            rows.append(
                {
                    "name": name,
                    "time": event.get("seconds_from_start"),
                    "event": event.get("event"),
                    "pid": writer.get("pid") if writer else None,
                    "producer": producer_name(writer, processes_by_pid),
                    "direct": writer is not None,
                }
            )
        elif name in inventory_paths:
            path = inventory_paths[name]
            writer = producer_for_path(path, writes)
            rows.append(
                {
                    "name": name,
                    "time": None,
                    "event": "运行前已存在/轮询未捕获",
                    "pid": writer.get("pid") if writer else None,
                    "producer": producer_name(writer, processes_by_pid),
                    "direct": writer is not None,
                }
            )
    return rows


def summarize(result):
    preflight = load_json(result / "00_preflight.json", {})
    collect = load_json(result / "01_pytest_collect.json", {})
    ast_info = load_json(result / "02_test_ast.json", {})
    dispatch = load_json(result / "02b_runtime_dispatch.json", [])
    run = load_json(result / "03_pytest_run.json", {})
    if not run:
        run = load_json(result / "01_test_command.json", {})
    processes = load_json(result / "04_process_timeline.json", [])
    events = load_json(result / "05_file_timeline.json", [])
    producers = load_json(result / "07_producer_map.json", {})
    source_matches = load_json(result / "08_source_name_matches.json", {})
    artifact = (
        result / "artifact_analysis"
        if (result / "artifact_analysis").is_dir()
        else result
    )
    inventory = load_json(artifact / "02_artifact_inventory.json", [])
    chain = load_json(artifact / "03_ir_chain_analysis.json", {})
    binaries = load_json(artifact / "06_binary" / "binary_summary.json", [])

    writes = producers.get("write_events", [])
    processes_by_pid = {row.get("pid"): row for row in processes}
    stages = collect_stage_rows(events, inventory, writes, processes_by_pid)
    state = choose_dispatch_state(dispatch)
    compiler_rows = process_candidates(processes)
    calls = set(ast_info.get("calls", []))
    contexts = set(ast_info.get("with_contexts", []))
    first_seen = chain.get("first_seen", {})

    sme_binaries = []
    for item in binaries:
        counts = item.get("instruction_counts", {})
        selected = {
            key: value for key, value in counts.items()
            if key in {"fmopa", "smopa", "umopa", "smstart", "smstop"} and value
        }
        if selected:
            sme_binaries.append((item.get("path"), selected))

    checks = {
        "pytest收集": collect.get("exit_code") == 0,
        "测试运行": run.get("exit_code") == 0,
        "测试源码": bool(ast_info.get("source_exists")),
        "torch.bmm调用": "torch.bmm" in calls,
        "FlagGems分发": bool(
            state and state.get("stage") == "after_use_gems_enter"
        ),
        "文件时间线": bool(events),
        "文件生产者": bool(writes),
        "IR层次": bool(chain.get("numbered_chain")),
        "SME二进制": bool(sme_binaries),
    }
    score = sum(checks.values())

    lines = [
        "================ 现场结论卡 ================",
        "结果目录：{}".format(result),
        "用例：{}".format(ast_info.get("nodeid") or preflight.get("nodeid") or "未知"),
        "pytest：收集{}；运行{}；追踪耗时{:.3f}s（不能作性能数据）".format(
            status(collect.get("exit_code")),
            status(run.get("exit_code")),
            run.get("elapsed_seconds", 0.0),
        ),
        "源码入口：torch.bmm={}；use_gems={}".format(
            "有" if "torch.bmm" in calls else "未发现",
            "有" if any("use_gems" in item for item in contexts) else "未发现",
        ),
    ]

    if state:
        function = state.get("flaggems_bmm") or {}
        lines.append(
            "实际分发：vendor={}，device={}，BMM={}.{} ({})".format(
                state.get("vendor_name"),
                state.get("device"),
                function.get("module"),
                function.get("qualname"),
                basename(function.get("source")),
            )
        )
    else:
        lines.append("实际分发：未捕获，请看02b_runtime_dispatch.json")

    lines.append(
        "追踪能力：strace={}；直接文件写入={}条；源码命名命中={}处".format(
            "启用" if preflight.get("strace_enabled") else "未启用/不可用",
            len(writes),
            len(source_matches.get("matches", [])),
        )
    )
    if compiler_rows:
        rendered = []
        for item in compiler_rows[:5]:
            rendered.append(
                "PID{}:{}".format(
                    item.get("pid"), basename(item.get("executable"))
                )
            )
        lines.append("编译相关进程：{}".format("，".join(rendered)))
    else:
        lines.append("编译相关进程：未看到独立外部编译器，可能在Python/JIT进程内部")

    lines.append("--- 关键文件生成顺序 ---")
    if stages:
        for item in stages[:14]:
            if item["time"] is None:
                time_text = "时间未知"
            else:
                time_text = "+{:.3f}s".format(item["time"])
            if item["direct"]:
                evidence = "PID{}:{}（strace直接证据）".format(
                    item["pid"], item["producer"]
                )
            else:
                evidence = "生产者未直接证明"
            lines.append(
                "{}  {}  {}".format(time_text, item["name"], evidence)
            )
    else:
        lines.append("没有识别到关键文件；检查--watch-root和--dump路径")

    lines.append("--- SME证据 ---")
    for key, label in (
        ("arm_sme_mlir", "ArmSME MLIR首次出现"),
        ("llvm_sme", "LLVM SME首次出现"),
        ("mopa", "MOPA文本首次出现"),
    ):
        row = first_seen.get(key)
        lines.append(
            "{}：{}".format(label, row.get("path") if row else "未发现")
        )
    if sme_binaries:
        for path, counts in sme_binaries[:5]:
            count_text = "，".join(
                "{}={}".format(key, value) for key, value in counts.items()
            )
            lines.append("机器码：{}，{}".format(path, count_text))
    else:
        lines.append("机器码：未发现SME静态指令，或反汇编工具不可用")

    missing = [name for name, passed in checks.items() if not passed]
    lines.extend(
        [
            "--- 完整度 ---",
            "证据完整度：{}/{}".format(score, len(checks)),
            "仍缺：{}".format("、".join(missing) if missing else "无"),
            "说明：有strace标记的生产者是直接证据，其余只作线索。",
            "============================================",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        help="Specific pytest_trace result directory.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("onsite_results"),
        help="Search root when --result is omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional onsite-only text output; stdout is always used.",
    )
    args = parser.parse_args()

    result = find_latest(args.result or args.root)
    if result is None:
        raise SystemExit(
            "没有找到现场报告；请用--result指定pytest_trace或分析结果目录。"
        )
    text = summarize(result)
    sys.stdout.write(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
