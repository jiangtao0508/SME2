#!/usr/bin/env python3
"""Trace pytest collection, child processes, and compiler artifact creation."""

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


SOURCE_SUFFIXES = {
    ".py",
    ".cc",
    ".cpp",
    ".cxx",
    ".c",
    ".h",
    ".hpp",
    ".inc",
    ".td",
    ".cmake",
    ".sh",
}
ARTIFACT_SUFFIXES = {
    ".mlir",
    ".ll",
    ".llir",
    ".ir",
    ".ttir",
    ".ttsharedir",
    ".o",
    ".obj",
    ".so",
    ".a",
    ".json",
    ".ref",
    ".cxx",
}
FIXED_DUMP_TOKENS = {
    "after_transform_interpreter",
    "earase_schedule",
    "erase_schedule",
    "convert_to_llvm",
    "canonicalize",
    "strip_debug",
    "legalize_float8",
    "ttsharedir",
    "_cpu_kernel_launcher",
    "_triton_shared",
}
SELECTED_ENV_PREFIXES = (
    "FLAG",
    "TRITON",
    "MLIR",
    "LLVM",
    "TORCH",
    "PYTEST",
    "DUMP",
    "CACHE",
)
SELECTED_ENV_EXACT = {
    "PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CC",
    "CXX",
}
SECRET_WORDS = ("TOKEN", "PASSWORD", "SECRET", "CREDENTIAL", "PRIVATE_KEY")


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json_lines(path):
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redact_env(env):
    selected = {}
    for key, value in sorted(env.items()):
        if key not in SELECTED_ENV_EXACT and not key.startswith(SELECTED_ENV_PREFIXES):
            continue
        if any(word in key.upper() for word in SECRET_WORDS):
            selected[key] = "<redacted>"
        else:
            selected[key] = value
    return selected


def run_capture(argv, cwd, env, timeout=120):
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
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
            "output": output + "\n[trace error] {}\n".format(exc),
        }


def qualified_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = qualified_name(node.value)
        return "{}.{}".format(prefix, node.attr) if prefix else node.attr
    if isinstance(node, ast.Call):
        return qualified_name(node.func)
    return None


def literal_summary(node):
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [literal_summary(item) for item in node.elts[:16]]
        suffix = ", ..." if len(node.elts) > 16 else ""
        opening, closing = ("[", "]") if isinstance(node, ast.List) else ("(", ")")
        return opening + ", ".join(values) + suffix + closing
    name = qualified_name(node)
    return name or type(node).__name__


def decorator_summary(node):
    if isinstance(node, ast.Call):
        return {
            "name": qualified_name(node.func),
            "args": [literal_summary(arg) for arg in node.args],
            "keywords": {
                item.arg or "**": literal_summary(item.value)
                for item in node.keywords
            },
        }
    return {"name": qualified_name(node), "args": [], "keywords": {}}


def analyze_test_ast(workdir, nodeid):
    parts = nodeid.split("::")
    source = (workdir / parts[0]).resolve()
    function_name = parts[1].split("[", 1)[0] if len(parts) > 1 else None
    payload = {
        "nodeid": nodeid,
        "source": str(source),
        "source_exists": source.is_file(),
        "function": function_name,
        "sha256": None,
        "decorators": [],
        "calls": [],
        "with_contexts": [],
        "imports": [],
        "line_start": None,
        "line_end": None,
        "parse_error": None,
    }
    if not source.is_file():
        return payload
    payload["sha256"] = sha256_file(source)
    try:
        tree = ast.parse(source.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError) as exc:
        payload["parse_error"] = str(exc)
        return payload

    for item in tree.body:
        if isinstance(item, ast.Import):
            payload["imports"].extend(alias.name for alias in item.names)
        elif isinstance(item, ast.ImportFrom):
            module = item.module or ""
            payload["imports"].extend(
                "{}.{}".format(module, alias.name).strip(".")
                for alias in item.names
            )

    target = None
    for item in ast.walk(tree):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name == function_name:
                target = item
                break
    if target is None:
        return payload

    payload["line_start"] = target.lineno
    payload["line_end"] = getattr(target, "end_lineno", None)
    payload["decorators"] = [
        decorator_summary(item) for item in target.decorator_list
    ]
    calls = set()
    contexts = set()
    for item in ast.walk(target):
        if isinstance(item, ast.Call):
            name = qualified_name(item.func)
            if name:
                calls.add(name)
        elif isinstance(item, (ast.With, ast.AsyncWith)):
            for context in item.items:
                name = qualified_name(context.context_expr)
                if name:
                    contexts.add(name)
    payload["calls"] = sorted(calls)
    payload["with_contexts"] = sorted(contexts)
    return payload


def snapshot_files(roots):
    rows = {}
    for root in roots:
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            try:
                candidates = list(root.rglob("*"))
            except OSError:
                candidates = []
        else:
            candidates = []
        for path in candidates:
            try:
                if not path.is_file():
                    continue
                stat = path.stat()
            except OSError:
                continue
            rows[str(path.resolve())] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    return rows


class FileTimeline:
    def __init__(self, roots, interval_seconds, epoch):
        self.roots = roots
        self.interval_seconds = interval_seconds
        self.epoch = epoch
        self.baseline = snapshot_files(roots)
        self.previous = dict(self.baseline)
        self.events = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self._sample()

    def _sample(self):
        current = snapshot_files(self.roots)
        now = time.time()
        for path, state in current.items():
            prior = self.previous.get(path)
            if prior is None:
                kind = "created"
            elif prior != state:
                kind = "modified"
            else:
                continue
            self.events.append(
                {
                    "seconds_from_start": now - self.epoch,
                    "event": kind,
                    "path": path,
                    "size": state["size"],
                    "mtime_ns": state["mtime_ns"],
                }
            )
        for path, state in self.previous.items():
            if path not in current:
                self.events.append(
                    {
                        "seconds_from_start": now - self.epoch,
                        "event": "deleted",
                        "path": path,
                        "size": state["size"],
                        "mtime_ns": state["mtime_ns"],
                    }
                )
        self.previous = current

    def _run(self):
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(self.interval_seconds)


def read_proc_process(pid):
    base = Path("/proc") / str(pid)
    try:
        cmdline = (base / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
        status_text = (base / "status").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return None
    ppid_match = re.search(r"^PPid:\s+(\d+)", status_text, re.MULTILINE)
    try:
        executable = os.readlink(str(base / "exe"))
    except OSError:
        executable = None
    try:
        child_text = (
            base / "task" / str(pid) / "children"
        ).read_text(encoding="utf-8", errors="replace")
        children = [int(value) for value in child_text.split()]
    except OSError:
        children = []
    return {
        "pid": pid,
        "ppid": int(ppid_match.group(1)) if ppid_match else None,
        "command": cmdline,
        "executable": executable,
        "children": children,
    }


class ProcessTimeline:
    def __init__(self, root_pid, interval_seconds, epoch):
        self.root_pid = root_pid
        self.interval_seconds = interval_seconds
        self.epoch = epoch
        self.rows = {}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self._sample()

    def _sample(self):
        if not Path("/proc").is_dir():
            return
        now = time.time() - self.epoch
        pending = [self.root_pid]
        visited = set()
        while pending:
            pid = pending.pop()
            if pid in visited:
                continue
            visited.add(pid)
            row = read_proc_process(pid)
            if row is None:
                continue
            pending.extend(row.pop("children"))
            existing = self.rows.get(pid)
            if existing is None:
                row["first_seen_seconds"] = now
                row["last_seen_seconds"] = now
                self.rows[pid] = row
            else:
                existing.update(row)
                existing["last_seen_seconds"] = now

    def _run(self):
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(self.interval_seconds)


def pump_output(pipe, log_handle):
    try:
        for line in iter(pipe.readline, ""):
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
    finally:
        pipe.close()


def probe_strace(workdir, env, output_dir):
    tool = shutil.which("strace")
    if not tool:
        return {"available": False, "usable": False, "reason": "strace not found"}
    probe_path = output_dir / "strace_probe.txt"
    result = run_capture(
        [tool, "-o", str(probe_path), "-e", "trace=process", "true"],
        workdir,
        env,
        timeout=20,
    )
    return {
        "available": True,
        "usable": result["exit_code"] == 0,
        "path": tool,
        "exit_code": result["exit_code"],
        "reason": result["output"].strip(),
    }


def decode_strace_string(value):
    try:
        return bytes(value, "utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return value


def relevant_artifact(path, watch_roots):
    candidate = Path(path)
    if candidate.suffix.lower() in ARTIFACT_SUFFIXES:
        return True
    if candidate.name in {"main.cxx", "_cpu_kernel_launcher.so"}:
        return True
    absolute = str(candidate)
    return any(absolute.startswith(str(root)) for root in watch_roots)


def parse_strace(trace_dir, workdir, watch_roots):
    exec_events = []
    write_events = []
    command_by_pid = {}
    trace_files = sorted(trace_dir.glob("trace.*"))
    if not trace_files and (trace_dir / "trace").is_file():
        trace_files = [trace_dir / "trace"]
    for trace_file in trace_files:
        match = re.search(r"\.(\d+)$", trace_file.name)
        pid = int(match.group(1)) if match else None
        current_cwd = workdir
        try:
            lines = trace_file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            continue
        for line in lines:
            timestamp_match = re.match(r"^(\d+\.\d+)\s+", line)
            timestamp = (
                float(timestamp_match.group(1)) if timestamp_match else None
            )
            quoted = [
                decode_strace_string(item)
                for item in re.findall(r'"((?:\\.|[^"\\])*)"', line)
            ]
            if "chdir(" in line and quoted and line.rstrip().endswith("= 0"):
                target = Path(quoted[0])
                current_cwd = (
                    target if target.is_absolute() else (current_cwd / target)
                ).resolve()
            if "execve(" in line and quoted:
                event = {
                    "timestamp_epoch": timestamp,
                    "pid": pid,
                    "executable": quoted[0],
                    "raw": line[:2000],
                }
                exec_events.append(event)
                if pid is not None:
                    command_by_pid[pid] = line[:2000]
                continue

            syscall = None
            destination = None
            if re.search(r"\b(?:open|openat|creat)\(", line):
                if not re.search(r"O_WRONLY|O_RDWR|O_CREAT|O_TRUNC|O_APPEND", line):
                    continue
                syscall = line.split("(", 1)[0].split()[-1]
                if quoted:
                    destination = quoted[0]
            elif re.search(r"\b(?:rename|renameat|renameat2)\(", line):
                syscall = line.split("(", 1)[0].split()[-1]
                if len(quoted) >= 2:
                    destination = quoted[-1]
            if not syscall or not destination:
                continue
            path = Path(destination)
            resolved = path if path.is_absolute() else current_cwd / path
            resolved = resolved.resolve()
            if not relevant_artifact(str(resolved), watch_roots):
                continue
            write_events.append(
                {
                    "timestamp_epoch": timestamp,
                    "pid": pid,
                    "syscall": syscall,
                    "path": str(resolved),
                    "producer_exec": command_by_pid.get(pid),
                    "raw": line[:2000],
                }
            )
    return {
        "trace_files": [str(path) for path in trace_files],
        "exec_events": exec_events,
        "write_events": write_events,
    }


def derive_scan_tokens(watch_roots):
    tokens = set(FIXED_DUMP_TOKENS)
    for root in watch_roots:
        if not root.is_dir():
            continue
        try:
            files = list(root.rglob("*"))
        except OSError:
            continue
        for path in files:
            if not path.is_file():
                continue
            stem = re.sub(r"^\d+_", "", path.stem)
            if len(stem) >= 5:
                tokens.add(stem)
    return sorted(tokens)


def scan_sources(scan_roots, tokens):
    matches = []
    lowered = [(token, token.lower()) for token in tokens]
    for root in scan_roots:
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            try:
                if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
                    continue
                if ".git" in path.parts or path.stat().st_size > 8 * 1024 * 1024:
                    continue
                lines = path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, 1):
                line_lower = line.lower()
                found = [token for token, low in lowered if low in line_lower]
                if found:
                    matches.append(
                        {
                            "file": str(path.resolve()),
                            "line": number,
                            "tokens": found,
                        }
                    )
                    if len(matches) >= 1000:
                        return matches
    return matches


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def artifact_role(path):
    name = Path(path).name.lower()
    if name == "00_input.mlir":
        return "初始MLIR输入"
    if "transform_interpreter" in name:
        return "Transform Dialect执行后"
    if "erase_schedule" in name or "earase_schedule" in name:
        return "删除调度描述后"
    if "convert_to_llvm" in name:
        return "转到LLVM方言后"
    if "canonicalize" in name:
        return "规范化后"
    if "strip_debug" in name:
        return "移除调试信息后"
    if "legalize_float8" in name:
        return "Float8合法化后"
    if name.endswith(".ttir"):
        return "Triton TTIR"
    if name.endswith(".ttsharedir"):
        return "Triton共享布局IR"
    if name.endswith((".llir", ".ll")):
        return "LLVM IR"
    if name.endswith((".o", ".obj")):
        return "目标文件"
    if name.endswith(".so"):
        return "共享库"
    if name == "main.cxx":
        return "C++驱动/启动器源码"
    return "其他产物"


def generate_report(
    output_dir,
    args,
    ast_info,
    collect_result,
    plugin_lines,
    command,
    run_result,
    processes,
    file_events,
    strace_info,
    source_matches,
    analyzer_result,
    runtime_dispatch,
):
    direct_writes = strace_info.get("write_events", [])
    created = [row for row in file_events if row["event"] == "created"]
    calls = set(ast_info.get("calls", []))
    rows = [
        "# pytest到中间文件完整追踪报告",
        "",
        "## 1. 本次测试身份",
        "",
        "- Node ID（pytest用例唯一标识）：`{}`。".format(args.nodeid),
        "- 测试文件：`{}`。".format(ast_info.get("source")),
        "- 测试文件SHA256：`{}`。".format(ast_info.get("sha256")),
        "- 测试函数：`{}`，源码行 `{}-{}`。".format(
            ast_info.get("function"),
            ast_info.get("line_start"),
            ast_info.get("line_end"),
        ),
        "- pytest收集退出码：`{}`；插件注册记录 `{}` 条。".format(
            collect_result.get("exit_code"), len(plugin_lines)
        ),
        "- 证据类型：pytest收集结果和文件哈希属于直接证据；AST只证明源码中存在调用。",
        "",
        "## 2. 源码静态调用",
        "",
    ]
    if ast_info.get("decorators"):
        for decorator in ast_info["decorators"]:
            rows.append(
                "- 装饰器：`{}`，参数 `{}`。".format(
                    decorator.get("name"), decorator.get("args")
                )
            )
    for call in ast_info.get("calls", []):
        rows.append("- 调用：`{}`。".format(call))
    if "torch.bmm" in calls:
        rows.append(
            "- 源码中存在`torch.bmm`；它最终是否被FlagGems重定向，必须结合"
            "`use_gems`上下文、注册日志和运行时子进程证明。"
        )
    rows.extend(
        [
            "",
            "## 3. FlagGems运行时分发",
            "",
        ]
    )
    states = [
        item for item in runtime_dispatch
        if item.get("event") == "flaggems_state"
    ]
    if states:
        for item in states:
            bmm = item.get("flaggems_bmm") or {}
            rows.extend(
                [
                    "- 阶段：`{}`；vendor：`{}`；device：`{}`。".format(
                        item.get("stage"),
                        item.get("vendor_name"),
                        item.get("device"),
                    ),
                    "- 实际FlagGems BMM：`{}.{}`，源码 `{}`。".format(
                        bmm.get("module"),
                        bmm.get("qualname"),
                        bmm.get("source"),
                    ),
                    "- `aten::bmm`完整分发表保存在`02b_runtime_dispatch.json`。",
                ]
            )
    else:
        rows.append(
            "- 未获得FlagGems运行时状态。可能是测试未导入FlagGems、使用了不同入口，"
            "或对方pytest未加载探针；查看`02b_runtime_dispatch.json`。"
        )
    rows.extend(
        [
            "",
            "## 4. pytest实际执行",
            "",
            "- 命令：`{}`。".format(shlex.join(command)),
            "- 工作目录：`{}`。".format(args.workdir.resolve()),
            "- 退出码：`{}`，追踪墙钟时间：`{:.6f}`秒。".format(
                run_result.get("exit_code"),
                run_result.get("elapsed_seconds", 0.0),
            ),
            "- 完整输出：`03_pytest_run.log`。",
            "- 追踪会引入明显开销，本次时间不能作为性能结果。",
            "",
            "## 5. 子进程与编译器命令",
            "",
            "| PID | PPID | 首次出现/s | 可执行文件 | 命令 |",
            "|---:|---:|---:|---|---|",
        ]
    )
    for item in processes[:80]:
        rows.append(
            "| {} | {} | {:.6f} | `{}` | `{}` |".format(
                item.get("pid"),
                item.get("ppid"),
                item.get("first_seen_seconds", 0.0),
                item.get("executable"),
                (item.get("command") or "").replace("|", "\\|")[:500],
            )
        )
    rows.extend(
        [
            "",
            "## 6. 文件生成时间线",
            "",
            "- 轮询观察到新建文件 `{}` 个，全部事件 `{}` 条。".format(
                len(created), len(file_events)
            ),
            "| 相对时间/s | 事件 | 文件 | 推断角色 | 大小 |",
            "|---:|---|---|---|---:|",
        ]
    )
    for item in file_events[:160]:
        rows.append(
            "| {:.6f} | `{}` | `{}` | {} | {} |".format(
                item["seconds_from_start"],
                item["event"],
                item["path"],
                artifact_role(item["path"]),
                item["size"],
            )
        )
    rows.extend(
        [
            "",
            "## 7. 文件生产者",
            "",
        ]
    )
    if direct_writes:
        rows.extend(
            [
                "`strace`记录到以下文件写入。它能证明某个PID打开/重命名了文件，"
                "但一次`openat`不等价于该进程独自完成全部内容。",
                "",
                "| 时间戳 | PID | 系统调用 | 文件 | 最近execve命令 |",
                "|---:|---:|---|---|---|",
            ]
        )
        for item in direct_writes[:200]:
            rows.append(
                "| {} | {} | `{}` | `{}` | `{}` |".format(
                    item.get("timestamp_epoch"),
                    item.get("pid"),
                    item.get("syscall"),
                    item.get("path"),
                    (item.get("producer_exec") or "未捕获").replace("|", "\\|")[:500],
                )
            )
    else:
        rows.append(
            "- 没有获得`strace`直接写入证据。只能依靠文件时间线和进程时间重叠作"
            "相关性判断，不能准确声称某个进程生成了某个文件。"
        )
    rows.extend(
        [
            "",
            "## 8. dump命名代码位置",
            "",
            "- 源码扫描命中 `{}` 个“文件名/阶段名”位置。扫描只保存文件、行号和"
            "关键词，不复制源码。".format(len(source_matches)),
        ]
    )
    for item in source_matches[:120]:
        rows.append(
            "- `{}` 第 `{}` 行：`{}`。".format(
                item["file"], item["line"], "`, `".join(item["tokens"])
            )
        )
    rows.extend(
        [
            "",
            "## 9. 当前能建立的因果链",
            "",
            "1. `pytest --collect-only/--trace-config`确认测试函数、参数实例和插件。",
            "2. AST确认测试源码中的`torch.bmm`、上下文管理器和辅助函数调用。",
            "3. pytest探针确认`use_gems`内实际后端函数和`aten::bmm`分发表。",
            "4. `/proc`时间线确认测试运行时启动的Python、编译器、链接器和启动器。",
            "5. 文件轮询确认各IR/对象文件的出现顺序。",
            "6. `strace execve/openat/rename`把文件写入关联到具体进程和命令。",
            "7. dump文件名在源码中的命中位置用于定位真正的dump实现。",
            "8. `artifact_analysis/`继续分析各层IR操作变化和最终SME机器码证据。",
            "",
            "## 10. 仍需谨慎的地方",
            "",
            "- 文件名只能提示阶段。准确Pass名称必须由编译器源码、Pass调试日志或"
            "开发者说明确认。",
            "- `torch.bmm`源码调用不自动证明FlagGems内核被执行；需要注册/分发证据。",
            "- IR中出现ArmSME操作不自动证明机器码或动态执行；还需对象反汇编和PMU。",
            "- `strace`、轮询和首次JIT都会干扰时间，性能测试必须另跑无追踪版本。",
            "- 若对方不允许保存系统调用或反汇编，报告只能留在现场并使用"
            "`--no-package`。",
            "",
            "## 11. 输出索引",
            "",
            "- `00_preflight.json`：Python、环境、strace能力和命令。",
            "- `01_pytest_collect.log/json`：pytest收集结果与插件。",
            "- `02_test_ast.json`：测试函数的静态结构，不含源码片段。",
            "- `02b_runtime_dispatch.json`：FlagGems后端函数与PyTorch分发表。",
            "- `03_pytest_run.log/json`：真实测试输出与退出状态。",
            "- `04_process_timeline.json`：子进程和命令。",
            "- `05_file_timeline.json/csv`：文件创建、修改和删除时间。",
            "- `06_strace/`：可选系统调用原始证据。",
            "- `07_producer_map.json/csv`：文件写入与PID关联。",
            "- `08_source_name_matches.json`：dump命名在源码中的位置。",
            "- `artifact_analysis/`：IR层次和二进制分析。",
        ]
    )
    if analyzer_result:
        rows.append(
            "- 后续IR分析器退出码：`{}`。".format(
                analyzer_result.get("exit_code")
            )
        )
    report = output_dir / "PYTEST_PIPELINE_REPORT_CN.md"
    report.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--nodeid", required=True)
    parser.add_argument("--watch-root", action="append", type=Path, required=True)
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--scan-root", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pytest-arg", action="append", default=[])
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument(
        "--strace",
        choices=("auto", "off", "required"),
        default="auto",
    )
    parser.add_argument("--interval-ms", type=int, default=50)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Required safety acknowledgement before executing pytest.",
    )
    args = parser.parse_args()

    if not args.run:
        parser.error("add --run after checking workdir, nodeid and watch roots")
    if not args.workdir.is_dir():
        parser.error("workdir not found: {}".format(args.workdir))
    if args.interval_ms < 10:
        parser.error("--interval-ms must be at least 10")

    env = dict(os.environ)
    for assignment in args.env:
        if "=" not in assignment:
            parser.error("--env requires NAME=VALUE")
        key, value = assignment.split("=", 1)
        env[key] = value

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output.resolve()
        if args.output
        else (Path.cwd() / "onsite_results" / "pytest_trace_{}".format(stamp)).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir = args.workdir.resolve()
    watch_roots = [path.resolve() for path in args.watch_root]
    scan_roots = [path.resolve() for path in args.scan_root] or [workdir]
    interval = args.interval_ms / 1000.0

    probe_log = output_dir / "02b_runtime_dispatch.jsonl"
    env["SME2_PYTEST_PROBE_LOG"] = str(probe_log)
    tools_dir = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = tools_dir + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    pytest_base = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "onsite_pytest_probe",
    ]
    collect_command = pytest_base + [
        "--trace-config",
        "--collect-only",
        "-q",
    ] + args.pytest_arg + [args.nodeid]
    collect_result = run_capture(collect_command, workdir, env, timeout=300)
    (output_dir / "01_pytest_collect.log").write_text(
        collect_result["output"], encoding="utf-8"
    )
    plugin_lines = [
        line.strip()
        for line in collect_result["output"].splitlines()
        if "PLUGIN registered:" in line or "active plugins:" in line.lower()
    ]
    collect_payload = dict(collect_result)
    collect_payload["plugin_lines"] = plugin_lines
    write_json(output_dir / "01_pytest_collect.json", collect_payload)

    ast_info = analyze_test_ast(workdir, args.nodeid)
    write_json(output_dir / "02_test_ast.json", ast_info)
    runtime_dispatch = read_json_lines(probe_log)
    write_json(output_dir / "02b_runtime_dispatch.json", runtime_dispatch)
    if collect_result["exit_code"] != 0:
        report = output_dir / "PYTEST_PIPELINE_REPORT_CN.md"
        report.write_text(
            "# pytest收集失败\n\n"
            "- Node ID：`{}`\n"
            "- 工作目录：`{}`\n"
            "- 退出码：`{}`\n"
            "- 为避免运行错误或错误用例，工具没有进入测试执行阶段。\n"
            "- 请查看`01_pytest_collect.log`修正路径、参数顺序、"
            "Python环境或插件问题。\n".format(
                args.nodeid, workdir, collect_result["exit_code"]
            ),
            encoding="utf-8",
        )
        print(report)
        raise SystemExit(collect_result["exit_code"] or 2)

    strace_probe = (
        {
            "available": bool(shutil.which("strace")),
            "usable": False,
            "reason": "disabled by --strace off",
        }
        if args.strace == "off"
        else probe_strace(workdir, env, output_dir)
    )
    if args.strace == "required" and not strace_probe["usable"]:
        raise SystemExit("strace required but unusable: {}".format(strace_probe))
    use_strace = args.strace != "off" and strace_probe["usable"]
    command = pytest_base + ["-s"] + args.pytest_arg + [args.nodeid]
    actual_command = list(command)
    trace_dir = output_dir / "06_strace"
    if use_strace:
        trace_dir.mkdir(parents=True, exist_ok=True)
        actual_command = [
            strace_probe["path"],
            "-ff",
            "-ttt",
            "-T",
            "-s",
            "512",
            "-o",
            str(trace_dir / "trace"),
            "-e",
            "trace=process,file",
        ] + command

    preflight = {
        "generated_utc": utc_now(),
        "collector_python": sys.executable,
        "workdir": str(workdir),
        "nodeid": args.nodeid,
        "watch_roots": [str(path) for path in watch_roots],
        "scan_roots": [str(path) for path in scan_roots],
        "pytest_command": command,
        "actual_command": actual_command,
        "environment": redact_env(env),
        "strace": strace_probe,
        "strace_enabled": use_strace,
    }
    write_json(output_dir / "00_preflight.json", preflight)

    epoch = time.time()
    file_timeline = FileTimeline(watch_roots, interval, epoch)
    file_timeline.start()
    log_path = output_dir / "03_pytest_run.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            actual_command,
            cwd=str(workdir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        process_timeline = ProcessTimeline(process.pid, interval, epoch)
        process_timeline.start()
        pump = threading.Thread(
            target=pump_output,
            args=(process.stdout, log_handle),
            daemon=True,
        )
        pump.start()
        try:
            exit_code = process.wait()
        except KeyboardInterrupt:
            process.send_signal(2)
            exit_code = process.wait()
        pump.join(timeout=10)
        process_timeline.stop()
    file_timeline.stop()
    elapsed = time.perf_counter() - started

    run_result = {
        "command": command,
        "actual_command": actual_command,
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "started_epoch": epoch,
        "finished_epoch": time.time(),
        "strace_enabled": use_strace,
    }
    write_json(output_dir / "03_pytest_run.json", run_result)
    runtime_dispatch = read_json_lines(probe_log)
    write_json(output_dir / "02b_runtime_dispatch.json", runtime_dispatch)

    processes = sorted(
        process_timeline.rows.values(),
        key=lambda row: (row["first_seen_seconds"], row["pid"]),
    )
    write_json(output_dir / "04_process_timeline.json", processes)

    file_events = sorted(
        file_timeline.events,
        key=lambda row: (row["seconds_from_start"], row["path"]),
    )
    write_json(output_dir / "05_file_timeline.json", file_events)
    write_csv(
        output_dir / "05_file_timeline.csv",
        file_events,
        ("seconds_from_start", "event", "path", "size", "mtime_ns"),
    )

    strace_info = (
        parse_strace(trace_dir, workdir, watch_roots)
        if use_strace
        else {"trace_files": [], "exec_events": [], "write_events": []}
    )
    write_json(output_dir / "07_producer_map.json", strace_info)
    write_csv(
        output_dir / "07_producer_map.csv",
        strace_info["write_events"],
        ("timestamp_epoch", "pid", "syscall", "path", "producer_exec", "raw"),
    )

    tokens = derive_scan_tokens(watch_roots)
    source_matches = scan_sources(scan_roots, tokens)
    write_json(
        output_dir / "08_source_name_matches.json",
        {"tokens": tokens, "matches": source_matches},
    )

    analyzer_result = None
    dump_root = args.dump.resolve() if args.dump else None
    analyzer = Path(__file__).with_name("analyze_onsite_case.py")
    if dump_root and dump_root.is_dir() and analyzer.is_file():
        analyzer_command = [
            sys.executable,
            str(analyzer),
            "--dump",
            str(dump_root),
            "--case-name",
            "pytest_{}".format(ast_info.get("function") or "case"),
            "--output",
            str(output_dir / "artifact_analysis"),
            "--test-workdir",
            str(workdir),
            "--test-command",
            shlex.join(command),
            "--no-package",
        ]
        analyzer_result = run_capture(
            analyzer_command, workdir, env, timeout=600
        )
        (output_dir / "09_artifact_analyzer.log").write_text(
            analyzer_result["output"], encoding="utf-8"
        )

    report = generate_report(
        output_dir,
        args,
        ast_info,
        collect_result,
        plugin_lines,
        command,
        run_result,
        processes,
        file_events,
        strace_info,
        source_matches,
        analyzer_result,
        runtime_dispatch,
    )
    print("\n{}".format(report))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
