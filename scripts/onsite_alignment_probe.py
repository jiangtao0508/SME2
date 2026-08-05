#!/usr/bin/env python3
"""Print a confidentiality-conscious build and MLIR structure fingerprint."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


STAGES = {
    "tt": ("tt.mlir", "bmm_kernel.ttir"),
    "ttshared": ("ttshared.mlir", "bmm_kernel.ttsharedir"),
    "00": ("00_input.mlir",),
    "01": ("01_after_transform_interpreter.mlir",),
    "02": ("02_after_erase_schedule.mlir", "02_after_earase_schedule.mlir"),
    "03": ("03_after_convert_to_llvm.mlir",),
    "bufferized": ("bufferized_before_sme.mlir",),
}

EXACT_OPS = {
    "func": "func.func",
    "for": "scf.for",
    "parallel": "scf.parallel",
    "if": "scf.if",
    "matmul": "linalg.matmul",
    "bmm": "linalg.batch_matmul",
    "generic": "linalg.generic",
    "alloc": "memref.alloc",
    "alloca": "memref.alloca",
    "load": "memref.load",
    "store": "memref.store",
    "copy": "memref.copy",
    "subview": "memref.subview",
    "vread": "vector.transfer_read",
    "vwrite": "vector.transfer_write",
    "contract": "vector.contract",
    "ttload": "tt.load",
    "ttdot": "tt.dot",
}


def run(*command: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def git_value(root: Path, expression: str) -> str:
    value = run("git", "rev-parse", expression, cwd=root)
    return value[:12] if value != "unavailable" else value


def git_dirty(root: Path, relative_path: str | None = None) -> str:
    command = ["git", "status", "--porcelain", "--untracked-files=no"]
    if relative_path:
        command.extend(("--", relative_path))
    value = run(*command, cwd=root)
    if value == "unavailable":
        return value
    return "1" if value else "0"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def count_exact(text: str, operation: str) -> int:
    boundary = r"[A-Za-z0-9_.$-]"
    return len(re.findall(rf"(?<!{boundary}){re.escape(operation)}(?!{boundary})", text))


def structure_line(stage: str, path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    counts = {label: count_exact(text, operation) for label, operation in EXACT_OPS.items()}
    counts.update(
        sme=len(re.findall(r"(?<![A-Za-z0-9_.$-])arm_sme\.", text)),
        tptr=len(re.findall(r"(?<![A-Za-z0-9_.$-])tptr\.", text)),
        transform=len(re.findall(r"(?<![A-Za-z0-9_.$-])transform\.", text)),
        llvm=len(re.findall(r"(?<![A-Za-z0-9_.$-])llvm\.", text)),
    )
    fields = " ".join(f"{name}={value}" for name, value in counts.items())
    return f"IR {stage} lines={text.count(chr(10)) + 1} {fields}"


def find_stage_files(dump_root: Path, names: tuple[str, ...]) -> list[Path]:
    matches: set[Path] = set()
    for name in names:
        matches.update(path for path in dump_root.rglob(name) if path.is_file())
    return sorted(matches)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report public source revisions and numeric MLIR structure only."
    )
    parser.add_argument("llvm_source", type=Path)
    parser.add_argument("triton_cpu_source", type=Path)
    parser.add_argument("dump_directory", type=Path)
    args = parser.parse_args()

    for path in (args.llvm_source, args.triton_cpu_source, args.dump_directory):
        if not path.is_dir():
            parser.error(f"not a directory: {path}")

    print("ALIGNMENT_PROBE_V1")
    print(
        f"HOST system={platform.system()} machine={platform.machine()} "
        f"python={platform.python_version()}"
    )
    print(
        f"PACKAGES torch={package_version('torch')} triton={package_version('triton')} "
        f"flag_gems={package_version('flag-gems')}"
    )
    print(
        f"LLVM commit={git_value(args.llvm_source, 'HEAD')} "
        f"dirty={git_dirty(args.llvm_source)}"
    )
    print(
        f"TRITON commit={git_value(args.triton_cpu_source, 'HEAD')} "
        f"dirty={git_dirty(args.triton_cpu_source)} "
        f"shared_tree={git_value(args.triton_cpu_source, 'HEAD:triton-shared')} "
        f"shared_dirty={git_dirty(args.triton_cpu_source, 'triton-shared')} "
        f"flag_gems_tree={git_value(args.triton_cpu_source, 'HEAD:FlagGems')} "
        f"flag_gems_dirty={git_dirty(args.triton_cpu_source, 'FlagGems')}"
    )
    environment_names = (
        "TRITON_USE_SHARED_BACKEND",
        "TRITON_SHARED_FORCE_SME_PIPELINE",
        "TRITON_DISABLE_LINE_INFO",
        "TRITON_ALWAYS_COMPILE",
        "MLIR_ENABLE_DUMP",
        "GEMS_VENDOR",
    )
    print(
        "ENV "
        + " ".join(f"{name}={os.environ.get(name, 'unset')}" for name in environment_names)
    )

    found = 0
    for stage, names in STAGES.items():
        matches = find_stage_files(args.dump_directory, names)
        if len(matches) == 1:
            print(structure_line(stage, matches[0]))
            found += 1
        elif matches:
            print(f"IR {stage} ambiguous_matches={len(matches)}")
        else:
            print(f"IR {stage} missing=1")
    if found == 0:
        print("ERROR no uniquely identifiable IR stage files", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
