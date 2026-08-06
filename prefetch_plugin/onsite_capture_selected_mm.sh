#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
  echo "usage: $0 <triton-cpu-root> <output-parent> [M=8192] [N=2048] [K=64] [dtype=bfloat16]" >&2
  exit 2
fi

TRITON_ROOT="$(cd "$1" && pwd)"
mkdir -p "$2"
OUTPUT_PARENT="$(cd "$2" && pwd)"
M="${3:-8192}"
N="${4:-2048}"
K="${5:-64}"
DTYPE="${6:-bfloat16}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$OUTPUT_PARENT/mm-${M}x${K}x${N}-$(date +%Y%m%d-%H%M%S)-$$"
SELECT_DIR="$RUN_DIR/selection"
CAPTURE_DIR="$RUN_DIR/selected-dump"
mkdir -p "$SELECT_DIR" "$CAPTURE_DIR"

if [[ ! -d "$TRITON_ROOT/FlagGems/src/flag_gems" || ! -d "$TRITON_ROOT/python/triton" ]]; then
  echo "triton-cpu root must contain FlagGems/ and python/triton/" >&2
  exit 1
fi

export GEMS_VENDOR=kunpeng
export TRITON_USE_SHARED_BACKEND=1
export TRITON_SHARED_FORCE_SME_PIPELINE=1
export TRITON_ALWAYS_COMPILE=1
export MLIR_ENABLE_DUMP=1
export TRITON_PRINT_AUTOTUNING=1
export PYTHONPATH="$TRITON_ROOT/python:$TRITON_ROOT/FlagGems/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[1/3] select the native Kunpeng MM config for the target shape"
unset TRITON_SHARED_DUMP_PATH
if ! python3 "$SCRIPT_DIR/onsite_mm_select.py" select \
    --m "$M" --n "$N" --k "$K" --dtype "$DTYPE" \
    --config-json "$SELECT_DIR/selected_config.json" \
    >"$SELECT_DIR/select.log" 2>&1; then
  cat "$SELECT_DIR/select.log" >&2
  exit 1
fi
cat "$SELECT_DIR/select.log"

echo "[2/3] force only the selected config in a fresh process and capture its lowering"
export TRITON_SHARED_DUMP_PATH="$CAPTURE_DIR"
if ! python3 "$SCRIPT_DIR/onsite_mm_select.py" capture \
    --m "$M" --n "$N" --k "$K" --dtype "$DTYPE" \
    --config-json "$SELECT_DIR/selected_config.json" \
    >"$CAPTURE_DIR/capture.log" 2>&1; then
  cat "$CAPTURE_DIR/capture.log" >&2
  exit 1
fi
cat "$CAPTURE_DIR/capture.log"

echo "[3/3] verify that exactly one native MM dump group was captured"
INPUT_COUNT="$(find "$CAPTURE_DIR" -type f -name 00_input.mlir -print | wc -l | tr -d ' ')"
if [[ "$INPUT_COUNT" -ne 1 ]]; then
  echo "expected exactly one selected MM 00_input.mlir, found $INPUT_COUNT" >&2
  exit 1
fi
MM_00="$(find "$CAPTURE_DIR" -type f -name 00_input.mlir -print -quit)"
MM_DIR="$(dirname "$MM_00")"
for name in tt.mlir ttshared.mlir 00_input.mlir 01_after_transform_interpreter.mlir; do
  if [[ ! -f "$MM_DIR/$name" ]]; then
    echo "missing selected MM stage: $name" >&2
    exit 1
  fi
done
if ! grep -Eq 'linalg\.(matmul|matmul_transpose_a)' "$MM_DIR/00_input.mlir"; then
  echo "selected dump does not contain an MM linalg payload" >&2
  exit 1
fi

python3 - "$SELECT_DIR/selected_config.json" "$MM_DIR" "$RUN_DIR" <<'PY'
import json
import pathlib
import re
import sys

selection_path = pathlib.Path(sys.argv[1])
mm_dir = pathlib.Path(sys.argv[2])
run_dir = pathlib.Path(sys.argv[3])
selection = json.loads(selection_path.read_text(encoding="utf-8"))
patterns = {
    "tt_dot": r"\btt\.dot\b",
    "linalg_matmul": r"\blinalg\.(?:matmul|matmul_transpose_a)\b",
    "scf_for": r"\bscf\.for\b",
    "memref_alloc": r"\bmemref\.alloc\b",
    "memref_copy": r"\bmemref\.copy\b",
    "vector_transfer_read": r"\bvector\.transfer_read\b",
    "arm_sme_mopa": r"\b(?:arm_sme|llvm\.aarch64\.sme)[A-Za-z0-9_.$-]*mopa[A-Za-z0-9_.$-]*\b",
    "prefetch": r"\b(?:memref\.prefetch|llvm\.intr\.prefetch|llvm\.prefetch)\b",
}
stage_counts = {}
for path in sorted(mm_dir.glob("*.mlir")):
    text = path.read_text(encoding="utf-8")
    stage_counts[path.name] = {
        "lines": len(text.splitlines()),
        **{key: len(re.findall(pattern, text)) for key, pattern in patterns.items()},
    }
summary = {
    "schema": "selected-kunpeng-mm-dump-v1",
    "selection": selection,
    "mm_dump_directory": str(mm_dir),
    "stages": sorted(path.name for path in mm_dir.iterdir() if path.is_file()),
    "stage_counts": stage_counts,
}
(run_dir / "MM_CAPTURE.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
selected = selection["selected"]
meta = selected["meta"]
print(
    "MM_SELECTED"
    f" BLOCK_M={meta.get('BLOCK_M')}"
    f" BLOCK_N={meta.get('BLOCK_N')}"
    f" BLOCK_K={meta.get('BLOCK_K')}"
    f" num_warps={selected.get('num_warps')}"
    f" num_stages={selected.get('num_stages')}"
)
keys = ("lines", *patterns)
for name, counts in stage_counts.items():
    values = " ".join(f"{key}={counts[key]}" for key in keys)
    print(f"MM_IR {name} {values}")
PY

echo "RUN_DIR=$RUN_DIR"
echo "SELECTED_CONFIG=$SELECT_DIR/selected_config.json"
echo "MM_DUMP_DIR=$MM_DIR"
echo "MM_00=$MM_DIR/00_input.mlir"
echo "SUMMARY=$RUN_DIR/MM_CAPTURE.json"
