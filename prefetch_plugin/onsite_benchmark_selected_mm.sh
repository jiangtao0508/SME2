#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <triton-cpu-root> <selected-config.json> <override-llir> <output-dir>" >&2
  exit 2
fi

TRITON_ROOT="$(cd "$1" && pwd)"
SELECTED_CONFIG="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
LLIR_FILE="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
mkdir -p "$4"
OUTPUT_DIR="$(cd "$4" && pwd)"
RUN_DIR="$OUTPUT_DIR/mm-bench-$(date +%Y%m%d-%H%M%S)-$$"
mkdir "$RUN_DIR"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -f "$SELECTED_CONFIG" || ! -f "$LLIR_FILE" ]]; then
  echo "selected config or LLIR file does not exist" >&2
  exit 1
fi

read -r M N K DTYPE < <(python3 - "$SELECTED_CONFIG" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
shape = data["shape"]
print(shape["M"], shape["N"], shape["K"], data["dtype"])
PY
)

export GEMS_VENDOR=kunpeng
export TRITON_USE_SHARED_BACKEND=1
export TRITON_SHARED_FORCE_SME_PIPELINE=1
export TRITON_ALWAYS_COMPILE=1
export PYTHONPATH="$TRITON_ROOT/python:$TRITON_ROOT/FlagGems/src${PYTHONPATH:+:$PYTHONPATH}"
WARMUP="${SME_BENCH_WARMUP:-5}"
REP="${SME_BENCH_REP:-20}"
ORDER="${SME_BENCH_ORDER:-baseline-first}"
if [[ "$ORDER" != baseline-first && "$ORDER" != prefetch-first ]]; then
  echo "SME_BENCH_ORDER must be baseline-first or prefetch-first" >&2
  exit 2
fi

PROBE_ROOT="$RUN_DIR/probe"
PROBE_CACHE="$RUN_DIR/probe-cache"
if ! (
  unset TRITON_KERNEL_OVERRIDE TRITON_OVERRIDE_DIR
  export TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$PROBE_ROOT" TRITON_CACHE_DIR="$PROBE_CACHE"
  python3 "$SCRIPT_DIR/onsite_mm_select.py" capture \
    --m "$M" --n "$N" --k "$K" --dtype "$DTYPE" --config-json "$SELECTED_CONFIG"
) >"$RUN_DIR/probe.log" 2>&1; then
  cat "$RUN_DIR/probe.log" >&2
  exit 1
fi

mapfile -t HASH_DIRS < <(
  find "$PROBE_ROOT" -mindepth 2 -maxdepth 2 -type f \
    -name 'mm_kernel_general.*' -exec dirname {} \; | sort -u
)
if [[ ${#HASH_DIRS[@]} -ne 1 ]]; then
  echo "expected one selected-MM source hash, found ${#HASH_DIRS[@]}" >&2
  exit 1
fi
SRC_HASH="$(basename "${HASH_DIRS[0]}")"
OVERRIDE_ROOT="$RUN_DIR/override"
mkdir -p "$OVERRIDE_ROOT/$SRC_HASH"
OVERRIDE_NAME="$(sed -n 's/^define[[:space:]]*void[[:space:]]*@\([A-Za-z0-9_]*\).*/\1/p' "$LLIR_FILE" | head -1)"
OVERRIDE_NAME="${OVERRIDE_NAME:-mm_kernel_general}"
cp "$LLIR_FILE" "$OVERRIDE_ROOT/$SRC_HASH/$OVERRIDE_NAME.llir"
echo "MM_SRC_HASH=$SRC_HASH"
echo "MM_OVERRIDE_PAYLOAD=$OVERRIDE_ROOT/$SRC_HASH/$OVERRIDE_NAME.llir"

VERIFY_LOG="$RUN_DIR/prefetch-correctness.log"
if ! (
  unset TRITON_KERNEL_DUMP TRITON_DUMP_DIR
  export TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR="$OVERRIDE_ROOT"
  export TRITON_CACHE_DIR="$RUN_DIR/cache-correctness"
  python3 "$SCRIPT_DIR/onsite_mm_select.py" verify \
    --m "$M" --n "$N" --k "$K" --dtype "$DTYPE" --config-json "$SELECTED_CONFIG"
) >"$VERIFY_LOG" 2>&1; then
  cat "$VERIFY_LOG" >&2
  exit 1
fi
grep -Fq 'Overriding kernel with file' "$VERIFY_LOG"
grep -Fq 'MM_CORRECTNESS=PASS' "$VERIFY_LOG"
echo "MM_CORRECTNESS=PASS"

run_case() {
  local label=$1
  local log="$RUN_DIR/$label.log"
  if ! (
    unset TRITON_KERNEL_DUMP TRITON_DUMP_DIR
    export TRITON_CACHE_DIR="$RUN_DIR/cache-$ORDER-$label"
    if [[ "$label" == baseline ]]; then
      unset TRITON_KERNEL_OVERRIDE TRITON_OVERRIDE_DIR
    else
      export TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR="$OVERRIDE_ROOT"
    fi
    python3 "$SCRIPT_DIR/onsite_mm_select.py" benchmark \
      --m "$M" --n "$N" --k "$K" --dtype "$DTYPE" --config-json "$SELECTED_CONFIG" \
      --warmup "$WARMUP" --rep "$REP" --label "$label"
  ) >"$log" 2>&1; then
    cat "$log" >&2
    return 1
  fi
  cat "$log"
}

echo "MM_RUN_ORDER=$ORDER"
if [[ "$ORDER" == baseline-first ]]; then
  run_case baseline
  run_case prefetch
else
  run_case prefetch
  run_case baseline
fi

if grep -Fq 'Overriding kernel with file' "$RUN_DIR/baseline.log"; then
  echo "native MM baseline unexpectedly used an override" >&2
  exit 1
fi
if ! grep -Fq 'Overriding kernel with file' "$RUN_DIR/prefetch.log"; then
  echo "selected MM prefetch run did not use the override" >&2
  exit 1
fi

BASELINE_MS="$(sed -n 's/^MM_BASELINE_LATENCY_MS=//p' "$RUN_DIR/baseline.log" | tail -1)"
PREFETCH_MS="$(sed -n 's/^MM_PREFETCH_LATENCY_MS=//p' "$RUN_DIR/prefetch.log" | tail -1)"
python3 - "$BASELINE_MS" "$PREFETCH_MS" <<'PY'
import sys
baseline, prefetch = map(float, sys.argv[1:])
print("MM_RESULT_OVERRIDE_HIT=1")
print(f"MM_RESULT_BASELINE_MS={baseline:.6f}")
print(f"MM_RESULT_PREFETCH_MS={prefetch:.6f}")
print(f"MM_RESULT_SPEEDUP={baseline / prefetch:.6f}x")
print(f"MM_RESULT_LATENCY_CHANGE_PERCENT={(prefetch / baseline - 1) * 100:+.3f}%")
PY
echo "MM_BENCH_DIR=$RUN_DIR"
