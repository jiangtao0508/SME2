#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 10 ]]; then
  echo "usage: $0 <llvm-install-prefix> <triton-shared-opt> <00_input.mlir> <output-dir> <mode> [distance] [locality] [coverage-lines] [issue-every] [cache-line-bytes]" >&2
  echo "mode: snapshot | roundtrip | gemm-rhs" >&2
  exit 2
fi

LLVM_INSTALL_DIR="$(cd "$1" && pwd)"
TRITON_SHARED_OPT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
ORIGINAL_INPUT="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$4"
OUTPUT_DIR="$(cd "$4" && pwd)"
MODE="$5"
DISTANCE="${6:-4}"
LOCALITY="${7:-3}"
COVERAGE_LINES="${8:-1}"
ISSUE_EVERY="${9:-1}"
CACHE_LINE_BYTES="${10:-64}"
MLIR_OPT="$LLVM_INSTALL_DIR/bin/mlir-opt"

if [[ "$MODE" != "snapshot" && "$MODE" != "roundtrip" && "$MODE" != "gemm-rhs" ]]; then
  echo "mode must be snapshot, roundtrip, or gemm-rhs" >&2
  exit 1
fi
if [[ "$OUTPUT_DIR" == *" "* ]]; then
  echo "output directory must not contain spaces: $OUTPUT_DIR" >&2
  exit 1
fi
if ! [[ "$DISTANCE" =~ ^[1-9][0-9]*$ ]]; then
  echo "distance must be a positive integer: $DISTANCE" >&2
  exit 1
fi
if ! [[ "$LOCALITY" =~ ^[0-3]$ ]]; then
  echo "locality must be 0, 1, 2, or 3: $LOCALITY" >&2
  exit 1
fi
for positive_option in "$COVERAGE_LINES" "$ISSUE_EVERY" "$CACHE_LINE_BYTES"; do
  if ! [[ "$positive_option" =~ ^[1-9][0-9]*$ ]]; then
    echo "coverage-lines, issue-every, and cache-line-bytes must be positive integers" >&2
    exit 1
  fi
done

PREFIX_INPUT="$OUTPUT_DIR/00_prefix_to_bufferize.mlir"
PREFIX_WITH_SCHEDULE="$OUTPUT_DIR/01_prefix_with_schedule.mlir"
BUFFERIZED="$OUTPUT_DIR/bufferized_before_sme.mlir"

python3 "$PLUGIN_DIR/split_transform_replay.py" prefix \
  "$ORIGINAL_INPUT" "$PREFIX_INPUT"
"$TRITON_SHARED_OPT" "$PREFIX_INPUT" \
  --mlir-disable-threading \
  --transform-interpreter \
  -o "$PREFIX_WITH_SCHEDULE"
"$TRITON_SHARED_OPT" "$PREFIX_WITH_SCHEDULE" \
  --test-transform-dialect-erase-schedule \
  -o "$BUFFERIZED"

grep -q 'scf.for' "$BUFFERIZED"
grep -q 'vector.transfer_read' "$BUFFERIZED"

if [[ "$MODE" == "snapshot" ]]; then
  echo "PASS: captured bufferized payload: $BUFFERIZED"
  exit 0
fi

RESUME_INPUT="$OUTPUT_DIR/00_resume_after_prefetch.mlir"
FINAL_01="$OUTPUT_DIR/01_gemm_rhs_prefetch.mlir"

if [[ "$MODE" == "roundtrip" ]]; then
  BASELINE_INPUT="$OUTPUT_DIR/00_baseline_tptr_preload.mlir"
  BASELINE_01="$OUTPUT_DIR/01_baseline_full_schedule.mlir"
  FINAL_01="$OUTPUT_DIR/01_roundtrip_no_prefetch.mlir"
  python3 "$PLUGIN_DIR/split_transform_replay.py" preload \
    "$ORIGINAL_INPUT" "$BASELINE_INPUT"
  "$TRITON_SHARED_OPT" "$BASELINE_INPUT" \
    --mlir-disable-threading \
    --transform-interpreter \
    -o "$BASELINE_01"
  python3 "$PLUGIN_DIR/split_transform_replay.py" resume \
    "$ORIGINAL_INPUT" "$BUFFERIZED" "$RESUME_INPUT"
  "$TRITON_SHARED_OPT" "$RESUME_INPUT" \
    --mlir-disable-threading \
    --transform-interpreter \
    -o "$FINAL_01"

  BASELINE_SME_COUNT="$(grep -c 'arm_sme.intr.mopa' "$BASELINE_01" || true)"
  ROUNDTRIP_SME_COUNT="$(grep -c 'arm_sme.intr.mopa' "$FINAL_01" || true)"
  ROUNDTRIP_PREFETCH_COUNT="$(grep -c 'llvm.intr.prefetch' "$FINAL_01" || true)"
  if [[ "$BASELINE_SME_COUNT" -eq 0 || "$BASELINE_SME_COUNT" -ne "$ROUNDTRIP_SME_COUNT" ]]; then
    echo "round-trip changed the ArmSME MOPA count" >&2
    echo "baseline=$BASELINE_SME_COUNT roundtrip=$ROUNDTRIP_SME_COUNT" >&2
    exit 1
  fi
  if [[ "$ROUNDTRIP_PREFETCH_COUNT" -ne 0 ]]; then
    echo "round-trip unexpectedly contains LLVM prefetch operations" >&2
    exit 1
  fi
  echo "PASS: no-prefetch split round-trip preserved arm_sme.intr.mopa=$ROUNDTRIP_SME_COUNT"
  echo "baseline: $BASELINE_01"
  echo "roundtrip: $FINAL_01"
  exit 0
fi

if [[ ${SME_SKIP_PLUGIN_BUILD:-0} != 1 ]]; then
  bash "$PLUGIN_DIR/build_and_smoke_mlir_opt.sh" "$LLVM_INSTALL_DIR"
fi

PLUGIN_LIBRARY=""
for candidate in \
  "$PLUGIN_DIR/build/PrefetchPassPlugin.so" \
  "$PLUGIN_DIR/build/PrefetchPassPlugin.dylib"; do
  if [[ -f "$candidate" ]]; then
    PLUGIN_LIBRARY="$candidate"
    break
  fi
done
if [[ -z "$PLUGIN_LIBRARY" ]]; then
  echo "plugin library was not produced" >&2
  exit 1
fi

PREFETCHED="$OUTPUT_DIR/bufferized_before_sme.prefetch.mlir"
"$MLIR_OPT" \
  --load-pass-plugin="$PLUGIN_LIBRARY" \
  --pass-pipeline="builtin.module(builtin.module(func.func(prefetch-gemm-rhs{distance=$DISTANCE locality=$LOCALITY coverage-lines=$COVERAGE_LINES issue-every=$ISSUE_EVERY cache-line-bytes=$CACHE_LINE_BYTES})))" \
  "$BUFFERIZED" \
  -o "$PREFETCHED"
grep -q 'memref.prefetch' "$PREFETCHED"

python3 "$PLUGIN_DIR/split_transform_replay.py" resume \
  "$ORIGINAL_INPUT" "$PREFETCHED" "$RESUME_INPUT"
"$TRITON_SHARED_OPT" "$RESUME_INPUT" \
  --mlir-disable-threading \
  --transform-interpreter \
  -o "$FINAL_01"

PREFETCH_COUNT="$(grep -c 'llvm.intr.prefetch' "$FINAL_01" || true)"
SME_COUNT="$(grep -c 'arm_sme.intr.mopa' "$FINAL_01" || true)"
if [[ "$PREFETCH_COUNT" -eq 0 || "$SME_COUNT" -eq 0 ]]; then
  echo "expected LLVM prefetch and ArmSME MOPA after resume" >&2
  echo "llvm.intr.prefetch=$PREFETCH_COUNT arm_sme.intr.mopa=$SME_COUNT" >&2
  exit 1
fi

echo "PASS: split replay completed"
echo "PASS: llvm.intr.prefetch=$PREFETCH_COUNT arm_sme.intr.mopa=$SME_COUNT"
echo "configuration: distance=$DISTANCE locality=$LOCALITY coverage-lines=$COVERAGE_LINES issue-every=$ISSUE_EVERY cache-line-bytes=$CACHE_LINE_BYTES"
echo "next: run onsite_stage2.sh with $FINAL_01"
