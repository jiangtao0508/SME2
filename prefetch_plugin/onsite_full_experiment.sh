#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 <llvm-install-prefix> <triton-shared-opt> <compiler.py> <00_input.mlir> <output-dir> [PrefetchPlan.json]" >&2
  exit 2
fi

PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
LLVM_INSTALL_DIR="$(cd "$1" && pwd)"
TRITON_SHARED_OPT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
COMPILER_PY="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
ORIGINAL_INPUT="$(cd "$(dirname "$4")" && pwd)/$(basename "$4")"
mkdir -p "$5"
OUTPUT_DIR="$(cd "$5" && pwd)"
SUMMARY="$OUTPUT_DIR/ONSITE_PREFETCH_RESULT.txt"
CURRENT_STEP="initialization"

if [[ "$OUTPUT_DIR" == *" "* ]]; then
  echo "output directory must not contain spaces: $OUTPUT_DIR" >&2
  exit 1
fi

on_error() {
  STATUS=$?
  {
    echo "FAIL: onsite prefetch experiment"
    echo "failed_step: $CURRENT_STEP"
    echo "exit_status: $STATUS"
    echo "inspect: $OUTPUT_DIR/$CURRENT_STEP.log"
  } > "$SUMMARY"
  echo "FAIL at $CURRENT_STEP; inspect $SUMMARY" >&2
  exit "$STATUS"
}
trap on_error ERR

CURRENT_STEP="01_preflight"
bash "$PLUGIN_DIR/onsite_preflight.sh" \
  "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" "$OUTPUT_DIR/preflight" \
  2>&1 | tee "$OUTPUT_DIR/$CURRENT_STEP.log"

CURRENT_STEP="02_roundtrip"
bash "$PLUGIN_DIR/onsite_split_replay.sh" \
  "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" "$ORIGINAL_INPUT" \
  "$OUTPUT_DIR/roundtrip" roundtrip \
  2>&1 | tee "$OUTPUT_DIR/$CURRENT_STEP.log"

CURRENT_STEP="03_prefetch"
if [[ $# -eq 6 ]]; then
  PLAN="$(cd "$(dirname "$6")" && pwd)/$(basename "$6")"
  bash "$PLUGIN_DIR/onsite_from_plan.sh" \
    "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" "$ORIGINAL_INPUT" \
    "$OUTPUT_DIR/prefetch" "$PLAN" \
    2>&1 | tee "$OUTPUT_DIR/$CURRENT_STEP.log"
  CONFIGURATION="PrefetchPlan=$PLAN"
else
  bash "$PLUGIN_DIR/onsite_split_replay.sh" \
    "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" "$ORIGINAL_INPUT" \
    "$OUTPUT_DIR/prefetch" gemm-rhs 4 3 \
    2>&1 | tee "$OUTPUT_DIR/$CURRENT_STEP.log"
  CONFIGURATION="distance=4 locality=3"
fi

CURRENT_STEP="04_object"
bash "$PLUGIN_DIR/onsite_stage2.sh" \
  "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" "$COMPILER_PY" \
  "$OUTPUT_DIR/prefetch/01_gemm_rhs_prefetch.mlir" \
  "$OUTPUT_DIR/final" \
  2>&1 | tee "$OUTPUT_DIR/$CURRENT_STEP.log"

PREFETCH_IR_COUNT="$(grep -c 'llvm.intr.prefetch' "$OUTPUT_DIR/prefetch/01_gemm_rhs_prefetch.mlir" || true)"
SME_IR_COUNT="$(grep -c 'arm_sme.intr.mopa' "$OUTPUT_DIR/prefetch/01_gemm_rhs_prefetch.mlir" || true)"
PRFM_COUNT="$(grep -c 'prfm' "$OUTPUT_DIR/final/kernel.disasm" || true)"
FMOPA_COUNT="$(grep -c 'fmopa' "$OUTPUT_DIR/final/kernel.disasm" || true)"

trap - ERR
{
  echo "PASS: onsite prefetch experiment"
  echo "configuration: $CONFIGURATION"
  echo "llvm.intr.prefetch: $PREFETCH_IR_COUNT"
  echo "arm_sme.intr.mopa: $SME_IR_COUNT"
  echo "AArch64 PRFM: $PRFM_COUNT"
  echo "AArch64 FMOPA: $FMOPA_COUNT"
  echo "bufferized_snapshot: $OUTPUT_DIR/prefetch/bufferized_before_sme.mlir"
  echo "prefetched_ir: $OUTPUT_DIR/prefetch/01_gemm_rhs_prefetch.mlir"
  echo "llir_override_candidate: $OUTPUT_DIR/final/kernel.llir"
  echo "disassembly: $OUTPUT_DIR/final/kernel.disasm"
} | tee "$SUMMARY"

echo "summary: $SUMMARY"
