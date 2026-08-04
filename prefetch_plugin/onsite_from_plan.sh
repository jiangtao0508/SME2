#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 <llvm-install-prefix> <triton-shared-opt> <00_input.mlir> <output-dir> <PrefetchPlan.json> [decision-id]" >&2
  exit 2
fi

PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
PLAN="$(cd "$(dirname "$5")" && pwd)/$(basename "$5")"
if [[ $# -eq 6 ]]; then
  IFS=$'\t' read -r DISTANCE LOCALITY COVERAGE_LINES ISSUE_EVERY CACHE_LINE_BYTES DECISION_ID OBJECT_ID < <(
    python3 "$PLUGIN_DIR/prefetch_plan_options.py" "$PLAN" --decision-id "$6"
  )
else
  IFS=$'\t' read -r DISTANCE LOCALITY COVERAGE_LINES ISSUE_EVERY CACHE_LINE_BYTES DECISION_ID OBJECT_ID < <(
    python3 "$PLUGIN_DIR/prefetch_plan_options.py" "$PLAN"
  )
fi
if [[ -z "${DISTANCE:-}" || -z "${LOCALITY:-}" ]]; then
  echo "could not resolve PrefetchPlan plugin options" >&2
  exit 1
fi

echo "PrefetchPlan decision: $DECISION_ID object=$OBJECT_ID distance=$DISTANCE locality=$LOCALITY coverage-lines=$COVERAGE_LINES issue-every=$ISSUE_EVERY cache-line-bytes=$CACHE_LINE_BYTES"
bash "$PLUGIN_DIR/onsite_split_replay.sh" \
  "$1" "$2" "$3" "$4" gemm-rhs "$DISTANCE" "$LOCALITY" \
  "$COVERAGE_LINES" "$ISSUE_EVERY" "$CACHE_LINE_BYTES"
