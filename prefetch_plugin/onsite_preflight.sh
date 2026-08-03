#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <llvm-install-prefix> <triton-shared-opt> [output-dir]" >&2
  exit 2
fi

LLVM_INSTALL_DIR="$(cd "$1" && pwd)"
TRITON_SHARED_OPT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_ARG="${3:-$PLUGIN_DIR/build/preflight}"
mkdir -p "$OUTPUT_ARG"
OUTPUT_DIR="$(cd "$OUTPUT_ARG" && pwd)"
MLIR_OPT="$LLVM_INSTALL_DIR/bin/mlir-opt"
REPORT="$OUTPUT_DIR/preflight_report.txt"

for required in "$MLIR_OPT" "$TRITON_SHARED_OPT"; do
  if [[ ! -x "$required" ]]; then
    echo "required executable is missing: $required" >&2
    exit 1
  fi
done

"$MLIR_OPT" --version > "$OUTPUT_DIR/mlir-opt.version.txt"
"$TRITON_SHARED_OPT" --version > "$OUTPUT_DIR/triton-shared-opt.version.txt"
"$TRITON_SHARED_OPT" --help > "$OUTPUT_DIR/triton-shared-opt.help.txt"

for required_option in \
  'transform-interpreter' \
  'test-transform-dialect-erase-schedule' \
  'tptr-to-llvm'; do
  if ! grep -q -- "$required_option" "$OUTPUT_DIR/triton-shared-opt.help.txt"; then
    echo "triton-shared-opt is missing required option/pass: $required_option" >&2
    exit 1
  fi
done

MLIR_VERSION="$(grep -Eo 'LLVM version [0-9]+\.[0-9]+\.[0-9]+' "$OUTPUT_DIR/mlir-opt.version.txt" | head -1 || true)"
TRITON_VERSION="$(grep -Eo 'LLVM version [0-9]+\.[0-9]+\.[0-9]+' "$OUTPUT_DIR/triton-shared-opt.version.txt" | head -1 || true)"
if [[ -z "$MLIR_VERSION" || -z "$TRITON_VERSION" ]]; then
  echo "could not parse LLVM versions; inspect $OUTPUT_DIR/*.version.txt" >&2
  exit 1
fi
if [[ "$MLIR_VERSION" != "$TRITON_VERSION" ]]; then
  echo "LLVM version mismatch: mlir-opt=$MLIR_VERSION triton-shared-opt=$TRITON_VERSION" >&2
  exit 1
fi

bash "$PLUGIN_DIR/build_and_smoke_mlir_opt.sh" "$LLVM_INSTALL_DIR" \
  > "$OUTPUT_DIR/mlir_plugin_smoke.txt" 2>&1

{
  echo "PASS: onsite preflight"
  echo "mlir-opt: $MLIR_OPT"
  echo "triton-shared-opt: $TRITON_SHARED_OPT"
  echo "version: $MLIR_VERSION"
  echo "PASS: triton-shared-opt provides Transform Interpreter, schedule erase, and tptr-to-llvm"
  echo "PASS: mlir-opt loaded PrefetchPassPlugin and lowered prefetch to AArch64 PRFM"
  echo "NOTE: this intentionally does not require triton-shared-opt to register the custom pass"
  if grep -q -- 'load-pass-plugin' "$OUTPUT_DIR/triton-shared-opt.help.txt"; then
    echo "INFO: triton-shared-opt exposes --load-pass-plugin, but split replay does not use it"
  fi
} | tee "$REPORT"

echo "report: $REPORT"
