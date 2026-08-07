#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <llvm-install-prefix>" >&2
  exit 2
fi

LLVM_INSTALL_DIR="$(cd "$1" && pwd)"
PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="$PLUGIN_DIR/build/gemm_rhs_pipeline.mlir"

bash "$PLUGIN_DIR/build_and_smoke_mlir_opt.sh" "$LLVM_INSTALL_DIR" >/dev/null
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

"$LLVM_INSTALL_DIR/bin/mlir-opt" \
  --load-pass-plugin="$PLUGIN_LIBRARY" \
  --pass-pipeline='builtin.module(builtin.module(func.func(pipeline-gemm-rhs-load)))' \
  "$PLUGIN_DIR/test/gemm_rhs_pipeline.mlir" \
  -o "$OUTPUT"

grep -q 'prefetch.explicit_pipeline_count = 2 : i64' "$OUTPUT"
PIPELINE_LOOP_COUNT="$(grep -c 'prefetch.explicit_pipeline}' "$OUTPUT" || true)"
if [[ "$PIPELINE_LOOP_COUNT" -ne 1 ]]; then
  echo "expected one software-pipelined loop, found $PIPELINE_LOOP_COUNT" >&2
  exit 1
fi
if grep -q 'memref.prefetch' "$OUTPUT"; then
  echo "explicit load pipeline unexpectedly emitted memref.prefetch" >&2
  exit 1
fi

# The old type fixture stores into the packed allocation in the consuming
# loop.  Moving its first read into a prologue would be a use-before-write, so
# this pass must reject it.
if "$LLVM_INSTALL_DIR/bin/mlir-opt" \
  --load-pass-plugin="$PLUGIN_LIBRARY" \
  --pass-pipeline='builtin.module(builtin.module(func.func(pipeline-gemm-rhs-load)))' \
  "$PLUGIN_DIR/test/gemm_rhs_pipeline_unsafe.mlir" \
  -o /dev/null >/dev/null 2>&1; then
  echo "unsafe same-loop pack/read fixture was unexpectedly pipelined" >&2
  exit 1
fi

echo "PASS: explicit packed A/B load pipeline carries two vectors across scf.for"
echo "PASS: same-loop pack/read use-before-write shape was rejected"
