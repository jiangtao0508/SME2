#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <llvm-install-prefix>" >&2
  exit 2
fi

LLVM_INSTALL_DIR="$(cd "$1" && pwd)"
PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PLUGIN_DIR/build"
MLIR_OPT="$LLVM_INSTALL_DIR/bin/mlir-opt"

PLUGIN_LIBRARY=""
for candidate in \
  "$BUILD_DIR/PrefetchPassPlugin.so" \
  "$BUILD_DIR/PrefetchPassPlugin.dylib"; do
  if [[ -f "$candidate" ]]; then
    PLUGIN_LIBRARY="$candidate"
    break
  fi
done
if [[ -z "$PLUGIN_LIBRARY" ]]; then
  echo "plugin library was not produced; run build_and_smoke_mlir_opt.sh first" >&2
  exit 1
fi

OUTPUT="$BUILD_DIR/bmm_source_a.prefetch.mlir"
"$MLIR_OPT" \
  --load-pass-plugin="$PLUGIN_LIBRARY" \
  --pass-pipeline='builtin.module(builtin.module(func.func(prefetch-bmm-source{argument-index=0 distance=8 locality=2 issue-every=8 expected-rows=4 expected-tile-k=4})))' \
  "$PLUGIN_DIR/test/bmm_source_a.mlir" \
  -o "$OUTPUT"

PREFETCH_COUNT="$(grep -c 'memref.prefetch' "$OUTPUT" || true)"
if [[ "$PREFETCH_COUNT" -ne 4 ]]; then
  echo "expected four A-source prefetches, found prefetch=$PREFETCH_COUNT" >&2
  exit 1
fi
grep -q 'memref.reinterpret_cast %arg0.*prefetch.distance_iterations = 8' "$OUTPUT"
grep -q 'prefetch.issue_every = 8' "$OUTPUT"
if grep 'memref.prefetch' "$OUTPUT" | grep -q '%alloc'; then
  echo "A-source prefetch unexpectedly targets a private alloc" >&2
  exit 1
fi

echo "PASS: source-A pass emitted four guarded prefetches from argument 0"
echo "PASS: distance=8 issue-every=8 and no private alloc target"
echo "output: $OUTPUT"
