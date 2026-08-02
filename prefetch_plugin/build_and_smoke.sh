#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <llvm-install-prefix> <triton-shared-opt>" >&2
  exit 2
fi

LLVM_INSTALL_DIR="$(cd "$1" && pwd)"
TRITON_SHARED_OPT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PLUGIN_DIR/build"

for required in \
  "$LLVM_INSTALL_DIR/lib/cmake/llvm/LLVMConfig.cmake" \
  "$LLVM_INSTALL_DIR/lib/cmake/mlir/MLIRConfig.cmake" \
  "$LLVM_INSTALL_DIR/bin/mlir-translate" \
  "$LLVM_INSTALL_DIR/bin/llc" \
  "$TRITON_SHARED_OPT"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 1
  fi
done

cmake -S "$PLUGIN_DIR" -B "$BUILD_DIR" \
  -DMLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j2

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
  echo "plugin library was not produced" >&2
  exit 1
fi

"$TRITON_SHARED_OPT" \
  --load-pass-plugin="$PLUGIN_LIBRARY" \
  --pass-pipeline='builtin.module(func.func(prefetch-materialize{argument-index=0 distance=4 locality=3}))' \
  "$PLUGIN_DIR/test/simple_stream.mlir" \
  -o "$BUILD_DIR/simple_stream.prefetch.mlir"

"$TRITON_SHARED_OPT" \
  --load-pass-plugin="$PLUGIN_LIBRARY" \
  "$PLUGIN_DIR/test/simple_stream_transform.mlir" \
  --mlir-disable-threading \
  --transform-interpreter \
  --test-transform-dialect-erase-schedule \
  -o "$BUILD_DIR/simple_stream.transform.prefetch.mlir"

"$TRITON_SHARED_OPT" \
  "$BUILD_DIR/simple_stream.prefetch.mlir" \
  --convert-scf-to-cf \
  --convert-to-llvm \
  --reconcile-unrealized-casts \
  -o "$BUILD_DIR/simple_stream.llvm.mlir"

"$LLVM_INSTALL_DIR/bin/mlir-translate" \
  --mlir-to-llvmir "$BUILD_DIR/simple_stream.llvm.mlir" \
  -o "$BUILD_DIR/simple_stream.ll"
"$LLVM_INSTALL_DIR/bin/llc" \
  -mtriple=aarch64-unknown-linux-gnu -O2 \
  "$BUILD_DIR/simple_stream.ll" \
  -o "$BUILD_DIR/simple_stream.aarch64.s"

grep -q 'memref.prefetch' "$BUILD_DIR/simple_stream.transform.prefetch.mlir"
grep -q 'llvm.prefetch' "$BUILD_DIR/simple_stream.ll"
grep -q 'prfm' "$BUILD_DIR/simple_stream.aarch64.s"

echo "PASS: plugin loaded directly and through transform.apply_registered_pass"
echo "PASS: memref.prefetch lowered to llvm.prefetch and AArch64 prfm"
echo "plugin: $PLUGIN_LIBRARY"
echo "artifacts: $BUILD_DIR"

