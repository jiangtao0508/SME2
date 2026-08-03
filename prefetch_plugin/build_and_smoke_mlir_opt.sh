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

for required in \
  "$LLVM_INSTALL_DIR/lib/cmake/llvm/LLVMConfig.cmake" \
  "$LLVM_INSTALL_DIR/lib/cmake/mlir/MLIRConfig.cmake" \
  "$MLIR_OPT" \
  "$LLVM_INSTALL_DIR/bin/mlir-translate" \
  "$LLVM_INSTALL_DIR/bin/llc"; do
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

"$MLIR_OPT" \
  --load-pass-plugin="$PLUGIN_LIBRARY" \
  --pass-pipeline='builtin.module(func.func(prefetch-materialize{argument-index=0 distance=4 locality=3}))' \
  "$PLUGIN_DIR/test/simple_stream.mlir" \
  -o "$BUILD_DIR/simple_stream.mlir-opt.prefetch.mlir"

"$MLIR_OPT" \
  "$BUILD_DIR/simple_stream.mlir-opt.prefetch.mlir" \
  --convert-scf-to-cf \
  --convert-to-llvm \
  --reconcile-unrealized-casts \
  -o "$BUILD_DIR/simple_stream.mlir-opt.llvm.mlir"

"$LLVM_INSTALL_DIR/bin/mlir-translate" \
  --mlir-to-llvmir "$BUILD_DIR/simple_stream.mlir-opt.llvm.mlir" \
  -o "$BUILD_DIR/simple_stream.mlir-opt.ll"
"$LLVM_INSTALL_DIR/bin/llc" \
  -mtriple=aarch64-unknown-linux-gnu -O2 \
  "$BUILD_DIR/simple_stream.mlir-opt.ll" \
  -o "$BUILD_DIR/simple_stream.mlir-opt.aarch64.s"

grep -q 'memref.prefetch' "$BUILD_DIR/simple_stream.mlir-opt.prefetch.mlir"
grep -q 'llvm.prefetch' "$BUILD_DIR/simple_stream.mlir-opt.ll"
grep -q 'prfm' "$BUILD_DIR/simple_stream.mlir-opt.aarch64.s"

echo "PASS: mlir-opt loaded prefetch plugin"
echo "PASS: memref.prefetch lowered to llvm.prefetch and AArch64 prfm"
echo "plugin: $PLUGIN_LIBRARY"

