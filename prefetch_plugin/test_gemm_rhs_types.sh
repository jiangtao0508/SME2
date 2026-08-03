#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <llvm-install-prefix>" >&2
  exit 2
fi

LLVM_INSTALL_DIR="$(cd "$1" && pwd)"
PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PLUGIN_DIR/build/type_matrix"
mkdir -p "$BUILD_DIR"

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

for element_type in f16 bf16 f32 f64; do
  sed "s/__ELEMENT_TYPE__/$element_type/g" \
    "$PLUGIN_DIR/test/gemm_rhs_type_template.mlir" \
    > "$BUILD_DIR/gemm_rhs_$element_type.mlir"
  "$LLVM_INSTALL_DIR/bin/mlir-opt" \
    --load-pass-plugin="$PLUGIN_LIBRARY" \
    --pass-pipeline='builtin.module(builtin.module(func.func(prefetch-gemm-rhs{distance=4 locality=3})))' \
    "$BUILD_DIR/gemm_rhs_$element_type.mlir" \
    -o "$BUILD_DIR/gemm_rhs_$element_type.prefetch.mlir"
  COUNT="$(grep -c 'memref.prefetch' "$BUILD_DIR/gemm_rhs_$element_type.prefetch.mlir" || true)"
  if [[ "$COUNT" -ne 1 ]]; then
    echo "expected one prefetch for $element_type, found $COUNT" >&2
    exit 1
  fi
  echo "PASS: prefetch-gemm-rhs matched $element_type"
done

echo "NOTE: this is a matcher type-independence test, not four full ArmSME pipelines"

for distance in 1 2 4 8; do
  for locality in 0 1 2 3; do
    OUTPUT="$BUILD_DIR/gemm_rhs_f32.d${distance}.l${locality}.mlir"
    "$LLVM_INSTALL_DIR/bin/mlir-opt" \
      --load-pass-plugin="$PLUGIN_LIBRARY" \
      --pass-pipeline="builtin.module(builtin.module(func.func(prefetch-gemm-rhs{distance=$distance locality=$locality})))" \
      "$BUILD_DIR/gemm_rhs_f32.mlir" \
      -o "$OUTPUT"
    grep -q "memref.prefetch.*locality<$locality>" "$OUTPUT"
  done
done
echo "PASS: distance/locality matrix covered distances 1,2,4,8 and localities 0,1,2,3"
