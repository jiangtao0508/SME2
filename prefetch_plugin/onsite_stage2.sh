#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <llvm-install-prefix> <triton-shared-opt> <compiler.py> <01-prefetch.mlir> <output-dir>" >&2
  exit 2
fi

LLVM_INSTALL_DIR="$(cd "$1" && pwd)"
TRITON_SHARED_OPT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
COMPILER_PY="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
INPUT_01="$(cd "$(dirname "$4")" && pwd)/$(basename "$4")"
PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$PLUGIN_DIR/.." && pwd)"
mkdir -p "$5"
OUTPUT_DIR="$(cd "$5" && pwd)"

"$TRITON_SHARED_OPT" "$INPUT_01" \
  --test-transform-dialect-erase-schedule \
  -o "$OUTPUT_DIR/02_after_erase_schedule.mlir"

"$TRITON_SHARED_OPT" "$OUTPUT_DIR/02_after_erase_schedule.mlir" \
  --convert-math-to-libm \
  --convert-vector-to-llvm \
  --convert-to-llvm \
  --llvm-promote-i1-to-i8 \
  -o "$OUTPUT_DIR/03_after_convert_to_llvm.mlir"

"$TRITON_SHARED_OPT" "$OUTPUT_DIR/03_after_convert_to_llvm.mlir" \
  --canonicalize \
  -o "$OUTPUT_DIR/04_after_canonicalize.mlir"

"$TRITON_SHARED_OPT" "$OUTPUT_DIR/04_after_canonicalize.mlir" \
  --strip-debuginfo \
  -o "$OUTPUT_DIR/05_after_strip_debug.mlir"

"$TRITON_SHARED_OPT" "$OUTPUT_DIR/05_after_strip_debug.mlir" \
  --llvm-legalize-float8-types \
  -o "$OUTPUT_DIR/06_after_legalize_float8.mlir"

cp "$OUTPUT_DIR/06_after_legalize_float8.mlir" "$OUTPUT_DIR/ll.mlir"
python3 "$PROJECT_DIR/reproduction/extract_public_llvm_function.py" \
  --compiler "$COMPILER_PY" \
  --input "$OUTPUT_DIR/ll.mlir"

"$LLVM_INSTALL_DIR/bin/mlir-translate" \
  "$OUTPUT_DIR/ll.mlir" \
  --mlir-to-llvmir \
  -o "$OUTPUT_DIR/kernel.llir"

"$LLVM_INSTALL_DIR/bin/llc" \
  "$OUTPUT_DIR/kernel.llir" \
  -filetype=obj \
  -mtriple=aarch64-unknown-linux-gnu \
  -mattr=+sme,+dotprod,+v9a,+v8.5a,+v8.4a,+v8.3a,+v8.2a,+v8.1a,+sve,+sve2 \
  -o "$OUTPUT_DIR/kernel.o"

"$LLVM_INSTALL_DIR/bin/llvm-objdump" -d "$OUTPUT_DIR/kernel.o" \
  > "$OUTPUT_DIR/kernel.disasm"

PREFETCH_COUNT="$(grep -c 'prfm' "$OUTPUT_DIR/kernel.disasm" || true)"
MOPA_COUNT="$(grep -c 'fmopa' "$OUTPUT_DIR/kernel.disasm" || true)"
EXPECT_PREFETCH="${SME_EXPECT_PREFETCH:-1}"

if [[ "$EXPECT_PREFETCH" != 0 && "$EXPECT_PREFETCH" != 1 ]]; then
  echo "SME_EXPECT_PREFETCH must be 0 or 1" >&2
  exit 2
fi
if [[ "$MOPA_COUNT" -eq 0 ]]; then
  echo "expected FMOPA in the generated object" >&2
  echo "PRFM=$PREFETCH_COUNT FMOPA=$MOPA_COUNT" >&2
  exit 1
fi
if [[ "$EXPECT_PREFETCH" -eq 1 && "$PREFETCH_COUNT" -eq 0 ]]; then
  echo "expected PRFM in the generated object" >&2
  exit 1
fi
if [[ "$EXPECT_PREFETCH" -eq 0 && "$PREFETCH_COUNT" -ne 0 ]]; then
  echo "no-prefetch control unexpectedly contains PRFM=$PREFETCH_COUNT" >&2
  exit 1
fi

echo "PASS: object contains expected PRFM=$PREFETCH_COUNT and FMOPA=$MOPA_COUNT"
echo "LLIR override candidate: $OUTPUT_DIR/kernel.llir"
echo "object: $OUTPUT_DIR/kernel.o"
echo "disassembly: $OUTPUT_DIR/kernel.disasm"
