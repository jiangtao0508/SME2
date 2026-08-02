#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 <llvm-install-prefix> <triton-shared-opt> <00_input.mlir> <output-dir> [distance] [locality]" >&2
  exit 2
fi

LLVM_INSTALL_DIR="$(cd "$1" && pwd)"
TRITON_SHARED_OPT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
INPUT_MLIR="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$4"
OUTPUT_DIR="$(cd "$4" && pwd)"
DISTANCE="${5:-4}"
LOCALITY="${6:-3}"

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

bash "$PLUGIN_DIR/build_and_smoke.sh" \
  "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT"

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

SNAPSHOT_MLIR="$OUTPUT_DIR/bufferized_before_sme.mlir"
SNAPSHOT_SCHEDULE="$OUTPUT_DIR/00_input.snapshot.mlir"

python3 "$PLUGIN_DIR/inject_schedule.py" \
  "$INPUT_MLIR" "$SNAPSHOT_SCHEDULE" \
  --mode snapshot \
  --snapshot-output="$SNAPSHOT_MLIR"

"$TRITON_SHARED_OPT" \
  --load-pass-plugin="$PLUGIN_LIBRARY" \
  "$SNAPSHOT_SCHEDULE" \
  --mlir-disable-threading \
  --transform-interpreter \
  -o "$OUTPUT_DIR/01_snapshot_pipeline_complete.mlir"

grep -q 'scf.for' "$SNAPSHOT_MLIR"
grep -q 'vector.transfer_read' "$SNAPSHOT_MLIR"

PREFETCH_SCHEDULE="$OUTPUT_DIR/00_input.gemm_rhs_prefetch.mlir"
python3 "$PLUGIN_DIR/inject_schedule.py" \
  "$INPUT_MLIR" "$PREFETCH_SCHEDULE" \
  --mode gemm-rhs \
  --distance "$DISTANCE" \
  --locality "$LOCALITY"

"$TRITON_SHARED_OPT" \
  --load-pass-plugin="$PLUGIN_LIBRARY" \
  "$PREFETCH_SCHEDULE" \
  --mlir-disable-threading \
  --transform-interpreter \
  -o "$OUTPUT_DIR/01_gemm_rhs_prefetch.mlir"

PREFETCH_COUNT="$(grep -c 'llvm.intr.prefetch' "$OUTPUT_DIR/01_gemm_rhs_prefetch.mlir" || true)"
SME_COUNT="$(grep -c 'arm_sme.intr.mopa' "$OUTPUT_DIR/01_gemm_rhs_prefetch.mlir" || true)"

if [[ "$PREFETCH_COUNT" -eq 0 ]]; then
  echo "snapshot succeeded, but the public GEMM RHS matcher did not match this kernel" >&2
  echo "keep $SNAPSHOT_MLIR and adapt the resolver before continuing" >&2
  exit 1
fi
if [[ "$SME_COUNT" -eq 0 ]]; then
  echo "prefetch survived, but no arm_sme.intr.mopa was found" >&2
  exit 1
fi

echo "PASS: captured bufferized payload: $SNAPSHOT_MLIR"
echo "PASS: llvm.intr.prefetch count after SME lowering: $PREFETCH_COUNT"
echo "PASS: arm_sme.intr.mopa count: $SME_COUNT"
echo "configuration: distance=$DISTANCE locality=$LOCALITY"
echo "next: continue from 01_gemm_rhs_prefetch.mlir using the site's existing lowering commands"
