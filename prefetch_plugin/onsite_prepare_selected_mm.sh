#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <llvm-prefix> <triton-shared-opt> <compiler.py> <selected-mm-00.mlir> <output-parent>" >&2
  exit 2
fi

LLVM_PREFIX="$(cd "$1" && pwd)"
TRITON_SHARED_OPT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
COMPILER_PY="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
INPUT_00="$(cd "$(dirname "$4")" && pwd)/$(basename "$4")"
mkdir -p "$5"
OUTPUT_PARENT="$(cd "$5" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$OUTPUT_PARENT/mm-prefetch-$(date +%Y%m%d-%H%M%S)-$$"
mkdir "$RUN_DIR"

echo "[1/4] no-prefetch split roundtrip"
bash "$SCRIPT_DIR/onsite_split_replay.sh" \
  "$LLVM_PREFIX" "$TRITON_SHARED_OPT" "$INPUT_00" "$RUN_DIR/roundtrip" roundtrip
SME_EXPECT_PREFETCH=0 bash "$SCRIPT_DIR/onsite_stage2.sh" \
  "$LLVM_PREFIX" "$TRITON_SHARED_OPT" "$COMPILER_PY" \
  "$RUN_DIR/roundtrip/01_roundtrip_no_prefetch.mlir" "$RUN_DIR/roundtrip/final"

echo "[2/4] packed RHS L2-to-L1 prefetch: distance=8, one line, every row"
bash "$SCRIPT_DIR/onsite_split_replay.sh" \
  "$LLVM_PREFIX" "$TRITON_SHARED_OPT" "$INPUT_00" "$RUN_DIR/packed-rhs-d8-l1" \
  gemm-rhs 8 3 1 1 64
bash "$SCRIPT_DIR/onsite_stage2.sh" \
  "$LLVM_PREFIX" "$TRITON_SHARED_OPT" "$COMPILER_PY" \
  "$RUN_DIR/packed-rhs-d8-l1/01_gemm_rhs_prefetch.mlir" \
  "$RUN_DIR/packed-rhs-d8-l1/final"

echo "[3/4] verify instruction conservation"
python3 - "$RUN_DIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
variants = {}
for name in ("roundtrip", "packed-rhs-d8-l1"):
    disasm = (root / name / "final" / "kernel.disasm").read_text().lower()
    variants[name] = {"prfm": disasm.count("prfm"), "fmopa": disasm.count("fmopa")}
if variants["roundtrip"]["prfm"] != 0:
    raise SystemExit("roundtrip unexpectedly contains PRFM")
if variants["packed-rhs-d8-l1"]["prfm"] == 0:
    raise SystemExit("packed RHS variant contains no PRFM")
if variants["roundtrip"]["fmopa"] == 0 or variants["roundtrip"]["fmopa"] != variants["packed-rhs-d8-l1"]["fmopa"]:
    raise SystemExit(f"FMOPA count changed: {variants}")
(root / "MM_PREPARED.json").write_text(json.dumps({"variants": variants}, indent=2) + "\n")
for name, counts in variants.items():
    print(f"MM_PREPARED variant={name} prfm={counts['prfm']} fmopa={counts['fmopa']}")
PY

echo "[4/4] preparation complete"
echo "MM_PREPARE_DIR=$RUN_DIR"
echo "MM_ROUNDTRIP_LLIR=$RUN_DIR/roundtrip/final/kernel.llir"
echo "MM_PREFETCH_LLIR=$RUN_DIR/packed-rhs-d8-l1/final/kernel.llir"
