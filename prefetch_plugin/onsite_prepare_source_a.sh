#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <llvm-install-prefix> <triton-shared-opt> <compiler.py> <00-input.mlir> <output-parent>" >&2
  exit 2
fi

LLVM_INSTALL_DIR="$(cd "$1" && pwd)"
TRITON_SHARED_OPT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
COMPILER_PY="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
INPUT_00="$(cd "$(dirname "$4")" && pwd)/$(basename "$4")"
PLUGIN_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$5"
OUTPUT_PARENT="$(cd "$5" && pwd)"
RUN_ID="source-a-$(date +%Y%m%d-%H%M%S)-$$"
RUN_DIR="$OUTPUT_PARENT/$RUN_ID"
mkdir "$RUN_DIR"

echo "[1/8] build plugin and run generic smoke test"
bash "$PLUGIN_DIR/build_and_smoke_mlir_opt.sh" "$LLVM_INSTALL_DIR"

echo "[2/8] verify strict source-A matcher on its fixture"
bash "$PLUGIN_DIR/test_bmm_source_a.sh" "$LLVM_INSTALL_DIR"

echo "[3/8] build no-prefetch split roundtrip"
SME_SKIP_PLUGIN_BUILD=1 bash "$PLUGIN_DIR/onsite_split_replay.sh" \
  "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" "$INPUT_00" \
  "$RUN_DIR/roundtrip" roundtrip

echo "[4/8] lower no-prefetch control to LLIR and object"
SME_EXPECT_PREFETCH=0 bash "$PLUGIN_DIR/onsite_stage2.sh" \
  "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" "$COMPILER_PY" \
  "$RUN_DIR/roundtrip/01_roundtrip_no_prefetch.mlir" \
  "$RUN_DIR/roundtrip/final"

echo "[5/8] build source-A distance=8 issue-every=8 L2 candidate"
SME_SKIP_PLUGIN_BUILD=1 bash "$PLUGIN_DIR/onsite_split_replay.sh" \
  "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" "$INPUT_00" \
  "$RUN_DIR/a8-e8-l2" bmm-source-a 8 2 4 8 64
bash "$PLUGIN_DIR/onsite_stage2.sh" \
  "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" "$COMPILER_PY" \
  "$RUN_DIR/a8-e8-l2/01_bmm_source_a_prefetch.mlir" \
  "$RUN_DIR/a8-e8-l2/final"

echo "[6/8] build source-A distance=8 issue-every=8 L1 candidate"
SME_SKIP_PLUGIN_BUILD=1 bash "$PLUGIN_DIR/onsite_split_replay.sh" \
  "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" "$INPUT_00" \
  "$RUN_DIR/a8-e8-l1" bmm-source-a 8 3 4 8 64
bash "$PLUGIN_DIR/onsite_stage2.sh" \
  "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" "$COMPILER_PY" \
  "$RUN_DIR/a8-e8-l1/01_bmm_source_a_prefetch.mlir" \
  "$RUN_DIR/a8-e8-l1/final"

echo "[7/8] record hashes and instruction counts"
python3 - "$INPUT_00" "$RUN_DIR" <<'PY'
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

input_path = pathlib.Path(sys.argv[1])
run_dir = pathlib.Path(sys.argv[2])

def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

variants = {}
for name in ("roundtrip", "a8-e8-l2", "a8-e8-l1"):
    final = run_dir / name / "final"
    disasm = final / "kernel.disasm"
    llir = final / "kernel.llir"
    object_path = final / "kernel.o"
    disassembly = disasm.read_text().lower()
    variants[name] = {
        "llir": str(llir),
        "llir_sha256": sha256(llir),
        "object": str(object_path),
        "object_sha256": sha256(object_path),
        "prfm": disassembly.count("prfm"),
        "fmopa": disassembly.count("fmopa"),
    }

report = {
    "schema": "onsite-source-a-candidate-bundle-v1",
    "input_00": str(input_path),
    "input_00_sha256": sha256(input_path),
    "run_dir": str(run_dir),
    "variants": variants,
    "status": "prepared",
}
(run_dir / "PREPARED.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
PY

echo "[8/8] preparation complete"
echo "RUN_DIR=$RUN_DIR"
echo "summary=$RUN_DIR/PREPARED.json"
echo "roundtrip LLIR=$RUN_DIR/roundtrip/final/kernel.llir"
echo "A8-E8-L2 LLIR=$RUN_DIR/a8-e8-l2/final/kernel.llir"
echo "A8-E8-L1 LLIR=$RUN_DIR/a8-e8-l1/final/kernel.llir"
echo "Next: install one LLIR at a time in the already verified override hash, run correctness, then paired timing."
