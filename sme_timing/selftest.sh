#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "$0")/.." && pwd -P)
test_root=$(mktemp -d "${TMPDIR:-/tmp}/sme-timing-selftest.XXXXXX")
trap 'rm -rf -- "$test_root"' EXIT

cd "$project_dir"
PYTHONPYCACHEPREFIX="$test_root/pycache" \
  python3 -m unittest -v tests.test_sme_timing

if command -v clang >/dev/null 2>&1; then
  clang --target=aarch64-unknown-linux-gnu -march=armv9-a+sme+bf16 \
    -c sme_timing/sme_fmopa_probe.S -o "$test_root/sme_fmopa_probe.o"
  strings "$test_root/sme_fmopa_probe.o" >/dev/null
  echo "PASS: assembler accepted RDSVL/SMSTART/BFMOPA for AArch64 SME"
else
  echo "SKIP: clang unavailable for cross-assembly syntax check"
fi
