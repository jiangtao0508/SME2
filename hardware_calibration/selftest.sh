#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd -P)
project_dir=$(cd "$script_dir/.." && pwd -P)
test_root=$(mktemp -d "${TMPDIR:-/tmp}/sme-hardware-selftest.XXXXXX")
trap 'rm -rf -- "$test_root"' EXIT

PYTHONPYCACHEPREFIX="$test_root/pycache" \
  python3 -m unittest tests.test_hardware_calibration

PYTHONPYCACHEPREFIX="$test_root/pycache" \
  bash "$script_dir/run_calibration.sh" quick "$test_root/output" \
  > "$test_root/summary.stdout"

python3 - "$test_root/output/HardwareProfile.v1.1.json" <<'PY'
import json
import pathlib
import sys

profile = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert profile["schema_version"] == "1.1"
assert profile["source"] == "measured"
assert profile["measurements"]["latency"]
assert profile["measurements"]["stride"]
assert profile["measurements"]["prefetch"]
assert profile["derived"]["memory_latency_ns"] > 0
assert profile["derived"]["sustainable_bandwidth_bytes_per_ns"] > 0
print("PASS: HardwareProfile v1.1 quick calibration")
PY
