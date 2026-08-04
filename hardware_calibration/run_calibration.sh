#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd -P)
mode=${1:-full}
output_dir=${2:-"$PWD/hardware_calibration_$(date +%Y%m%d_%H%M%S)"}

case "$mode" in
  quick)
    exec python3 "$script_dir/calibrate_hardware.py" --quick --output "$output_dir"
    ;;
  full)
    exec python3 "$script_dir/calibrate_hardware.py" --output "$output_dir"
    ;;
  -h|--help)
    echo "usage: $0 [quick|full] [output-dir]"
    ;;
  *)
    echo "mode must be quick or full" >&2
    exit 2
    ;;
esac
