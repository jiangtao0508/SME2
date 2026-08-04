#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd -P)
output_dir=${1:-"$PWD/sme_timing_$(date +%Y%m%d_%H%M%S)"}
groups=${2:-1000000}

exec python3 "$script_dir/calibrate_sme.py" --output "$output_dir" --groups "$groups"

