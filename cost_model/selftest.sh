#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "$0")/.." && pwd -P)
test_root=$(mktemp -d "${TMPDIR:-/tmp}/sme-cost-model-selftest.XXXXXX")
trap 'rm -rf -- "$test_root"' EXIT

cd "$project_dir"
PYTHONPYCACHEPREFIX="$test_root/pycache" \
  python3 -m unittest -v tests.test_gemm_cost_model tests.test_prefetch_plan_options

