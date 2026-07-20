#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

required=(
  "README.md"
  "SECURITY_AND_DATA_BOUNDARY.md"
  "07_onsite_workflow/README.md"
  "07_onsite_workflow/tools/analyze_onsite_case.py"
  "docs/现场实验操作清单.md"
  "scripts/run_onsite_analysis.sh"
  "scripts/run_onsite_selftest.sh"
  "tests/test_onsite_workflow.py"
)

missing=0
for path in "${required[@]}"; do
  if [[ ! -f "$ROOT/$path" ]]; then
    printf 'MISSING %s\n' "$path"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

printf 'OK: %d required files are present.\n' "${#required[@]}"
