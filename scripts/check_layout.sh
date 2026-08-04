#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

required=(
  "README.md"
  "SECURITY_AND_DATA_BOUNDARY.md"
  "07_onsite_workflow/README.md"
  "07_onsite_workflow/tools/analyze_onsite_case.py"
  "07_onsite_workflow/tools/onsite_pytest_probe.py"
  "07_onsite_workflow/tools/show_onsite_summary.py"
  "07_onsite_workflow/tools/trace_pytest_pipeline.py"
  "docs/现场实验操作清单.md"
  "docs/pytest到中间文件完整追踪.md"
  "docs/已知BMM调用链与现场待证实项.md"
  "scripts/run_onsite_analysis.sh"
  "scripts/run_full_pytest_trace.sh"
  "scripts/show_onsite_summary.sh"
  "scripts/run_onsite_selftest.sh"
  "tests/test_onsite_workflow.py"
  "prefetch_plugin/onsite_preflight.sh"
  "prefetch_plugin/onsite_split_replay.sh"
  "prefetch_plugin/onsite_full_experiment.sh"
  "prefetch_plugin/prefetch_plan_options.py"
  "prefetch_plugin/onsite_extract_gemm_profile.sh"
  "prefetch_plugin/gemm-kernel-profile-v1.schema.json"
  "hardware_calibration/run_calibration.sh"
  "cost_model/plan_gemm_rhs.py"
  "cost_model/prefetch-plan-v1.1.schema.json"
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
