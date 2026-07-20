#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "${PYTHON_BIN:-python3}" \
  "$ROOT/07_onsite_workflow/tools/trace_pytest_pipeline.py" "$@"
