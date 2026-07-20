#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "${PYTHON_BIN:-python3}" "$ROOT/tests/test_onsite_workflow.py"
