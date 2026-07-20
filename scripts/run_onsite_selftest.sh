#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "${PYTHON_BIN:-python3}" -m unittest discover \
  -s "$ROOT/tests" -p 'test_*.py'
