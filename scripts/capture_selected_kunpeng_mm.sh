#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/prefetch_plugin/onsite_capture_selected_mm.sh" "$@"
