#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <llvm-install-prefix>" >&2
  exit 2
fi

llvm_prefix=$(cd "$1" && pwd -P)
plugin_dir=$(cd "$(dirname "$0")" && pwd -P)
test_dir="$plugin_dir/build/analysis_test"
mkdir -p "$test_dir"

bash "$plugin_dir/build_and_smoke_mlir_opt.sh" "$llvm_prefix" >/dev/null
plugin_library=""
for candidate in \
  "$plugin_dir/build/PrefetchPassPlugin.so" \
  "$plugin_dir/build/PrefetchPassPlugin.dylib"; do
  if [[ -f "$candidate" ]]; then
    plugin_library="$candidate"
    break
  fi
done
if [[ -z "$plugin_library" ]]; then
  echo "plugin library was not produced" >&2
  exit 1
fi

input="$test_dir/gemm_rhs_bf16.mlir"
profile="$test_dir/GemmKernelProfile.v1.json"
sed 's/__ELEMENT_TYPE__/bf16/g' \
  "$plugin_dir/test/gemm_rhs_type_template.mlir" > "$input"

"$llvm_prefix/bin/mlir-opt" \
  --load-pass-plugin="$plugin_library" \
  --pass-pipeline="builtin.module(prefetch-analyze-gemm-rhs{output-path=$profile})" \
  "$input" -o /dev/null

python3 - "$profile" <<'PY'
import json
import pathlib
import sys

profile = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert profile["schema_version"] == "1.0"
assert profile["candidate_count"] == 1
feature = profile["candidates"][0]
assert feature["element_bytes"] == 2
assert feature["source_allocation_bytes"] == 32768
assert feature["rhs_row_bytes"] == 512
assert feature["vector_read_bytes"] == 32
assert feature["loop_trip_count"] == 64
assert feature["bytes_advanced_per_loop_iteration"] == 512
assert feature["lineage"]["writer_operation_count"] == 1
assert feature["lineage"]["memref_copy_writer_count"] == 1
assert feature["lineage"]["source_argument_indices"] == [0]
print("PASS: extracted numeric GEMM RHS kernel profile")
PY
