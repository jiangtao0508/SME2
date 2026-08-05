#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <llvm-install-prefix> <bufferized-before-sme.mlir> <output-dir>" >&2
  exit 2
fi

llvm_prefix=$(cd "$1" && pwd -P)
input_mlir=$(cd "$(dirname "$2")" && pwd -P)/$(basename "$2")
plugin_dir=$(cd "$(dirname "$0")" && pwd -P)
mkdir -p "$3"
output_dir=$(cd "$3" && pwd -P)

if [[ ! -f "$input_mlir" ]]; then
  echo "bufferized MLIR not found: $input_mlir" >&2
  exit 1
fi

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

profile="$output_dir/GemmKernelProfile.v1.json"
"$llvm_prefix/bin/mlir-opt" \
  --load-pass-plugin="$plugin_library" \
  --mlir-disable-threading \
  --pass-pipeline="builtin.module(prefetch-analyze-gemm-rhs{output-path=$profile})" \
  "$input_mlir" -o /dev/null

python3 -m json.tool "$profile" >/dev/null
echo "PASS: extracted numeric GEMM RHS features without changing the input IR"
echo "GEMM_KERNEL_PROFILE=$profile"
python3 "$plugin_dir/summarize_gemm_profile.py" "$profile"
