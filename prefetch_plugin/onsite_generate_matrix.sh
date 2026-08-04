#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  onsite_generate_matrix.sh LLVM_PREFIX TRITON_SHARED_OPT COMPILER_PY \
    INPUT_00_MLIR OUTPUT_DIR OVERRIDE_TEMPLATE_DIR

Builds seven prefetch variants, lowers every variant to LLIR/object code, and
creates a correctly named override tree by copying the relative hash/name from
OVERRIDE_TEMPLATE_DIR. OUTPUT_DIR must not already exist.
EOF
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  usage
  exit 0
fi
if [[ $# -ne 6 ]]; then
  usage >&2
  exit 2
fi

script_dir=$(cd "$(dirname "$0")" && pwd -P)
llvm_prefix=$(cd "$1" && pwd -P)
triton_shared_opt=$(cd "$(dirname "$2")" && pwd -P)/$(basename "$2")
compiler_py=$(cd "$(dirname "$3")" && pwd -P)/$(basename "$3")
input_00=$(cd "$(dirname "$4")" && pwd -P)/$(basename "$4")
output_dir=$5
template_dir=$(cd "$6" && pwd -P)

for required in "$triton_shared_opt" "$compiler_py" "$input_00"; do
  if [[ ! -f $required ]]; then
    echo "ERROR: required file does not exist: $required" >&2
    exit 2
  fi
done
if [[ -e $output_dir ]]; then
  echo "ERROR: output directory already exists: $output_dir" >&2
  exit 2
fi
if [[ $output_dir == *" "* ]]; then
  echo "ERROR: output directory must not contain spaces" >&2
  exit 2
fi

template_count=$(find "$template_dir" -type f -name '*.llir' | wc -l | tr -d '[:space:]')
if [[ $template_count != 1 ]]; then
  echo "ERROR: expected exactly one LLIR in override template; found $template_count" >&2
  exit 2
fi
template_llir=$(find "$template_dir" -type f -name '*.llir' -print -quit)
override_relative=${template_llir#"$template_dir"/}
if [[ $override_relative == "$template_llir" ]]; then
  echo "ERROR: failed to derive override hash/name from template" >&2
  exit 2
fi

mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd -P)
manifest="$output_dir/matrix.tsv"
printf 'variant\tdistance\tlocality\tcoverage_lines\tissue_every\tcache_line_bytes\tprfm\tfmopa\toverride_dir\n' > "$manifest"

echo "Building the plugin once..."
bash "$script_dir/build_and_smoke_mlir_opt.sh" "$llvm_prefix"
export SME_SKIP_PLUGIN_BUILD=1

variants=(
  'd1_l1_e1 1 3 1 1 64'
  'd2_l1_e1 2 3 1 1 64'
  'd4_l1_e1 4 3 1 1 64'
  'd8_l1_e1 8 3 1 1 64'
  'd4_l1_e2 4 3 1 2 64'
  'd8_l1_e2 8 3 1 2 64'
  'd8_l2_e2 8 2 1 2 64'
)

for spec in "${variants[@]}"; do
  read -r variant distance locality coverage issue_every line_bytes <<< "$spec"
  variant_dir="$output_dir/$variant"
  stage1_dir="$variant_dir/stage1"
  final_dir="$variant_dir/final"
  override_dir="$variant_dir/override"

  echo "=== Generating $variant ==="
  bash "$script_dir/onsite_split_replay.sh" \
    "$llvm_prefix" "$triton_shared_opt" "$input_00" "$stage1_dir" \
    gemm-rhs "$distance" "$locality" "$coverage" "$issue_every" "$line_bytes"

  bash "$script_dir/onsite_stage2.sh" \
    "$llvm_prefix" "$triton_shared_opt" "$compiler_py" \
    "$stage1_dir/01_gemm_rhs_prefetch.mlir" "$final_dir"

  prfm=$(grep -c 'prfm' "$final_dir/kernel.disasm" || true)
  fmopa=$(grep -c 'fmopa' "$final_dir/kernel.disasm" || true)
  if [[ $prfm -eq 0 || $fmopa -eq 0 ]]; then
    echo "ERROR: $variant produced PRFM=$prfm FMOPA=$fmopa" >&2
    exit 1
  fi

  mkdir -p "$override_dir/$(dirname "$override_relative")"
  cp "$final_dir/kernel.llir" "$override_dir/$override_relative"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$variant" "$distance" "$locality" "$coverage" "$issue_every" \
    "$line_bytes" "$prfm" "$fmopa" "$override_dir" >> "$manifest"
done

echo "MATRIX_READY=$output_dir"
echo "MATRIX_MANIFEST=$manifest"
