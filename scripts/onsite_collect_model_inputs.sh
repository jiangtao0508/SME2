#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <llvm-install-prefix> <bufferized-before-sme.mlir> [output-root] [quick|full]" >&2
}

if [[ $# -lt 2 || $# -gt 4 ]]; then
  usage
  exit 2
fi

project_dir=$(cd "$(dirname "$0")/.." && pwd -P)
llvm_prefix=$(cd "$1" && pwd -P)
bufferized_mlir=$(cd "$(dirname "$2")" && pwd -P)/$(basename "$2")
output_root=${3:-"$PWD/onsite_model_inputs"}
mode=${4:-full}

if [[ ! -f "$bufferized_mlir" ]]; then
  echo "bufferized MLIR not found: $bufferized_mlir" >&2
  exit 1
fi
if [[ ! -x "$llvm_prefix/bin/mlir-opt" ]]; then
  echo "mlir-opt not found below LLVM prefix: $llvm_prefix/bin/mlir-opt" >&2
  exit 1
fi
if [[ "$mode" != "quick" && "$mode" != "full" ]]; then
  echo "mode must be quick or full" >&2
  exit 2
fi

mkdir -p "$output_root"
output_root=$(cd "$output_root" && pwd -P)
run_dir="$output_root/model_inputs_$(date +%Y%m%d_%H%M%S)_$$"
if [[ "$run_dir" == *" "* ]]; then
  echo "output path must not contain spaces: $run_dir" >&2
  exit 1
fi
mkdir -p "$run_dir/logs"

run_stage() {
  local name=$1
  shift
  echo "[$name] running"
  if "$@" >"$run_dir/logs/$name.log" 2>&1; then
    echo "[$name] PASS"
  else
    echo "[$name] FAILED: $run_dir/logs/$name.log" >&2
    tail -n 40 "$run_dir/logs/$name.log" >&2 || true
    printf 'failed_step=%s\n' "$name" > "$run_dir/FAILED.txt"
    exit 1
  fi
}

architecture=$(uname -m)
case "$architecture" in
  aarch64|arm64) ;;
  *)
    echo "this calibration must run on the target AArch64 machine, found: $architecture" >&2
    exit 1
    ;;
esac

echo "OUTPUT_DIR=$run_dir"
echo "Privacy: outputs remain onsite; no IR is copied into this directory."

run_stage 00_layout bash "$project_dir/scripts/check_layout.sh"
run_stage 01_hardware_selftest bash "$project_dir/hardware_calibration/selftest.sh"
run_stage 02_sme_selftest bash "$project_dir/sme_timing/selftest.sh"
run_stage 03_cost_model_selftest bash "$project_dir/cost_model/selftest.sh"
run_stage 04_hardware_profile \
  bash "$project_dir/hardware_calibration/run_calibration.sh" \
  "$mode" "$run_dir/hardware"
run_stage 05_sme_timing \
  bash "$project_dir/sme_timing/run_sme_timing.sh" "$run_dir/sme_timing"
run_stage 06_gemm_kernel_profile \
  bash "$project_dir/prefetch_plugin/onsite_extract_gemm_profile.sh" \
  "$llvm_prefix" "$bufferized_mlir" "$run_dir/kernel"

hardware_profile="$run_dir/hardware/HardwareProfile.v1.1.json"
sme_profile="$run_dir/sme_timing/SmeTimingProfile.v1.json"
kernel_profile="$run_dir/kernel/GemmKernelProfile.v1.json"
for required in "$hardware_profile" "$sme_profile" "$kernel_profile"; do
  if [[ ! -s "$required" ]]; then
    echo "expected profile was not produced: $required" >&2
    exit 1
  fi
done

{
  printf 'status=PASS\n'
  printf 'mode=%s\n' "$mode"
  printf 'hardware_profile=%s\n' "$hardware_profile"
  printf 'sme_timing_profile=%s\n' "$sme_profile"
  printf 'gemm_kernel_profile=%s\n' "$kernel_profile"
  printf 'next=derive static FMOPA count per K step, then generate PrefetchPlan v1.1\n'
  printf 'privacy=no source, IR, tensor, object, or disassembly was copied into this result\n'
} > "$run_dir/MODEL_INPUTS_READY.txt"

echo "PASS: all numeric model inputs were collected"
echo "SUMMARY=$run_dir/MODEL_INPUTS_READY.txt"
echo "Keep the entire output directory onsite."
