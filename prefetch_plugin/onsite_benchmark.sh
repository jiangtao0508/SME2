#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  onsite_benchmark.sh OVERRIDE_DIR

Runs the same FlagGems BMM twice with triton.testing.do_bench:
  1. baseline with kernel override disabled
  2. prefetch with OVERRIDE_DIR enabled

Defaults:
  shape:  [4, 8192, 64] x [4, 64, 2048]   # M=8192, K=64, N=2048（现场确认 MKN）
  dtype:  bfloat16
  warmup: 5
  reps:   20

Optional environment variables:
  SME_BENCH_BATCH, SME_BENCH_M, SME_BENCH_N, SME_BENCH_K
  SME_BENCH_DTYPE, SME_BENCH_WARMUP, SME_BENCH_REP
  SME_BENCH_ORDER=baseline-first (default) or prefetch-first
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

if [[ ! -d $1 ]]; then
  echo "ERROR: override directory does not exist: $1" >&2
  exit 2
fi

override_dir=$(cd "$1" && pwd -P)
llir_file=$(find "$override_dir" -type f -name '*.llir' | head -1)
if [[ -z "$llir_file" ]]; then
  echo "ERROR: no .llir found below the override directory: $override_dir" >&2
  exit 2
fi

command -v python >/dev/null 2>&1 || {
  echo "ERROR: python is not available; source the onsite env.sh first" >&2
  exit 2
}

python -c 'import torch, triton, flag_gems' >/dev/null 2>&1 || {
  echo "ERROR: cannot import torch/triton/flag_gems; source the onsite env.sh first" >&2
  exit 2
}

export SME_BENCH_BATCH=${SME_BENCH_BATCH:-4}
export SME_BENCH_M=${SME_BENCH_M:-8192}
export SME_BENCH_N=${SME_BENCH_N:-2048}
export SME_BENCH_K=${SME_BENCH_K:-64}
export SME_BENCH_DTYPE=${SME_BENCH_DTYPE:-bfloat16}
export SME_BENCH_WARMUP=${SME_BENCH_WARMUP:-5}
export SME_BENCH_REP=${SME_BENCH_REP:-20}
export SME_BENCH_ORDER=${SME_BENCH_ORDER:-baseline-first}

if [[ $SME_BENCH_ORDER != baseline-first && $SME_BENCH_ORDER != prefetch-first ]]; then
  echo "ERROR: SME_BENCH_ORDER must be baseline-first or prefetch-first" >&2
  exit 2
fi

run_root=$(mktemp -d "${TMPDIR:-/tmp}/sme-prefetch-bench.XXXXXX")
trap 'rm -rf -- "$run_root"' EXIT

# Triton's kernel override looks up $TRITON_OVERRIDE_DIR/<src-hash>/kernel.llir.
# The src hash is only known after compiling the kernel once, so probe it via
# the TRITON_KERNEL_DUMP mechanism (dump dir is also keyed by src hash), then
# lay out the override payload under the hash subdirectory.
probe_root="$run_root/probe"
probe_cache="$run_root/probe_cache"
(
  unset TRITON_KERNEL_OVERRIDE TRITON_OVERRIDE_DIR
  export TRITON_ALWAYS_COMPILE=1
  export TRITON_KERNEL_DUMP=1
  export TRITON_DUMP_DIR="$probe_root"
  export TRITON_CACHE_DIR="$probe_cache"
  python - <<'PY'
import os

import flag_gems
import torch

batch = int(os.environ["SME_BENCH_BATCH"])
m = int(os.environ["SME_BENCH_M"])
n = int(os.environ["SME_BENCH_N"])
k = int(os.environ["SME_BENCH_K"])
dtype = getattr(torch, os.environ["SME_BENCH_DTYPE"])

a = torch.randn((batch, m, k), dtype=dtype, device=flag_gems.device)
b = torch.randn((batch, k, n), dtype=dtype, device=flag_gems.device)
with flag_gems.use_gems():
    torch.bmm(a, b)  # compile only; source hash lands in TRITON_DUMP_DIR/<hash>/
PY
) >/dev/null 2>&1

src_hash="$(basename "$(ls -dt "$probe_root"/*/ 2>/dev/null | head -1)" 2>/dev/null || true)"
if [[ -z "$src_hash" || ! -d "$probe_root/$src_hash" ]]; then
  echo "ERROR: could not determine the kernel source hash from $probe_root" >&2
  echo "       confirm TRITON_KERNEL_DUMP is supported by the onsite Triton" >&2
  exit 1
fi
override_structure="$run_root/override"
mkdir -p "$override_structure/$src_hash"
cp "$llir_file" "$override_structure/$src_hash/kernel.llir"
echo "src_hash: $src_hash"
echo "override payload: $override_structure/$src_hash/kernel.llir"

run_case() {
  local label=$1
  local log_file=$2

  (
    unset TRITON_KERNEL_DUMP TRITON_DUMP_DIR
    export TRITON_ALWAYS_COMPILE=1
    export TRITON_CACHE_DIR="$run_root/cache_$label"

    if [[ $label == baseline ]]; then
      unset TRITON_KERNEL_OVERRIDE TRITON_OVERRIDE_DIR
    else
      export TRITON_KERNEL_OVERRIDE=1
      export TRITON_OVERRIDE_DIR="$override_structure"
    fi

    export SME_BENCH_LABEL=$label
    python - <<'PY'
import os

import flag_gems
import torch
import triton


label = os.environ["SME_BENCH_LABEL"]
batch = int(os.environ["SME_BENCH_BATCH"])
m = int(os.environ["SME_BENCH_M"])
n = int(os.environ["SME_BENCH_N"])
k = int(os.environ["SME_BENCH_K"])
warmup = int(os.environ["SME_BENCH_WARMUP"])
reps = int(os.environ["SME_BENCH_REP"])
dtype = getattr(torch, os.environ["SME_BENCH_DTYPE"])

a = torch.randn((batch, m, k), dtype=dtype, device=flag_gems.device)
b = torch.randn((batch, k, n), dtype=dtype, device=flag_gems.device)

with flag_gems.use_gems():
    fn = lambda: torch.bmm(a, b)
    fn()  # Compile before timing.
    latency = triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=reps,
        return_mode="median",
    )

print(f"SME_{label.upper()}_LATENCY_MS={float(latency):.6f}")
PY
  ) 2>&1 | tee "$log_file"
}

echo "SME benchmark: B=$SME_BENCH_BATCH M=$SME_BENCH_M N=$SME_BENCH_N K=$SME_BENCH_K dtype=$SME_BENCH_DTYPE"
echo "Run order: $SME_BENCH_ORDER"
if [[ $SME_BENCH_ORDER == baseline-first ]]; then
  echo "Running baseline..."
  run_case baseline "$run_root/baseline.log"
  echo "Running prefetch override..."
  run_case prefetch "$run_root/prefetch.log"
else
  echo "Running prefetch override..."
  run_case prefetch "$run_root/prefetch.log"
  echo "Running baseline..."
  run_case baseline "$run_root/baseline.log"
fi

if grep -Fq 'Overriding kernel with file' "$run_root/baseline.log"; then
  echo "ERROR: native baseline unexpectedly loaded an LLIR override" >&2
  exit 1
fi
if ! grep -Fq 'Overriding kernel with file' "$run_root/prefetch.log"; then
  echo "ERROR: prefetch run did not report a Triton LLIR override; results are invalid" >&2
  exit 1
fi

baseline_latency=$(sed -n 's/^SME_BASELINE_LATENCY_MS=//p' "$run_root/baseline.log" | tail -n 1)
prefetch_latency=$(sed -n 's/^SME_PREFETCH_LATENCY_MS=//p' "$run_root/prefetch.log" | tail -n 1)

if [[ -z $baseline_latency || -z $prefetch_latency ]]; then
  echo "ERROR: failed to read benchmark latency" >&2
  exit 1
fi

python - "$baseline_latency" "$prefetch_latency" <<'PY'
import sys

baseline = float(sys.argv[1])
prefetch = float(sys.argv[2])
speedup = baseline / prefetch
change = (prefetch / baseline - 1.0) * 100.0

print("RESULT_OVERRIDE_HIT=1")
print("RESULT_BASELINE_OVERRIDE_HIT=0")
print(f"RESULT_BASELINE_MS={baseline:.6f}")
print(f"RESULT_PREFETCH_MS={prefetch:.6f}")
print(f"RESULT_SPEEDUP={speedup:.6f}x")
print(f"RESULT_LATENCY_CHANGE_PERCENT={change:+.3f}%")
PY
