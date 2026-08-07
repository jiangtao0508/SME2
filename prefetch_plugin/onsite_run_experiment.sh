#!/usr/bin/env bash
# 现场一键预取实验：形状确认 -> 硬件采集 -> preflight -> source-A 准备
#   -> 每变体正确性 -> 配对计时 -> 汇总报告
#
# 用法：
#   bash onsite_run_experiment.sh [site.env]
#
# site.env 内容见 site.env.example。所有路径也可用环境变量提供，未设置的
# 常见路径会自动探测。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -ge 1 && -f "$1" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$1"; set +a
fi

# 现场 env.sh 已导出的变量优先（TRITON_SHARED_OPT_PATH 可能是目录或文件）
if [[ -z "${TRITON_SHARED_OPT:-}" && -n "${TRITON_SHARED_OPT_PATH:-}" ]]; then
  _p="${TRITON_SHARED_OPT_PATH%/}"
  if [[ -d "$_p" ]]; then
    if [[ -x "$_p/triton-shared-opt" ]]; then
      TRITON_SHARED_OPT="$_p/triton-shared-opt"
    else
      TRITON_SHARED_OPT="$(find "$_p" -maxdepth 2 -name triton-shared-opt -type f 2>/dev/null | head -1)"
    fi
  else
    TRITON_SHARED_OPT="$_p"
  fi
fi

find_in() {
  local name=$1
  shift
  local root found
  for root in "$@"; do
    [[ -d "$root" ]] || continue
    found="$(find "$root" -name "$name" -type f 2>/dev/null | head -1)"
    [[ -n "$found" ]] && { echo "$found"; return 0; }
  done
  return 1
}

if [[ -z "${TRITON_SHARED_OPT:-}" ]]; then
  TRITON_SHARED_OPT="$(find_in triton-shared-opt /home/share "$HOME" "$PWD" 2>/dev/null || true)"
fi

if [[ -z "${LLVM_INSTALL_DIR:-}" ]]; then
  _mlir_opt="$(find_in mlir-opt /home/share "$HOME" "$PWD" 2>/dev/null || true)"
  if [[ -n "$_mlir_opt" ]]; then
    LLVM_INSTALL_DIR="$(dirname "$(dirname "$_mlir_opt")")"
  fi
fi

if [[ -z "${COMPILER_PY:-}" ]]; then
  COMPILER_PY="$(find /home/share "$HOME" -path '*triton-shared/backend/compiler.py' -type f 2>/dev/null | head -1 || true)"
fi

if [[ -z "${FLAGGEMS_DIR:-}" ]]; then
  FLAGGEMS_DIR="$(find /home/share "$HOME" -maxdepth 5 -type d -name FlagGems 2>/dev/null | head -1 || true)"
fi

if [[ -z "${DUMP_DIR:-}" ]]; then
  DUMP_DIR="$(find /home/share "$HOME" -type d -name bmm_kernel 2>/dev/null | head -1 || true)"
fi

for v in LLVM_INSTALL_DIR TRITON_SHARED_OPT COMPILER_PY; do
  if [[ -z "${!v:-}" || ! -e "${!v}" ]]; then
    echo "ERROR: $v 未找到。请先 source 现场 env.sh（会导出 TRITON_SHARED_OPT_PATH/LLVM 路径），" >&2
    echo "       或在 site.env 里显式填写 $v（示例见 site.env.example）。" >&2
    exit 2
  fi
done

PLUGIN_DIR="$(cd "$SCRIPT_DIR" && pwd)"
EXPERIMENT_OUT="${EXPERIMENT_OUT:-$(pwd)/onsite_experiment}"
RUN_ID="run-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$EXPERIMENT_OUT/$RUN_ID"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/REPORT.txt"
CURRENT_STEP="init"

on_error() {
  local status=$?
  {
    echo "FAIL: onsite prefetch experiment"
    echo "run_dir: $RUN_DIR"
    echo "failed_step: $CURRENT_STEP"
    echo "exit_status: $status"
  } > "$REPORT"
  echo "FAIL at $CURRENT_STEP; inspect $REPORT" >&2
  exit "$status"
}
trap on_error ERR

echo "run_dir: $RUN_DIR"

# ---------- 1. 形状确认 ----------
CURRENT_STEP="01_shape_check"
echo "[1/7] shape check M=8192 K=64 N=2048 batch=4 bf16"
if [[ -n "${FLAGGEMS_DIR:-}" && -d "${FLAGGEMS_DIR:-}" ]]; then
  ( cd "$FLAGGEMS_DIR" && python - <<'PY'
import torch
import flag_gems

# 1) shape 组合合法性（防 N/K 顺序错）：torch 本身验证，必须通过
a = torch.randn((4, 8192, 64), dtype=torch.bfloat16)
b = torch.randn((4, 64, 2048), dtype=torch.bfloat16)
ref = torch.bmm(a.float(), b.float())
assert ref.shape == (4, 8192, 2048), ref.shape
print("SHAPE_OK M=8192 K=64 N=2048 batch=4 bf16")

# 2) 现场 flag_gems 入口可用性：失败仅警告，正确性以第 5 步 pytest 为准
try:
    with flag_gems.use_gems():
        out = torch.bmm(a, b)
    assert out.shape == (4, 8192, 2048), out.shape
    err = (out.float() - ref).abs().max().item()
    if err < 1.0:
        print(f"FLAGGEMS_OK max_abs_err={err:.3f}")
    else:
        print(f"WARN: bf16 vs fp32 max_abs_err={err:.3f}（数值差异大，以 pytest 为准）")
except Exception as exc:
    print(f"WARN: flag_gems 入口检查跳过: {exc}")
PY
  ) 2>&1 | tee "$RUN_DIR/$CURRENT_STEP.log"
else
  echo "FLAGGEMS_DIR 未设置，跳过形状确认（确保现场测试参数为 M=8192 K=64 N=2048 batch=4 bf16）" \
    | tee "$RUN_DIR/$CURRENT_STEP.log"
fi

# ---------- 2. 硬件采集 ----------
CURRENT_STEP="02_hardware"
echo "[2/7] collect hardware info"
{
  echo "=== cpu ==="; lscpu 2>/dev/null || true
  echo "=== sme flag ==="; grep -m1 -o 'sme[^ ]*' /proc/cpuinfo 2>/dev/null || echo "sme flag not found"
  echo "=== cache ==="
  for i in 0 1 2 3; do
    base=/sys/devices/system/cpu/cpu0/cache/index$i
    if [[ -d $base ]]; then
      echo "index$i level=$(cat $base/level) size=$(cat $base/size 2>/dev/null) line=$(cat $base/coherency_line_size 2>/dev/null) shared=$(cat $base/shared_cpu_list 2>/dev/null)"
    fi
  done
  echo "=== OMP ==="; echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
  echo "=== rdvl ==="
  if command -v gcc >/dev/null 2>&1; then
    tmp=$(mktemp -d)
    printf 'int main(){unsigned long v;asm volatile("rdvl %%0, #1":"=r"(v));return v/16;}\n' > "$tmp/rdvl.c"
    gcc -march=armv8-a+sve "$tmp/rdvl.c" -o "$tmp/rdvl" 2>/dev/null && "$tmp/rdvl" && echo "SVL_bytes/16=$("$tmp/rdvl")" || echo "rdvl failed"
    rm -rf "$tmp"
  else
    echo "gcc not available, skip rdvl"
  fi
} 2>&1 | tee "$RUN_DIR/hardware.txt"

# ---------- 3. preflight ----------
CURRENT_STEP="03_preflight"
echo "[3/7] preflight tool versions"
bash "$PLUGIN_DIR/onsite_preflight.sh" \
  "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" "$RUN_DIR/preflight" \
  2>&1 | tee "$RUN_DIR/$CURRENT_STEP.log"

# ---------- 4. source-A 变体准备 ----------
CURRENT_STEP="04_prepare_source_a"
INPUT_00="${INPUT_00:-}"
if [[ -z "$INPUT_00" && -n "${DUMP_DIR:-}" ]]; then
  INPUT_00="$DUMP_DIR/00_input.mlir"
fi
if [[ -z "$INPUT_00" || ! -f "$INPUT_00" ]]; then
  echo "ERROR: INPUT_00 not found; set INPUT_00 in site.env (TRITON_SHARED_DUMP_PATH/bmm_kernel/00_input.mlir)" >&2
  exit 2
fi
echo "[4/7] prepare source-A variants (roundtrip / a8-e8-l2 / a8-e8-l1)"
bash "$PLUGIN_DIR/onsite_prepare_source_a.sh" \
  "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" "$COMPILER_PY" \
  "$INPUT_00" "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/$CURRENT_STEP.log"

# prepare_source_a creates its own source-a-<ts>-<pid>/ subdir below RUN_DIR.
PREPARE_ROOT="$(ls -dt "$RUN_DIR"/source-a-* 2>/dev/null | head -1)"
if [[ -z "$PREPARE_ROOT" ]]; then
  echo "ERROR: prepare_source_a output dir not found under $RUN_DIR" >&2
  exit 2
fi

# ---------- 5/6. 每变体：正确性 + 配对计时 ----------
run_variant() {
  local name=$1
  local llir_dir="$PREPARE_ROOT/$name/final"
  local override_dir="$RUN_DIR/override/$name"
  mkdir -p "$override_dir"
  cp "$llir_dir/kernel.llir" "$override_dir/kernel.llir"

  echo "[5/7] correctness: $name"
  if [[ -n "${FLAGGEMS_DIR:-}" && -d "${FLAGGEMS_DIR:-}" ]]; then
    (
      cd "$FLAGGEMS_DIR"
      TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR="$override_dir" \
        ${PYTEST_CMD:-pytest -s tests/test_blas_ops_mem.py::test_accuracy_bmm_membound[dtype2-8192-64-2048]}
    ) 2>&1 | tee "$RUN_DIR/$CURRENT_STEP.log"
  else
    echo "FLAGGEMS_DIR 未设置，跳过 pytest 正确性" | tee "$RUN_DIR/$CURRENT_STEP.log"
  fi

  echo "[6/7] paired benchmark: $name (baseline-first + prefetch-first)"
  local bench_log="$RUN_DIR/benchmark_$name.log"
  (
    export SME_BENCH_BATCH SME_BENCH_M SME_BENCH_N SME_BENCH_K SME_BENCH_DTYPE SME_BENCH_WARMUP SME_BENCH_REP
    SME_BENCH_ORDER=baseline-first bash "$PLUGIN_DIR/onsite_benchmark.sh" "$override_dir"
    SME_BENCH_ORDER=prefetch-first bash "$PLUGIN_DIR/onsite_benchmark.sh" "$override_dir"
  ) 2>&1 | tee "$bench_log"
}

for name in roundtrip a8-e8-l2 a8-e8-l1; do
  CURRENT_STEP="variant_$name"
  run_variant "$name"
done

# ---------- 7. 汇总 ----------
CURRENT_STEP="07_report"
echo "[7/7] build report"
{
  echo "PASS: onsite prefetch experiment"
  echo "run_dir: $RUN_DIR"
  echo "input_00: $INPUT_00"
  echo
  echo "== PREPARED (PRFM/FMOPA per variant) =="
  python3 - "$PREPARE_ROOT/PREPARED.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
for name, v in data["variants"].items():
    print(f"{name}: prfm={v['prfm']} fmopa={v['fmopa']} llir_sha={v['llir_sha256'][:12]}")
PY
  echo
  echo "== benchmark (paired baseline vs prefetch) =="
  for name in roundtrip a8-e8-l2 a8-e8-l1; do
    echo "-- $name --"
    grep -E 'RESULT_|SME_|Run order' "$RUN_DIR/benchmark_$name.log" 2>/dev/null | grep -E 'RESULT_|Latency' || true
  done
  echo
  echo "== hardware =="
  grep -E 'Model name|^model name|sme flag|index[0-9]|OMP_NUM_THREADS|SVL' "$RUN_DIR/hardware.txt" || true
} | tee "$REPORT"

trap - ERR
echo "done: $REPORT"
