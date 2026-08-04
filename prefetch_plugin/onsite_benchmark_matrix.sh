#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  onsite_benchmark_matrix.sh MATRIX_DIR

Benchmarks every override listed in MATRIX_DIR/matrix.tsv. Every variant is run
once baseline-first and once prefetch-first per repeat. Paired speedups are
combined with a geometric mean to cancel multiplicative run-order bias.

SME_MATRIX_REPEATS defaults to 1 (two paired measurements per variant). The
same SME_BENCH_* variables accepted by onsite_benchmark.sh may also be used.
EOF
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  usage
  exit 0
fi
if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

script_dir=$(cd "$(dirname "$0")" && pwd -P)
matrix_dir=$(cd "$1" && pwd -P)
manifest="$matrix_dir/matrix.tsv"
if [[ ! -f $manifest ]]; then
  echo "ERROR: matrix manifest not found: $manifest" >&2
  exit 2
fi

result_dir="$matrix_dir/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$result_dir"
summary="$result_dir/results.tsv"
printf 'variant\trepeat\torder\tbaseline_ms\tprefetch_ms\tspeedup\tlatency_change_percent\n' > "$summary"
repeats=${SME_MATRIX_REPEATS:-1}
if ! [[ $repeats =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SME_MATRIX_REPEATS must be a positive integer" >&2
  exit 2
fi

variant_count=0
while IFS=$'\t' read -r variant distance locality coverage issue_every line_bytes prfm fmopa override_dir; do
  if [[ $variant == variant ]]; then
    continue
  fi

  variant_count=$((variant_count + 1))
  for ((repeat = 1; repeat <= repeats; ++repeat)); do
    for order in baseline-first prefetch-first; do
      log="$result_dir/${variant}_r${repeat}_${order}.log"
      echo "=== Benchmarking $variant repeat=$repeat ($order) ==="
      SME_BENCH_ORDER=$order \
        bash "$script_dir/onsite_benchmark.sh" "$override_dir" 2>&1 | tee "$log"

      baseline=$(sed -n 's/^RESULT_BASELINE_MS=//p' "$log" | tail -n 1)
      prefetch=$(sed -n 's/^RESULT_PREFETCH_MS=//p' "$log" | tail -n 1)
      speedup=$(sed -n 's/^RESULT_SPEEDUP=//p' "$log" | tail -n 1)
      change=$(sed -n 's/^RESULT_LATENCY_CHANGE_PERCENT=//p' "$log" | tail -n 1)
      if [[ -z $baseline || -z $prefetch || -z $speedup || -z $change ]]; then
        echo "ERROR: incomplete benchmark result for $variant" >&2
        exit 1
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$variant" "$repeat" "$order" "$baseline" "$prefetch" \
        "$speedup" "$change" >> "$summary"
    done
  done
done < "$manifest"

if [[ $variant_count -eq 0 ]]; then
  echo "ERROR: matrix contains no variants" >&2
  exit 1
fi

python - "$summary" <<'PY'
import csv
import math
import statistics
import sys

path = sys.argv[1]
with open(path, newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))

groups = {}
for row in rows:
    groups.setdefault(row["variant"], []).append(row)

aggregates = []
for variant, samples in groups.items():
    baselines = [float(sample["baseline_ms"]) for sample in samples]
    overrides = [float(sample["prefetch_ms"]) for sample in samples]
    ratios = [float(sample["speedup"].rstrip("x")) for sample in samples]
    paired_speedup = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
    aggregates.append(
        {
            "variant": variant,
            "samples": len(samples),
            "baseline": statistics.median(baselines),
            "override": statistics.median(overrides),
            "paired_speedup": paired_speedup,
        }
    )

control = next(
    (item for item in aggregates if item["variant"] == "control_roundtrip"),
    None,
)
control_latency = control["override"] if control else None
for item in aggregates:
    item["vs_control"] = (
        control_latency / item["override"] if control_latency is not None else float("nan")
    )

aggregates.sort(key=lambda item: item["vs_control"], reverse=True)
print("RANK\tVARIANT\tSAMPLES\tBASELINE_MEDIAN\tOVERRIDE_MEDIAN\tPAIRED_NATIVE_SPEEDUP\tVS_CONTROL")
for rank, item in enumerate(aggregates, 1):
    print(
        f'{rank}\t{item["variant"]}\t{item["samples"]}\t'
        f'{item["baseline"]:.6f}\t{item["override"]:.6f}\t'
        f'{item["paired_speedup"]:.6f}x\t{item["vs_control"]:.6f}x'
    )
PY

echo "MATRIX_BENCHMARK_RESULTS=$summary"
