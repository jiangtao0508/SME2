#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  onsite_benchmark_matrix.sh MATRIX_DIR

Benchmarks every override listed in MATRIX_DIR/matrix.tsv. Each variant gets a
paired baseline measurement. Run order alternates between baseline-first and
prefetch-first to reduce thermal/order bias. Results are written below
MATRIX_DIR and ranked by relative speedup.

The same SME_BENCH_* variables accepted by onsite_benchmark.sh may be used.
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
printf 'variant\torder\tbaseline_ms\tprefetch_ms\tspeedup\tlatency_change_percent\n' > "$summary"

index=0
while IFS=$'\t' read -r variant distance locality coverage issue_every line_bytes prfm fmopa override_dir; do
  if [[ $variant == variant ]]; then
    continue
  fi

  if (( index % 2 == 0 )); then
    order=baseline-first
  else
    order=prefetch-first
  fi
  index=$((index + 1))
  log="$result_dir/$variant.log"

  echo "=== Benchmarking $variant ($order) ==="
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$variant" "$order" "$baseline" "$prefetch" "$speedup" "$change" >> "$summary"
done < "$manifest"

if [[ $index -eq 0 ]]; then
  echo "ERROR: matrix contains no variants" >&2
  exit 1
fi

python - "$summary" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))

rows.sort(key=lambda row: float(row["speedup"].rstrip("x")), reverse=True)
print("RANK\tVARIANT\tBASELINE_MS\tPREFETCH_MS\tSPEEDUP\tCHANGE")
for rank, row in enumerate(rows, 1):
    print(
        f'{rank}\t{row["variant"]}\t{row["baseline_ms"]}\t'
        f'{row["prefetch_ms"]}\t{row["speedup"]}\t'
        f'{row["latency_change_percent"]}'
    )
PY

echo "MATRIX_BENCHMARK_RESULTS=$summary"
