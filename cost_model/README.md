# Formula-driven packed GEMM RHS planner

This planner consumes only numeric inputs:

- `HardwareProfile.v1.1.json` from `hardware_calibration/`;
- `GemmKernelProfile.v1.json` from `prefetch-analyze-gemm-rhs`;
- the measured or otherwise justified time of one matched K-loop step.

It does not enumerate a benchmark matrix. The selected values are derived as:

```text
distance       = ceil(memory_latency_ns / anchor_step_ns)
coverage_lines = ceil(vector_read_bytes / cache_line_bytes)
issue_every    = max(issue-cost bound, bandwidth bound, outstanding bound)
```

If required timing is missing, the command fails instead of substituting a
guess. If the derived distance is outside the K-loop, it emits an explicit
no-prefetch plan.

Run:

```bash
python3 cost_model/plan_gemm_rhs.py \
  --hardware /path/to/HardwareProfile.v1.1.json \
  --kernel /path/to/GemmKernelProfile.v1.json \
  --anchor-step-ns <measured-k-step-ns> \
  --output /path/to/PrefetchPlan.v1.1.json
```

`anchor_step_ns` must describe the matched K-loop step. A whole pytest wall
time must not be used: it includes dispatch, JIT, data generation and many
parallel kernel instances. Until the dedicated SME throughput/PMU probe is
available, this input remains explicit so that the model cannot silently turn
an unrelated wall time into a prefetch distance.

PrefetchPlan 1.1 adds per-decision `emission.issue_every` and
`emission.coverage_lines`. `onsite_from_plan.sh` forwards both values to the
MLIR pass. PrefetchPlan 1.0 remains accepted and defaults to `issue_every=1`.

