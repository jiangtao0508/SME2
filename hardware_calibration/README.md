# Hardware calibration for the prefetch cost model

This directory measures hardware primitives once per target platform. It does
not inspect or copy Triton, FlagGems, MLIR, LLVM IR, objects, or test data.

The full calibration performs:

- dependent randomized pointer chasing over increasing working sets;
- single-thread sequential read bandwidth;
- cold single-pass stride measurements;
- hot-buffer loops with no prefetch and `issue_every=1/2/4`;
- Linux sysfs or macOS sysctl cache discovery;
- assembly verification that the compiler retained a PRFM/prefetch operation.

Run a quick smoke test:

```bash
bash hardware_calibration/run_calibration.sh quick /tmp/sme-hardware-quick
```

Run the measured profile used by the cost model:

```bash
bash hardware_calibration/run_calibration.sh full /path/to/project-output/hardware
```

Outputs:

```text
HardwareProfile.v1.1.json
HARDWARE_CALIBRATION_SUMMARY.txt
```

The JSON keeps raw repeated measurements, medians, derived values, provenance,
and explicit warnings. Nanosecond values are authoritative. Cycle estimates
are emitted only when a platform frequency can be read and must be treated as
lower-confidence under DVFS.

`hardware_prefetch_effective_stride_bytes_heuristic` is a coarse stride-sweep
indicator, not a PMU-derived coverage probability. Likewise,
`max_outstanding_prefetches` remains `null` until a dedicated saturation probe
is implemented; the tool does not fabricate unavailable hardware values.

The calibration output is a numeric onsite artifact. Whether it can leave a
restricted platform is governed by that platform's data policy.
