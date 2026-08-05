# SME BFMOPA timing calibration

This native AArch64 probe measures two different quantities with the
architected counter `CNTVCT_EL0`:

- repeated BF16-to-F32 BFMOPA on one ZA tile, which exposes the dependency-chain cost;
- four interleaved ZA tiles, which estimates sustained BFMOPA throughput.

`SMSTART`, `SMSTOP`, counter reads and loop overhead are outside or subtracted
from the reported per-BFMOPA values. `RDSVL` records the actual streaming vector
length, so the profile can also report floating-point work per BFMOPA. The
probe uses the same BF16-to-F32 operation class as the current onsite BMM.

Run on the target SME machine:

```bash
bash sme_timing/run_sme_timing.sh /path/to/project-output/sme-timing
```

The output is `SmeTimingProfile.v1.json`. The probe contains no Triton,
FlagGems, input tensor, IR or kernel source data.

If a restricted platform raises `SIGILL` for `CNTVCT_EL0`, the runner
automatically repeats the same SME loops with `CLOCK_MONOTONIC_RAW`. If that
second run also raises `SIGILL`, the failure is an SME execution or feature
issue rather than a counter-access restriction.

The four-tile result is a compute lower bound for a real GEMM K step. The Cost
Model must combine it with the static number of BFMOPA operations and the
measured memory-transfer lower bound; it must not call the BFMOPA number itself
the complete K-step time.
