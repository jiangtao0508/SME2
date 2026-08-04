# SME FMOPA timing calibration

This native AArch64 probe measures two different quantities with the
architected counter `CNTVCT_EL0`:

- repeated FMOPA on one ZA tile, which exposes the dependency-chain cost;
- four interleaved ZA tiles, which estimates sustained FMOPA throughput.

`SMSTART`, `SMSTOP`, counter reads and loop overhead are outside or subtracted
from the reported per-FMOPA values. `RDSVL` records the actual streaming vector
length, so the profile can also report floating-point work per FMOPA.

Run on the target SME machine:

```bash
bash sme_timing/run_sme_timing.sh /path/to/project-output/sme-timing
```

The output is `SmeTimingProfile.v1.json`. The probe contains no Triton,
FlagGems, input tensor, IR or kernel source data.

The four-tile result is a compute lower bound for a real GEMM K step. The Cost
Model must combine it with the static number of FMOPA operations and the
measured memory-transfer lower bound; it must not call the FMOPA number itself
the complete K-step time.

