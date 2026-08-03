# Prefetch pass plugin

This project-owned MLIR pass plugin can be loaded by an MLIR-compatible driver.
It does not modify or install into LLVM, Triton, or Triton-Shared. Some
statically linked `triton-shared-opt` builds do not share their pass registry
with a plugin built against the LLVM installation. For those builds, the
supported workflow runs the plugin in LLVM's `mlir-opt` between two replayed
parts of the original Triton Transform schedule.

The first smoke-test implementation supports a deliberately narrow pattern:

- one ranked memref function argument;
- one `scf.for`;
- a `memref.load` indexed directly by that loop's induction variable;
- a positive distance in loop iterations;
- a guarded read/data `memref.prefetch`.

It is a plugin/IR-lowering proof, not yet the real BMM resolver.

The plugin now also contains:

- `prefetch-snapshot`: writes the exact bufferized `func.func` seen between
  bufferization and ArmSME lowering;
- `prefetch-gemm-rhs`: an experimental structural matcher for the packed RHS
  panel in the public GEMM reproducer.

The public GEMM test completes the full chain:

```text
prefetch-gemm-rhs
-> memref.prefetch
-> llvm.intr.prefetch
-> AArch64 PRFM
```

## Build

```bash
cmake -S prefetch_plugin -B prefetch_plugin/build \
  -DMLIR_DIR=/absolute/path/to/lib/cmake/mlir \
  -DLLVM_DIR=/absolute/path/to/lib/cmake/llvm
cmake --build prefetch_plugin/build
```

## Smoke test

```bash
triton-shared-opt \
  --load-pass-plugin=/absolute/path/to/PrefetchPassPlugin.dylib \
  --pass-pipeline='builtin.module(func.func(prefetch-materialize{argument-index=0 distance=4 locality=3}))' \
  prefetch_plugin/test/simple_stream.mlir
```

Linux builds normally produce `PrefetchPassPlugin.so` instead of `.dylib`.

For a portable build plus direct-pass, Transform Interpreter, LLVM IR, and
AArch64 instruction smoke test, run:

```bash
bash prefetch_plugin/build_and_smoke.sh \
  /absolute/path/to/llvm-install \
  /absolute/path/to/triton-shared-opt
```

If `mlir-opt` succeeds but `triton-shared-opt` reports that
`prefetch-materialize` is not registered, verify and use the split workflow:

```bash
bash prefetch_plugin/build_and_smoke_mlir_opt.sh \
  /absolute/path/to/llvm-install

bash prefetch_plugin/onsite_split_replay.sh \
  /absolute/path/to/llvm-install \
  /absolute/path/to/triton-shared-opt \
  /absolute/path/to/00_input.mlir \
  /absolute/path/to/output \
  gemm-rhs 4 3
```

The split is bufferization -> `mlir-opt` prefetch insertion -> resumed ArmSME
lowering. It preserves the platform pipeline without requiring the custom pass
to appear in `triton-shared-opt --help`.

For the complete guarded workflow, including tool/version preflight, a
no-prefetch split round-trip, PrefetchPlan materialization, LLVM/object
generation, and a compact result file, run:

```bash
bash prefetch_plugin/onsite_full_experiment.sh \
  /absolute/path/to/llvm-install \
  /absolute/path/to/triton-shared-opt \
  /absolute/path/to/triton-shared/backend/compiler.py \
  /absolute/path/to/00_input.mlir \
  /absolute/path/to/output \
  /absolute/path/to/PrefetchPlan.json
```

The final summary is `output/ONSITE_PREFETCH_RESULT.txt`. The PrefetchPlan
adapter currently accepts one `TILE_PREFETCH` decision with `ITERATION`
distance and `PANEL`/`TILE` granularity. Its explicit MVP cache mapping is
`L1 -> locality 3` and `L2 -> locality 2`, matching LLVM 20's AArch64
lowering (`1` denotes L3 keep and `0` denotes streaming).

`test_gemm_rhs_types.sh` verifies that the structural matcher accepts f16,
bf16, f32, and f64 synthetic bufferized panels. This is intentionally reported
as matcher coverage, not as four complete ArmSME pipelines. It also exercises
the cross-product of distances `1,2,4,8` and localities `0,1,2,3`.

See `ONSITE_RUNBOOK.md` for the no-platform-modification workflow.
