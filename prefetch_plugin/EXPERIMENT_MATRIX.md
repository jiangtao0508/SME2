# 现场实验矩阵

先完成 baseline 正确性与计时，再运行预取变体。第一轮只改变 distance，不同时
改变 locality、线程、绑核、输入、tile 或 autotune 配置。

| 变体 | Distance | Locality | 目标 | 必须检查 |
|---|---:|---:|---|---|
| baseline | 0 | - | 无预取 | 正确性、FMOPA、0 PRFM |
| rhs-d1 | 1 | 3 | packed RHS | 正确性、PRFM |
| rhs-d2 | 2 | 3 | packed RHS | 正确性、PRFM |
| rhs-d4 | 4 | 3 | packed RHS | 正确性、PRFM |
| rhs-d8 | 8 | 3 | packed RHS | 正确性、PRFM |

生成不同距离的 split replay 产物：

```bash
for distance in 1 2 4 8; do
  bash prefetch_plugin/onsite_split_replay.sh \
    "$LLVM_INSTALL_DIR" \
    "$TRITON_SHARED_OPT" \
    /absolute/path/to/00_input.mlir \
    "/absolute/path/to/results/d-$distance" \
    gemm-rhs "$distance" 3 1 1 64
done
```

确定一个可用distance后，固定distance和层级，再扫描覆盖范围与频率：

```bash
for coverage in 1 2 4; do
  for issue_every in 1 2; do
    bash prefetch_plugin/onsite_split_replay.sh \
      "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" \
      /absolute/path/to/00_input.mlir \
      "/absolute/path/to/results/c-$coverage-e-$issue_every" \
      gemm-rhs 4 3 "$coverage" "$issue_every" 64
  done
done
```

每个变体继续执行 `onsite_stage2.sh`，并把 `kernel.llir` 放入各自
独立的 override/cache 目录。不要让两个变体共用 Triton cache。

每次实验记录：

| 字段 | 内容 |
|---|---|
| kernel/shape/dtype | |
| Triton/Triton-Shared commit | |
| LLVM commit/version | |
| CPU 型号、SVL、频率策略 | |
| 线程数、绑核 | |
| tile/autotune 配置 | |
| variant、distance、locality、coverage、issue_every | |
| LLIR SHA-256 | |
| object SHA-256 | |
| PRFM/FMOPA 数量 | |
| correctness/max error | |
| warmup 次数 | |
| measurement 次数 | |
| median latency | |
| p25/p75 | |
| 相对 baseline | |

测量规则：

- baseline 与所有变体使用完全相同输入和编译配置；
- 设置 `TRITON_ALWAYS_COMPILE=1`，并确认 override 日志命中；
- 先做正确性，再计时；
- warmup 不计入结果；
- 至少记录 median 和 p25/p75；
- 负收益和无收益也保留，不只保留最佳结果；
- coverage扫描仍只针对RHS；完成后再决定是否扩展A panel。
