# Kunpeng MM + SME 软件预取主线

项目的性能研究主线从固定 `4x4x4` BMM 转向 Kunpeng MM。BMM 工具继续作为
pass/override 链路验证，不再用来代表成熟 CPU GEMM 的预取收益。

Kunpeng MM 现有 `256x256x256` 与 `8x8x8` 两个宏块候选；FlagGems `libtuner`
按照 M、N、K 和关键 stride 选择配置，Triton-Shared 再按照目标硬件 SVL 将
宏块切成 SME 微块。软件预取必须分析最终选中的配置，不能假设一定为 256 或 8。

现场使用：

```bash
bash scripts/capture_selected_kunpeng_mm.sh \
  /绝对路径/triton-cpu \
  /绝对路径/现场输出目录 \
  8192 2048 64 bfloat16
```

脚本先执行原生 `libtuner`，随后在新进程中仅保留胜出配置并强制重新编译，避免
多个候选依次编译时最后一个候选覆盖 dump。最终打印 `MM_SELECTED`、逐层
`MM_IR` 算子计数、`MM_00` 和完整 dump 目录。现场不能带出 IR 时，只需记录这些
短摘要行；不要手抄 IR 正文。

取得真实 MM IR 后按以下顺序推进：

1. 从 TT/TTShared 记录宏块、grid、K trip 和 A/B 跨 program 复用；
2. 从 bufferize 后 IR 定位原始 A/B 到 private tile 的第一次真实读取；
3. 计算动态 PRFM 数量和重复预取倍数，允许 Cost Model 输出 `NONE`；
4. 由 cache-line 周期确定 `issue_every`，由延迟/K-step 时间确定 distance；
5. 只生成 roundtrip 与模型唯一预测版本；
6. 正确性、结构审计和 PRFM/FMOPA 守恒通过后才计时。

不要直接把 BMM 的 `argument-index=0, distance=8, issue-every=8` 套到 MM。MM 的
宏块、尾块、stride 和 bufferize 结构均需由本次真实 dump 确认。
