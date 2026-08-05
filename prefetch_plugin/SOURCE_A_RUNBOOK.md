# 明日现场：原始 A 地址预取

## 目标

只验证一个可解释的决策：对 `TILE_K=4`、`bf16`、64 字节 cache line，
每行每次 K 迭代消费 8 字节，因此在 K 迭代 0 预取迭代 8 的下一条 A cache
line。四行分别预取，共四条静态 `memref.prefetch`。

旧 `prefetch-gemm-rhs` 以 private `memref.alloc` 为候选，只作为链路验证，
不再用于本轮性能结论。本轮必须使用 `prefetch-bmm-source`。

## 唯一准备命令

从克隆后的 `prefetch_model` 根目录运行：

```bash
bash prefetch_plugin/onsite_prepare_source_a.sh \
  /绝对路径/llvm-install \
  /绝对路径/triton-shared-opt \
  /绝对路径/triton-shared/backend/compiler.py \
  /绝对路径/本次真实00.mlir \
  /绝对路径/现场输出父目录
```

脚本自动创建带时间和 PID 的新目录，不会因旧输出目录存在而失败。它生成：

- `roundtrip/final/kernel.llir`：相同 split/replay，无 PRFM；
- `a8-e8-l2/final/kernel.llir`：原始 A，distance 8，频率 8，L2；
- `a8-e8-l1/final/kernel.llir`：原始 A，distance 8，频率 8，L1；
- `PREPARED.json`：输入、LLIR、object 的 SHA-256 和 PRFM/FMOPA 数量；
- 每个预取目录中的 `source_a_audit.json`。

## 结构审计通过的含义

脚本会拒绝以下情况：

- 没有唯一的 `%arg0 -> reinterpret_cast -> memref.copy -> private alloc`；
- tile 不是静态 `4x4`；
- 预取数量不是四条；
- 预取指向 `%alloc`；
- distance、issue-every 或 locality 与候选不一致；
- prefetch 出现在第一次 `memref.copy` 之后；
- SME lowering 后没有同时保留 LLVM prefetch 和 MOPA；
- roundtrip object 意外出现 PRFM。

## Override 与正确性

仍使用已经在现场验证成功的 hash 获取和 override 方法。每次只安装一个 LLIR，
设置 `TRITON_ALWAYS_COMPILE=1`，并亲眼确认 override 命中日志。

顺序固定为：

1. native；
2. roundtrip；
3. A8-E8-L2；
4. A8-E8-L1。

每个版本先跑同一个原生 pytest 正确性用例。任一版本不通过，不进入计时。

## 计时

不要把所有 baseline 放前面、所有 prefetch 放后面。使用交错顺序：

```text
roundtrip, L2, L2, roundtrip
L1, roundtrip, roundtrip, L1
```

重复至少 8 组；每个进程 warmup 5～10 次，正式测量至少 20 次。记录每组
median，最后报告配对比值及 p25/p75。

如果 `perf stat` 有权限，同时记录：

```bash
perf stat -e cycles,instructions,cache-references,cache-misses <原计时命令>
```

无权限只记录错误，不改变平台设置。

## 停止条件

- roundtrip 与 native 已有稳定差异：停止，不能归因于预取；
- audit 失败：保存日志，不手改 IR 绕过；
- FMOPA 数量变化：停止；
- cache miss 不下降且时间无收益：不再扫描 distance；
- miss 下降但时间变慢：下一步优化动态判断/PRFM 数量，不扩大参数矩阵。
