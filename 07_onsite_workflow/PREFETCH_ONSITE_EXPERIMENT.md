# 现场预取实验方案（BMM source-A）

> 目标：在现场（Kunpeng SME，bf16 BMM 8192×64×2048 batch=4）验证软件预取
> 的正确性与性能。主线是 **source-A**（`prefetch-bmm-source`，对原始 A 面板
> 插 4 条静态 prefetch）；旧 `prefetch-gemm-rhs`（packed RHS private alloc）
> 只保留作链路对照，**不用于性能结论**。
>
> 配套配置：`07_onsite_workflow/config/onsite_hardware_profile.json` 与
> `onsite_bmm_case.json`（宏观参数已确认，硬件待现场实测填充）。
>
> 插件/benchmark 脚本统一在 `SME2-public/prefetch_plugin/`（现场用 git 克隆
> 下来的 `prefetch_plugin/` 路径）。

### 形状确认（第一件事，防止 N/K 顺序错误）

现场测试 `8192-64-2048` 已确认是 **M=8192, K=64, N=2048**：

```text
A[4, 8192, 64] × B[4, 64, 2048] = O[4, 8192, 2048]   （bf16）
A 行 stride = K*2 = 128 B，B 行 stride = N*2 = 4096 B
```

`onsite_benchmark.sh` 默认值已按此修正；现场跑计时前先用以下命令核对，若
与现场实际不符立即停止并报告（预取距离计算完全依赖 K 值）：

```bash
python - <<'PY'
import torch, flag_gems
a = torch.randn((4, 8192, 64), dtype=torch.bfloat16)
b = torch.randn((4, 64, 2048), dtype=torch.bfloat16)
with flag_gems.use_gems():
    out = torch.bmm(a, b)
assert out.shape == (4, 8192, 2048)
print("shape OK: M=8192 K=64 N=2048 batch=4")
PY
```

## 0. 前置核对（先做，约 30 分钟）

### 0.1 环境

```bash
export LLVM_INSTALL_DIR=/absolute/path/to/llvm20/llvm-project/install
export TRITON_SHARED_OPT=/absolute/path/to/triton-cpu/python/build/cmake.linux_aarch64-cpython-3.11/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
export COMPILER_PY=/absolute/path/to/triton-cpu/triton-shared/backend/compiler.py
export FLAGGEMS_DIR=/absolute/path/to/FlagGems
export TEST_NODE='test/test_blas_ops_mem.py::test_accuracy_bmm_membound[dtype2-8192-64-2048]'

"$TRITON_SHARED_OPT" --version
"$LLVM_INSTALL_DIR/bin/mlir-translate" --version
git -C "$FLAGGEMS_DIR" rev-parse HEAD
git -C /absolute/path/to/triton-cpu rev-parse HEAD
```

### 0.2 硬件实测（结果填进 config/onsite_hardware_profile.json，置 calibrated:true）

```bash
lscpu
grep -m1 -o 'sme[^ ]*' /proc/cpuinfo        # 确认 SME 特性
cat /sys/devices/system/cpu/cpu0/cache/index0/size      # L1d
cat /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size
cat /sys/devices/system/cpu/cpu0/cache/index2/size      # L2
cat /sys/devices/system/cpu/cpu0/cache/index2/shared_cpu_list  # L2 共享域
echo $OMP_NUM_THREADS                        # 线程配置
```

### 0.3 确认测试可跑、dump 路径

```bash
cd "$FLAGGEMS_DIR"
pytest -s "$TEST_NODE"
# 记录：pytest 时间、dumps 真实路径、cache 真实路径、是否命中缓存
```

用 SME2 现场分析确认链路：

```bash
bash scripts/run_onsite_analysis.sh \
  --dump /absolute/path/to/dumps \
  --case-name bmm_8192_64_2048 \
  --test-workdir "$FLAGGEMS_DIR" \
  --test-command "pytest -s '$TEST_NODE'"
```

## 1. 基线（正确性优先，不做预取）

### 1.1 native 基线

上面 0.3 的 pytest 就是 native（0 PRFM）。保存结果并记录：

```text
PRFM=0, FMOPA=N（从 kernel.o 反汇编数），正确性通过
```

### 1.2 roundtrip（split-replay 无预取，证明工作流不破坏 SME）

```bash
bash prefetch_plugin/onsite_prepare_source_a.sh \
  "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" "$COMPILER_PY" \
  /absolute/path/to/dumps/00_input.mlir \
  /absolute/path/to/experiment
```

脚本产出 `roundtrip/final/kernel.llir`（无 PRFM）与 `PREPARED.json`。检查：

```bash
grep -c fmopa <(llvm-objdump -d roundtrip/final/kernel.o)   # 必须 = native 的 N
grep -c prfm  <(llvm-objdump -d roundtrip/final/kernel.o)   # 必须 = 0
```

**停止条件 A**：roundtrip 与 native 已存在稳定差异 → 停止，问题在工具链，不是预取。

## 2. source-A 变体准备与审计

`onsite_prepare_source_a.sh` 已生成两个候选 + roundtrip：

| 变体 | distance | issue-every | locality | 目标 |
|---|---:|---:|---:|---|
| `a8-e8-l2` | 8 | 8 | 2 | A 面板，L2 |
| `a8-e8-l1` | 8 | 8 | 3 | A 面板，L1 |

每个变体目录有 `source_a_audit.json`，脚本已拒绝以下情况：

```text
没有唯一 %arg0 -> reinterpret_cast -> memref.copy -> private alloc 链
tile 不是静态 4x4
预取数量不是 4 条
预取指向 %alloc 而不是原始 A
distance/issue-every/locality 与候选不一致
prefetch 出现在第一次 memref.copy 之后
SME lowering 后没有同时保留 LLVM prefetch 和 MOPA
roundtrip object 意外出现 PRFM
```

## 3. 正确性验证（每个变体，先正确性再计时）

顺序固定：**native → roundtrip → a8-e8-l2 → a8-e8-l1**。

每步只安装一个 LLIR 到 override 目录，设 `TRITON_ALWAYS_COMPILE=1`，**亲眼确认
override 命中日志**（`Overriding kernel with file ...`），然后跑：

```bash
export TRITON_ALWAYS_COMPILE=1
cd "$FLAGGEMS_DIR"
pytest -s "$TEST_NODE"
```

任一变体正确性不过 → 保存日志，**不手改 IR 绕过**，停止该变体。

## 4. 计时（交错，防漂移）

用 SME2 的 benchmark 工具（基于 `triton.testing.do_bench`，内置 baseline/
prefetch 配对 + 独立 cache）：

```bash
# 单个变体：baseline 与 prefetch 配对（一次 baseline-first + 一次 prefetch-first）
SME_BENCH_N=2048 SME_BENCH_K=64 SME_BENCH_WARMUP=5 SME_BENCH_REP=20 \
bash prefetch_plugin/onsite_benchmark.sh \
  /absolute/path/to/experiment/a8-e8-l2/final
```

矩阵版（每个变体 baseline-first + prefetch-first 配对，几何平均消除顺序偏
差；用 `SME_MATRIX_REPEATS` 控制配对次数，默认 1）：

```bash
bash prefetch_plugin/onsite_benchmark_matrix.sh \
  /absolute/path/to/experiment/matrix
```

`matrix.tsv` 每行：`variant distance locality coverage issue_every line_bytes prfm fmopa override_dir`。
重复配对 ≥8 组（`SME_MATRIX_REPEATS=8`）达到 TOMORROW_SOURCE_A 的"至少 8 组"
要求；每个进程 warmup 5~10、正式测量 ≥20 次（`SME_BENCH_WARMUP/SME_BENCH_REP`）。
报告配对比值（prefetch/baseline）与 p25/p75。

有 perf 权限时同时记录：

```bash
perf stat -e cycles,instructions,cache-references,cache-misses <原计时命令>
```

无权限只记录错误，不改变平台设置。

## 5. 参数扫描（仅当 d8 有效）

确定可用 distance 后，固定 distance/层级，再扫 coverage 与 issue-every：

```bash
for distance in 1 2 4 8; do
  bash prefetch_plugin/onsite_split_replay.sh \
    "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" \
    /absolute/path/to/dumps/00_input.mlir \
    "/absolute/path/to/experiment/d-$distance" \
    bmm-source-a "$distance" 3 4 8 64      # locality=3, coverage=4, issue-every=8
done

for coverage in 1 2 4; do
  for issue_every in 1 2; do
    bash prefetch_plugin/onsite_split_replay.sh \
      "$LLVM_INSTALL_DIR" "$TRITON_SHARED_OPT" \
      /absolute/path/to/dumps/00_input.mlir \
      "/absolute/path/to/experiment/c-$coverage-e-$issue_every" \
      bmm-source-a 8 3 "$coverage" "$issue_every" 64
  done
done
```

每个变体 LLIR 放入**各自独立** override 目录（benchmark 脚本已为每 label 建
独立 `TRITON_CACHE_DIR`，不会共用）。coverage 扫描仍只针对 A；B 面板在 A
定案后再决定。

## 6. 每变体记录字段

| 字段 | 内容 |
|---|---|
| kernel/shape/dtype | bmm_kernel / 8192-64-2048 / bf16 |
| Triton/Triton-Shared commit | 现场 rev-parse |
| LLVM commit/version | 现场确认 |
| CPU 型号、SVL、频率策略 | 0.2 实测 |
| 线程数、绑核 | 填 config/runtime_onsite_fill |
| tile/autotune 配置 | 4x4x4 / GROUP_M=1 |
| variant、distance、locality、coverage、issue_every | |
| LLIR SHA-256 / object SHA-256 | `sha256sum` |
| PRFM / FMOPA 数量 | objdump |
| correctness / max error | pytest |
| warmup 次数 / measurement 次数 | |
| median latency / p25 / p75 | |
| 相对 baseline | 配对比值 |

规则：baseline 与所有变体用完全相同输入与编译配置；`TRITON_ALWAYS_COMPILE=1`；
先正确性再计时；warmup 不计入；**负收益和无收益也保留**，不只留最佳。

## 7. 停止条件

- **A**：roundtrip 与 native 有稳定差异 → 停止，不能归因于预取；
- **B**：audit 失败 → 保存日志，不手改 IR 绕过；
- **C**：FMOPA 数量变化 → 停止；
- **D**：cache miss 不下降且时间无收益 → 不再扫 distance；
- **E**：miss 下降但时间变慢 → 下一步优化动态判断/PRFM 数量，不扩大参数矩阵。

## 8. 现场交付物

```text
experiment/<run-id>/
  PREPARED.json          # 输入/LLIR/object 哈希 + PRFM/FMOPA
  roundtrip/ a8-e8-l1/ a8-e8-l2/
    final/kernel.llir, kernel.o
    source_a_audit.json
  timing/                # 每组 raw 计时 + median/p25/p75 + 配对比值
  perf/                  # cache-misses 等（有权限时）
  ONSITE_PREFETCH_RESULT.md  # 结论 + 负收益保留
```
