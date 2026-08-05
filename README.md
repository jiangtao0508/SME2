# SME2 Linux现场编译链分析工具

这是一个面向 Linux 命令行现场环境的只读分析工具，用于整理
FlagGems/Triton/MLIR/LLVM/Arm SME 编译中间产物。

工具不会修改输入 dump，也不会把原始源码或完整 IR 复制进报告目录。它会生成：

- 环境、CPU、Cache、Python包和编译工具版本；
- 中间产物的相对路径、大小和 SHA256；
- 编号 MLIR 与 Triton Cache IR 的相邻层变化；
- Arm SME、LLVM SME、MOPA/FMOPA 的静态证据；
- `.o/.obj/.so` 的 `file/readelf/nm/objdump` 分析；
- 可用于现场交流的中文报告。

## 克隆与自检

```bash
git clone https://github.com/jiangtao0508/SME2.git
cd SME2
bash scripts/check_layout.sh
bash scripts/run_onsite_selftest.sh
```

自检只使用程序运行时创建的合成数据，不包含任何第三方源码、IR 或二进制。

## 最简分析

```bash
bash scripts/run_onsite_analysis.sh \
  --dump /absolute/path/to/dumps \
  --case-name bmm_8192_64_2048
```

输出位于：

```text
onsite_results/<case-name>-<timestamp>/
```

优先阅读：

```text
ONSITE_REPORT_CN.md
04_ir_transitions.md
05_sme_evidence.txt
06_binary/binary_summary.json
```

## 从pytest追踪到文件生产者

需要搞清楚pytest插件、测试函数、子进程和每个中间文件的生产者时：

```bash
PYTHON_BIN=/path/to/their/python \
bash scripts/run_full_pytest_trace.sh \
  --workdir /path/to/FlagGems \
  --nodeid 'test/test_blas_ops_mem.py::test_accuracy_bmm_membound[dtype2-8192-64-2048]' \
  --watch-root /path/to/dumps \
  --dump /path/to/dumps \
  --scan-root /path/to/FlagGems \
  --scan-root /path/to/compiler/source-or-package \
  --strace auto \
  --run
```

首先阅读`PYTEST_PIPELINE_REPORT_CN.md`。完整说明见
[docs/pytest到中间文件完整追踪.md](docs/pytest到中间文件完整追踪.md)。
公开FlagGems机制与现场待证实内容的边界见
[docs/已知BMM调用链与现场待证实项.md](docs/已知BMM调用链与现场待证实项.md)。

## 运行时预取 Pass 插件

`prefetch_plugin/` 是独立于平台源码构建的 MLIR Pass 插件，不修改或安装到
LLVM、Triton、Triton-Shared。已在公开 Triton CPU SME 流程上验证：

```text
bufferize 后、ArmSME 前加载插件
-> 导出 bufferized snapshot
-> 识别 packed GEMM RHS panel
-> 插入 memref.prefetch
-> 生成 llvm.intr.prefetch
-> AArch64 对象中同时存在 PRFM 与 FMOPA
```

现场从 [prefetch_plugin/ONSITE_RUNBOOK.md](prefetch_plugin/ONSITE_RUNBOOK.md)
开始。推荐直接运行带预检、无预取 round-trip 和分阶段日志的一键流程：

```bash
bash prefetch_plugin/onsite_full_experiment.sh \
  /path/to/llvm-install \
  /path/to/triton-shared-opt \
  /path/to/triton-shared/backend/compiler.py \
  /path/to/00_input.mlir \
  /path/to/project-output \
  /path/to/PrefetchPlan.json
```

如果 `mlir-opt` 成功，而 `triton-shared-opt` 报 pass 未注册，说明两者没有
共享 MLIR Pass Registry。无需修改平台源码，使用分段 replay：

```bash
bash prefetch_plugin/onsite_split_replay.sh \
  /path/to/llvm-install \
  /path/to/triton-shared-opt \
  /path/to/00_input.mlir \
  /path/to/project-output \
  gemm-rhs 4 3
```

最终结论写入 `project-output/ONSITE_PREFETCH_RESULT.txt`；任何失败都会标明
具体阶段及对应日志。`mlir-opt` 只负责运行项目插件，`tptr-to-llvm`、ArmSME
和 Transform schedule 始终由 `triton-shared-opt` 执行。

RHS策略现支持distance、L1/L2层级、连续cache-line覆盖数和K循环发射频率。
例如 `gemm-rhs 4 2 2 2 64` 表示每两个K迭代，为提前4次迭代的RHS地址发出
两条连续64字节的L2预取。

插件必须在现场使用与 `triton-shared-opt` ABI 匹配的 LLVM/MLIR 重新编译，
不能复制其他机器上预编译的 `.so`。

GEMM 在每一层 IR 中可利用的数据、预取语义，以及继续 materialize 所需的
下层和硬件信息，见
[docs/GEMM_各IR层软件预取信息与依赖.md](docs/GEMM_各IR层软件预取信息与依赖.md)。

## 一次性硬件校准

不再通过逐个试验预取组合来猜硬件参数。先在目标机器上运行一次完整
校准：

```bash
bash hardware_calibration/run_calibration.sh full /path/to/project-output/hardware
```

它会测量 Cache 容量/行大小、依赖访存延迟、单核读带宽、stride 行为和
`issue_every=1/2/4` 的预取发射成本，产生 `HardwareProfile.v1.1.json`。
快速自检：

```bash
bash hardware_calibration/selftest.sh
```

详细的测量边界和字段说明见
[hardware_calibration/README.md](hardware_calibration/README.md)。该文件只包含数值测量和
机器基本信息，不读取 Triton/FlagGems 源码、IR 或测试数据；是否可带离现场
仍以对方的数据规则为准。

## 本地/现场 lowering 对齐

在修改真实 GEMM 地址解析器之前，先用同一探针对公开源码版本、关键环境变量和
各层 IR 的标准操作计数进行对齐。探针不输出 IR 文本、symbol、location、路径、
地址或原始文件散列：

```bash
python3 scripts/onsite_alignment_probe.py \
  /path/to/llvm-project \
  /path/to/triton-cpu \
  /path/to/one-bmm-dump-directory \
  --compact
```

如果一个目录中含有多次编译产生的同名文件，探针会报告
`ambiguous_matches=N`，并按修改时间从新到旧给出不含路径的 `GROUP 0..N` 摘要。
用 `--group-index N` 选择含 BMM/dot/matmul 特征的组后重跑即可。版本对齐后，
按 `tt -> ttshared -> 00 -> 01/02` 顺序比较结构；第一个发生分歧的阶段就是需要修正
的 frontend、转换器或 SME schedule 边界。`count_sig` 仅由标准操作计数向量计算，
不是原始 IR 文件的散列；去掉 `--compact` 才会打印用于诊断首个分歧阶段的完整计数。
如果混入的文件来自本项目的 round-trip/prefetch 输出，探针会优先自动选择同一目录
中同时含原生 `tt.mlir` 和 `ttshared.mlir` 的编译器 dump，忽略项目生成的复制件。

## GEMM 数值特征与 Cost Model

对 `bufferized_before_sme.mlir` 只读提取 K-loop 和 packed RHS 数值特征：

```bash
bash prefetch_plugin/onsite_extract_gemm_profile.sh \
  /path/to/llvm-install \
  /path/to/bufferized_before_sme.mlir \
  /path/to/project-output/model
```

生成 `GemmKernelProfile.v1.json`，其中不包含 IR 文本、操作名、location 或地址。
将它与实测硬件 Profile 交给公式型 Cost Model：

```bash
python3 cost_model/plan_gemm_rhs.py \
  --hardware /path/to/HardwareProfile.v1.1.json \
  --kernel /path/to/GemmKernelProfile.v1.json \
  --anchor-step-ns <K-loop单步时间> \
  --output /path/to/PrefetchPlan.v1.1.json
```

这一版不搜索候选矩阵；距离、覆盖 cache-line 和发射频率都由延迟、流量与成本
约束直接推导。详见 [cost_model/README.md](cost_model/README.md)。

K-loop 的 SME 计算下界由独立 BFMOPA 探针测量：

```bash
bash sme_timing/run_sme_timing.sh /path/to/project-output/sme-timing
```

探针在 AArch64 上原生执行 `SMSTART/RDSVL/BFMOPA/SMSTOP`，用 `CNTVCT_EL0` 分别测量
单 ZA tile 依赖延迟和四 tile 吞吐，并扣除空循环开销。详见
[sme_timing/README.md](sme_timing/README.md)。

现场一次性采集上述三类模型输入：

```bash
bash scripts/onsite_collect_model_inputs.sh \
  /path/to/llvm-install \
  /path/to/bufferized_before_sme.mlir \
  /path/to/onsite-output \
  full
```

脚本会创建带时间戳的新目录，因此不会再因为旧输出目录已存在而失败。

## 原始A地址软件预取候选

确认真实 BMM 是 `TILE_M=TILE_N=TILE_K=4` 且已经取得完整 `00` 后，使用
[prefetch_plugin/SOURCE_A_RUNBOOK.md](prefetch_plugin/SOURCE_A_RUNBOOK.md)。
现有 LLVM 20 Pass 插件新增了严格的原始 A 地址 matcher 与一键脚本，生成：

- 无预取 split roundtrip；
- A-source `distance=8, issue-every=8` 的 L2 候选；
- 同一地址和频率的 L1 候选。

该工具只改写输出目录内的 IR 副本，不修改现场 LLVM、Triton、
Triton-Shared、FlagGems 源码或安装文件。

## 一屏查看现场结论

不阅读长报告，自动选择最新一次追踪结果：

```bash
bash scripts/show_onsite_summary.sh
```

指定某次结果：

```bash
bash scripts/show_onsite_summary.sh \
  --result onsite_results/pytest_trace_<时间>
```

它只在现场终端打印约20行结论，不上传、不打包原始结果。

## 执行并记录测试

确认测试命令和工作目录无误后：

```bash
bash scripts/run_onsite_analysis.sh \
  --dump /absolute/path/to/dumps \
  --case-name bmm_8192_64_2048 \
  --test-workdir /absolute/path/to/FlagGems \
  --test-command "pytest -s 'test/test_blas_ops_mem.py::test_accuracy_bmm_membound[dtype2-8192-64-2048]'" \
  --run-test
```

这里记录的是完整 pytest 墙钟时间，可能包含 JIT（即时编译）、输入生成、
正确性检查和 Cache（缓存）命中。它不自动等价于内核执行时间。

## 依赖

基础分析功能只要求 Linux Shell 和 Python 3.8+。Pass 插件还要求 CMake、
C++17 编译器及现场 LLVM/MLIR CMake package。下列工具存在时会自动使用：

```text
llvm-objdump / objdump
readelf
nm
file
lscpu
perf
```

详细步骤见 [docs/现场实验操作清单.md](docs/现场实验操作清单.md)，数据边界见
[SECURITY_AND_DATA_BOUNDARY.md](SECURITY_AND_DATA_BOUNDARY.md)。
