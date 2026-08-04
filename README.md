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
