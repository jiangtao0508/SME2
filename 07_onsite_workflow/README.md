# 07 Linux 现场编译链分析

本阶段用于只能通过 Linux 命令行访问的现场环境。工具不会修改 dump，也不会默认
复制完整源码或 IR；它采集环境、文件清单、IR 层间变化和二进制证据，并生成中文
报告。

## 最简使用

先运行对方测试，再分析已有 dump：

```bash
bash scripts/run_onsite_analysis.sh \
  --dump /absolute/path/to/dumps \
  --case-name bmm_3x8192_64_2048
```

由工具执行并记录测试，然后分析：

```bash
bash scripts/run_onsite_analysis.sh \
  --dump /absolute/path/to/dumps \
  --case-name bmm_8192_64_2048 \
  --test-workdir /absolute/path/to/FlagGems \
  --test-command 'pytest -s test/test_blas_ops_mem.py::test_accuracy_bmm_membound[dtype2-8192-64-2048]' \
  --run-test
```

默认输出到：

```text
onsite_results/<case>-<time>/
```

首先查看：

```text
ONSITE_REPORT_CN.md
04_ir_transitions.md
05_sme_evidence.txt
06_binary/
```

## 工具回答的问题

1. pytest 实际用了哪个 Python、PyTorch、Triton 和编译器？
2. dump 中每个文件可能由哪个编译阶段生成？
3. `linalg/scf/vector/arm_sme/LLVM` 分别在哪一层出现或消失？
4. SME 是在 MLIR、LLVM intrinsic（内建操作）还是机器码阶段出现？
5. `.o/.so` 中是否真的存在 `fmopa/smstart/smstop`？
6. 哪些结论是直接证据，哪些只是根据文件名和内容作出的推断？

## 依赖

基础分析只要求：

```text
Linux shell
Python 3.8+
```

克隆后先运行自检：

```bash
bash scripts/run_onsite_selftest.sh
```

以下工具存在时会自动使用，不存在不会阻断基础报告：

```text
llvm-objdump / objdump
readelf
nm
file
lscpu
perf
```

## 输出边界

- `artifact_inventory` 记录相对路径、大小和 SHA256，不复制原文件。
- `generic_analysis_summary` 是启发式识别结果，不等价于编译器 Pass 的严格语义证明。
- 默认导出的通用分析摘要不包含IR原文证据片段，只保留统计、结论和相对路径。
- 文件角色中的 `high/medium/low confidence` 分别表示高、中、低置信度。
- 性能数据来自被记录的测试命令；本工具不会从 IR 静态计数伪造运行时间。
