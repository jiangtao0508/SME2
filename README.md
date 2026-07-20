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

基础功能只要求 Linux Shell 和 Python 3.8+。下列工具存在时会自动使用：

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
