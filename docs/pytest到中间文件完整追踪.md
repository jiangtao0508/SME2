# pytest到中间文件的完整追踪

## 1. 要回答的核心问题

本工具不只回答“dump里有什么”，还要回答：

1. pytest到底收集并执行了哪个参数化测试实例？
2. 哪些pytest插件和`conftest.py`参与了测试？
3. 测试源码是否调用`torch.bmm`、`flag_gems.use_gems`或其他入口？
4. 运行时启动了哪些Python、编译器、链接器和启动器进程？
5. 每个MLIR、LLVM IR、目标文件和共享库在什么时间出现？
6. 哪个PID（进程号）实际打开或重命名了这些输出文件？
7. `00_input.mlir`等命名逻辑位于哪份源码、哪一行？
8. 哪些结论是直接证据，哪些只是根据文件名作出的推断？

## 2. 先确认环境

在对方要求的环境中：

```bash
cd /path/to/FlagGems
which python
python -m pytest --version
command -v strace || true
```

如果pytest来自指定Python：

```bash
export PYTHON_BIN=/path/to/their/python
"$PYTHON_BIN" -m pytest --version
```

`strace`是Linux系统调用追踪器。它存在且允许使用时，工具可以看到
`execve`（启动程序）、`openat`（打开/创建文件）和`rename`（重命名文件）。
如果系统禁止`ptrace`，工具会退化为进程轮询和文件时间线。

## 3. 只收集，不执行

先由pytest确认Node ID（用例唯一标识）：

```bash
"$PYTHON_BIN" -m pytest --collect-only -q \
  'test/test_blas_ops_mem.py::test_accuracy_bmm_membound[dtype2-8192-64-2048]'
```

再查看插件：

```bash
"$PYTHON_BIN" -m pytest --trace-config --collect-only -q \
  'test/test_blas_ops_mem.py::test_accuracy_bmm_membound[dtype2-8192-64-2048]'
```

这两步能确认测试是否存在、参数顺序是否正确，以及哪些插件参与收集；不会进入
测试函数主体。

一键工具也会先做同样的收集。若收集失败，它会停止，不会继续执行一个路径或参数
可能错误的测试。

## 4. 一键执行完整追踪

假设测试运行后生成`/path/to/dumps`：

```bash
cd /path/to/SME2

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

如果`dumps`目录在执行前不存在，`--watch-root`仍可以指向它；工具会持续检查。
如果真实输出位置不确定，可以先监控其上一级专用目录，但不要监控整个根文件系统。

测试还需要额外pytest参数时重复添加：

```bash
--pytest-arg=--quick
--pytest-arg=--某个参数
```

测试依赖额外环境变量时：

```bash
--env NAME=value
```

不要把密码、令牌或私钥通过`--env`写入命令历史。

## 5. 结果阅读顺序

默认输出：

```text
onsite_results/pytest_trace_<时间>/
```

按下面顺序查看：

```bash
cd onsite_results/pytest_trace_<时间>

less PYTEST_PIPELINE_REPORT_CN.md
less 01_pytest_collect.log
less 03_pytest_run.log
column -s, -t 05_file_timeline.csv | less -S
column -s, -t 07_producer_map.csv | less -S
less artifact_analysis/ONSITE_REPORT_CN.md
less artifact_analysis/04_ir_transitions.md
less artifact_analysis/05_sme_evidence.txt
```

## 6. 每类证据说明

### pytest收集证据

`01_pytest_collect.*`说明：

- 测试函数和参数实例是否真的被收集；
- `conftest.py`和第三方插件是否加载；
- 收集阶段是否已经报错或跳过。

### AST静态分析

AST是Abstract Syntax Tree（抽象语法树）。`02_test_ast.json`只保存：

- 测试函数位置和SHA256；
- 装饰器和参数化结构；
- 函数内部调用名称；
- `with`上下文管理器名称。

它不复制源码。AST看到`torch.bmm`只证明源码写了这个调用，不证明运行时一定走到
FlagGems或SME。

### FlagGems运行时分发

工具通过只记录元数据的pytest探针生成`02b_runtime_dispatch.json`，内容包括：

- `flag_gems.vendor_name`、设备和版本；
- 实际`flag_gems.bmm`函数的模块、限定名和源码文件；
- 进入`flag_gems.use_gems()`前后，PyTorch的`aten::bmm`分发表；
- 测试开始和结束的Node ID。

探针不读取或保存张量内容。若对方测试不用`flag_gems.use_gems()`或使用了改名入口，
报告会明确显示没有捕获到`after_use_gems_enter`事件，此时要根据测试AST调整探针。

### 进程证据

`04_process_timeline.json`来自Linux`/proc`，记录：

- PID和父PID；
- 可执行文件路径；
- 命令行；
- 首次和最后一次被观察的时间。

轮询可能漏掉寿命特别短的进程，所以有`strace`时优先参考其`execve`记录。

### 文件时间线

`05_file_timeline.*`说明文件何时创建、修改或删除。它可以重建大致顺序，但时间
重叠不能单独证明生产者。

### strace生产者映射

`07_producer_map.*`把`openat/rename`写入事件关联到PID和最近的`execve`。这是
“哪个进程写了哪个文件”的最直接现场证据。

一次`openat`仍不等价于该进程完成全部内容；必要时继续看原始
`06_strace/trace.<pid>`。

### dump命名源码位置

`08_source_name_matches.json`搜索：

```text
after_transform_interpreter
earase_schedule / erase_schedule
convert_to_llvm
canonicalize
strip_debug
legalize_float8
ttsharedir
_cpu_kernel_launcher
```

只保存源码路径、行号和命中关键词。若命中编译器源码，就能继续向上找到实际
dump函数和Pass流水线。

## 7. 如何确认每一步究竟由什么生成

对每个中间文件使用三角验证：

1. 文件时间线确认生成顺序。
2. 运行时分发探针确认`torch.bmm`进入哪个FlagGems后端函数。
3. `strace`确认写文件的PID和进程命令。
4. 源码命名扫描确认文件名在哪个函数中构造。

然后进入命中源码附近，确认：

```text
dump调用之前执行了哪个Pass
dump调用之后准备执行哪个Pass
PassManager中真实注册的Pass名称
对应命令行参数或环境变量
是否由一个编译器进程内部连续dump，而不是七条外部命令
```

很可能出现这种情况：

```text
pytest只启动一次Python/JIT编译
编译器进程内部运行多个MLIR Pass
每个Pass后调用内部dump函数
所以外部看不到七条mlir-opt命令
```

此时“每层由什么生成”必须依靠编译器Pass日志或源码，不能仅靠`ps`。

## 8. 追踪与性能必须分开

本次完整追踪包含：

- pytest插件打印；
- `/proc`轮询；
- 文件系统扫描；
- 可选`strace`；
- 首次JIT和dump写盘。

因此追踪时间不是算子性能。流程确认后，再原样执行一次无追踪测试作为性能基线：

```bash
cd /path/to/FlagGems
"$PYTHON_BIN" -m pytest -s \
  'test/test_blas_ops_mem.py::test_accuracy_bmm_membound[dtype2-8192-64-2048]'
```

还应把首次JIT、缓存命中和内核计时分开记录。

## 9. 现场数据边界

原始`strace`、测试日志和反汇编可能泄露安装路径、命令和符号。是否可以带出必须
遵守对方规定。不能确认时，只在现场查看结果，不复制整个输出目录。
