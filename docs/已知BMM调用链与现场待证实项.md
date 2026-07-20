# 已知BMM调用链与现场待证实项

## 1. 为什么必须分成两部分

本机公开FlagGems与对方环境存在明显差异：

- 本机公开仓库没有`test_blas_ops_mem.py`和
  `test_accuracy_bmm_membound`；
- 对方使用指定编译器和Triton-CPU相关缓存产物；
- 对方dump包含定制MLIR阶段和Arm SME操作；
- 公开FlagGems会持续更新，现场版本可能来自分支、补丁或内部集成。

因此下面只把公开代码用于理解“上层机制”，现场真实流程必须由追踪报告证明。

## 2. 公开FlagGems中已经确认的上层流程

本机阅读版本：

```text
FlagGems commit: 7bdbf1204
flag_gems version: 5.4.0dev
```

### 第一步：pytest收集参数化测试

公开`tests/test_bmm.py`使用`pytest.mark.parametrize`把形状和数据类型展开为多个
独立Node ID。pytest负责：

```text
导入测试模块
加载conftest.py和插件
展开参数组合
执行setup/call/teardown
收集成功、失败和跳过结果
```

pytest本身不实现BMM，也不生成MLIR；它只是测试组织和执行框架。

### 第二步：建立参考结果

公开测试创建三维输入张量，并在FlagGems作用域外调用：

```text
torch.bmm(ref_mat1, ref_mat2)
```

这一调用用于生成参考结果。参考设备由测试配置决定，不能仅凭函数名假定一定是
CPU。

### 第三步：进入FlagGems分发作用域

测试随后进入：

```text
with flag_gems.use_gems():
    torch.bmm(mat1, mat2)
```

`use_gems.__enter__`调用`enable`或`only_enable`，创建
`torch.library.Library("aten", "IMPL")`并注册算子实现。

公开`_FULL_CONFIG`包含：

```text
("bmm", bmm)
("bmm.out", bmm_out)
```

因此作用域内的`torch.bmm`可以通过PyTorch Dispatcher（分发器）进入FlagGems
实现。离开作用域时注册句柄被销毁。

### 第四步：选择后端BMM实现

FlagGems导入时由`SpecOpRegistrar`根据vendor/backend选择特定实现。公开Arm后端
存在：

```text
src/flag_gems/runtime/backend/_arm/ops/bmm.py
```

现场到底加载通用`ops/bmm.py`、Arm版本还是对方自定义版本，必须查看：

```text
flag_gems.vendor_name
flag_gems.backend_info
bmm函数的__module__
bmm函数源码文件
运行时注册日志
```

### 第五步：调用Triton JIT内核

公开Arm BMM实现：

```text
读取batch、M、N、K
把输入变为连续布局
分配输出
构造三维grid
调用bmm_kernel[grid_fn](...)
```

`bmm_kernel`带有：

```text
@triton.jit
@triton.autotune
@triton.heuristics
```

第一次遇到新的形状、数据类型和配置时，Triton通常执行JIT（即时编译）；缓存
命中时可能直接复用已有二进制。

### 第六步：Triton内核语义

公开内核的核心结构是：

```text
计算batch和二维tile编号
构造A/B/C指针
按K维循环
tl.load读取A和B
tl.dot执行块矩阵乘加
tl.store写回输出
```

`tl.dot`如何变成CPU/SVE/SME指令由后端编译器决定，不由pytest决定。

## 3. 对方环境中根据文件名形成的暂定链路

下面是“高可能性解释”，还不是全部事实：

| 文件 | 暂定含义 | 当前把握 |
|---|---|---|
| `bmm_kernel.ttir` | Triton高层IR | 中 |
| `bmm_kernel.ttsharedir` | Triton共享布局/CPU后端中间IR | 中低 |
| `bmm_kernel.llir` | LLVM IR | 中高 |
| `bmm_kernel.obj` | 内核目标文件 | 高 |
| `bmm_kernel.json` | JIT缓存元数据 | 中 |
| `_grp_bmm_kernel.json` | 分组/配置缓存元数据 | 中低 |
| `_cpu_kernel_launcher.so` | Python可加载的CPU启动共享库，可能也静态包含内核 | 中 |
| `_triton_shared.ref` | 共享库或缓存引用记录 | 低 |
| `00_input.mlir` | 定制MLIR流水线入口 | 高 |
| `01_after_transform_interpreter.mlir` | 执行Transform Dialect后 | 高 |
| `02_after_earase_schedule.mlir` | 删除调度描述后，文件名含`earase`拼写 | 中高 |
| `03_after_convert_to_llvm.mlir` | 转到LLVM方言边界后 | 高 |
| `04_after_canonicalize.mlir` | 规范化后 | 高 |
| `05_after_strip_debug.mlir` | 移除调试信息后 | 高 |
| `06_after_legalize_float8.mlir` | Float8合法化后 | 高 |
| `tt.mlir` | TTIR的MLIR文本镜像或转换输入 | 中低 |
| `ttshared.mlir` | 共享布局IR的MLIR文本镜像 | 中低 |
| `ll.mlir` | LLVM方言MLIR | 中 |
| `ll.ir` | LLVM文本IR | 高 |
| `kernel.o` | dump目录中的目标文件 | 高 |
| `main.cxx` | 启动器或独立复现驱动源码 | 中高 |

“高”主要表示文件名语义清楚，不表示已经知道其生产函数或准确Pass参数。

## 4. 目前最关键的未知量

1. `test_accuracy_bmm_membound`由谁维护，和公开`tests/test_bmm.py`有什么关系？
2. `dtype2`实际对应哪种PyTorch数据类型？
3. 参数`8192-64-2048`在函数参数中对应`M-K-N`还是其他顺序？
4. 测试是否在`flag_gems.use_gems()`中调用`torch.bmm`？
5. `torch.bmm`最终注册到哪个Python函数和哪个源码文件？
6. 哪个组件开启`dumps`，对应环境变量、pytest插件还是编译器参数？
7. 编号MLIR是一个进程内部逐Pass写出，还是多条外部命令生成？
8. cache和dumps中的对象文件是否内容相同？
9. `_cpu_kernel_launcher.so`为什么含SME指令：静态链接内核，还是直接承载计算？
10. MLIR中的ArmSME操作来自哪一个Pass，Pass源码和参数是什么？

## 5. 现场如何把未知量变成事实

运行：

```bash
PYTHON_BIN=/path/to/their/python \
bash scripts/run_full_pytest_trace.sh \
  --workdir /path/to/FlagGems \
  --nodeid 'test/test_blas_ops_mem.py::test_accuracy_bmm_membound[dtype2-8192-64-2048]' \
  --watch-root /path/to/dumps \
  --dump /path/to/dumps \
  --scan-root /path/to/FlagGems \
  --scan-root /path/to/compiler/package-or-source \
  --strace auto \
  --run
```

然后按证据回答：

| 问题 | 主要证据 |
|---|---|
| pytest执行了谁 | `01_pytest_collect.*` |
| 测试源码调用了谁 | `02_test_ast.json` |
| FlagGems实际注册了谁 | `02b_runtime_dispatch.json` |
| 运行时启动了谁 | `04_process_timeline.json`、`06_strace execve` |
| 文件何时出现 | `05_file_timeline.*` |
| 谁写了文件 | `07_producer_map.*` |
| 文件名在哪构造 | `08_source_name_matches.json` |
| IR每层改变了什么 | `artifact_analysis/03_ir_chain_analysis.json` |
| 最终是否有SME | `artifact_analysis/05_sme_evidence.txt`和反汇编 |

## 6. 最终应形成的准确表述

完成现场追踪后，流程应能写成：

```text
pytest版本与插件
-> 精确Node ID和参数值
-> 测试函数文件/行号/SHA256
-> torch.bmm调用及FlagGems注册实现
-> Triton kernel Python函数
-> JIT编译进程与完整命令
-> TTIR/TTSharedIR
-> 定制MLIR Pass流水线
-> LLVM方言/LLVM IR
-> 目标文件与launcher共享库
-> C/Python启动执行
-> 正确性、时间和SME静态/动态证据
```

每个箭头旁边都应附一个文件、日志、PID、命令或源码位置，不能只写“应该是”。
