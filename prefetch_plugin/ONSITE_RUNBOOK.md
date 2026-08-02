# 现场 Pass 插件验证清单

目标：只在克隆下来的项目目录内构建动态插件，运行时加载到现场已有的
`triton-shared-opt`。不修改、不覆盖、不安装任何 LLVM、Triton 或
Triton-Shared 文件。

## 1. 克隆并记录环境

```bash
git clone <PROJECT_GIT_URL> prefetch-model
cd prefetch-model

TRITON_SHARED_OPT=/absolute/path/to/triton-shared-opt
LLVM_INSTALL_DIR=/absolute/path/to/the/matching/llvm-install

"$TRITON_SHARED_OPT" --version
"$TRITON_SHARED_OPT" --help | grep -E 'load-pass-plugin|transform-interpreter'
"$LLVM_INSTALL_DIR/bin/mlir-translate" --version
git rev-parse HEAD
```

这里的 LLVM 安装必须与现场 `triton-shared-opt` 使用的 LLVM/MLIR 版本和
ABI 匹配。不能拿本机编好的 `.so` 直接复制过去。

## 2. 一键构建和冒烟测试

```bash
bash prefetch_plugin/build_and_smoke.sh \
  "$LLVM_INSTALL_DIR" \
  "$TRITON_SHARED_OPT"
```

成功标志：

```text
PASS: plugin loaded directly and through transform.apply_registered_pass
PASS: memref.prefetch lowered to llvm.prefetch and AArch64 prfm
```

Linux 插件通常是 `prefetch_plugin/build/PrefetchPassPlugin.so`；macOS 是
`.dylib`。

## 3. 对现场 lowering 做只读基线记录

先使用对方原始命令生成一份未修改的完整 dump。把需要实验的
`00_input.mlir`、后续各层 IR 和 LLIR **复制**到项目目录。所有改写只对副本
进行，并记录：

```bash
sha256sum 00_input.mlir
git -C /absolute/path/to/triton status --short
git -C /absolute/path/to/llvm-project status --short
```

如果现场仓库本来就有未提交内容，只记录，不清理、不覆盖。

## 4. 第一阶段一键实验

取得现场生成的 `00_input.mlir` 后运行：

```bash
bash prefetch_plugin/onsite_stage1.sh \
  "$LLVM_INSTALL_DIR" \
  "$TRITON_SHARED_OPT" \
  /absolute/path/to/00_input.mlir \
  /absolute/path/to/project-output
```

脚本依次完成：

1. 构建插件并运行最小冒烟测试；
2. 用 `prefetch-snapshot` 导出 `bufferized_before_sme.mlir`；
3. 尝试公开 GEMM 的 `prefetch-gemm-rhs` 结构匹配器；
4. 检查 SME lowering 后是否同时存在 `llvm.intr.prefetch` 和
   `arm_sme.intr.mopa`。

如果第 2 步成功、第 3 步不匹配，插件和插入点仍然已经验证成功。此时保存
`bufferized_before_sme.mlir`，按现场真实地址链调整 resolver。

如果第一阶段同时找到了 prefetch 和 MOPA，继续生成 LLIR 和对象：

```bash
bash prefetch_plugin/onsite_stage2.sh \
  "$LLVM_INSTALL_DIR" \
  "$TRITON_SHARED_OPT" \
  /absolute/path/to/triton-shared/backend/compiler.py \
  /absolute/path/to/project-output/01_gemm_rhs_prefetch.mlir \
  /absolute/path/to/project-output/final
```

成功标志类似：

```text
PASS: object contains PRFM=4 and FMOPA=16
LLIR override candidate: .../kernel.llir
```

最后把生成的 `kernel.llir` 按现场 dump 中的真实 kernel 名称放入对应的
override hash 目录，并设置 `TRITON_ALWAYS_COMPILE=1` 验证 override 日志。
正确性通过后，按照 `prefetch_plugin/EXPERIMENT_MATRIX.md` 运行
`distance=1/2/4/8` 的第一轮扫描。

## 5. 插入点

使用脚本生成副本；脚本会在 `@__transform_main` 中，把调用插在
`@__bufferize_schedule` 与 `@__arm_sme_lowering_schedule` 之间，并拒绝
覆盖输入文件。插件目标是此时已经 bufferize 的 `func.func`：

```bash
python3 prefetch_plugin/inject_schedule.py \
  00_input.mlir 00_input.prefetch.mlir \
  --mode gemm-rhs --distance 4 --locality 3
```

生成的核心 Transform IR 是：

```mlir
%prefetch_funcs = transform.structured.match ops{["func.func"]} in %bufferized
  : (!transform.any_op) -> !transform.op<"func.func">
%prefetched = transform.apply_registered_pass "prefetch-gemm-rhs" to %prefetch_funcs
  {options = "distance=4 locality=3"}
  : (!transform.op<"func.func">) -> !transform.op<"func.func">
```

随后用插件运行现场原有 Transform Interpreter 命令：

```bash
"$TRITON_SHARED_OPT" \
  --load-pass-plugin=/absolute/path/to/PrefetchPassPlugin.so \
  00_input.prefetch.mlir \
  --mlir-disable-threading \
  --transform-interpreter \
  -o 01_after_transform_interpreter.prefetch.mlir
```

因为后续 Transform 已执行，先确认 `01` 中存在 `llvm.intr.prefetch`，再继续
执行现场原有的 erase schedule、LLVM 后处理、`mlir-translate` 和 LLIR
override 流程。

## 6. 明天第一轮的停止条件

- `--load-pass-plugin` 不存在：停止插件路线，保留输出，改走独立 IR rewrite。
- 动态库加载时报 undefined symbol/ABI 错误：停止，不替换平台库；改用现场准确
  的 LLVM/MLIR 安装重新编译。
- 冒烟测试通过但真实 BMM 匹配失败：这是 resolver 规则不够，不是插件机制
  失败。保存 bufferize 后的 IR，用它扩展 B-panel 匹配规则。
- `memref.prefetch` 在后续 pipeline 消失：保存消失前后的相邻两层 IR，再决定
  移动插入点或阻止错误的 canonicalization。

当前插件已经在公开 GEMM 上完成 RHS panel 到 AArch64 `PRFM` 的全链路验证，
但 matcher 仍是面向公开 IR 结构的实验版本，不是最终通用 BMM resolver。现场
首要目标仍是取得真实 bufferized payload，再判断 matcher 能否直接复用。
