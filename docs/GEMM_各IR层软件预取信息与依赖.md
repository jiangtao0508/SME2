# GEMM 各 IR 层的软件预取信息、语义与依赖

## 1. 核心结论

软件预取决策不能由单一 IR 层完整决定。它至少依赖四类信息：

```text
S：Semantic，计算和对象语义
Q：Schedule，循环、tile、向量化和执行顺序
A：Address，实际被消费对象的地址表达式
H：Hardware，Cache、延迟、带宽和机器执行能力
```

完整决策可以写成：

```text
PrefetchPlan = F(S, Q, A, H)
```

其中：

- 高层 IR 最擅长回答“预取什么”；
- 中层 IR 最擅长回答“在哪个循环、以什么地址预取”；
- 低层 IR 最擅长验证“最终变成了什么指令”；
- 硬件测量回答“是否真的有收益”。

因此，不能因为高层知道 `B_PANEL` 就直接假设它知道最终地址，也不能因为
LLVM IR 有一个 `ptr` 就假设它知道该指针是 GEMM 的 B。

## 2. GEMM 中可预取的数据对象

以：

```text
C[M,N] += A[M,K] * B[K,N]
```

为例，逻辑上至少有三个对象：

| 对象 | 典型访问 | 复用来源 | 第一版预取价值 |
|---|---|---|---|
| A panel | `A[m_tile, k]` | 同一 A 值参与多个 N lane/tile | 候选 |
| B/RHS panel | `B[k, n_tile]` | 同一 B 值参与多个 M lane/tile | 首选候选 |
| C tile | 循环内反复累加，最后写回 | 累加器/ZA 内部复用 | 通常不预取写流 |

但是，逻辑对象和物理对象并不是一一对应。公开 GEMM 的 B 实际经历：

```text
逻辑 B
  -> 函数参数 %arg1
  -> memref.reinterpret_cast
  -> memref.copy 的目标 %alloc_2
  -> transpose/packing 的目标 %alloc_3
  -> K-loop 中的 memref.subview
  -> vector.transfer_read
  -> SME FMOPA
```

所以至少存在两类不同的 B 预取：

### 2.1 输入/打包预取

```text
目标：外部 B 输入
位置：copy、transpose 或 pack 循环之前
目的：隐藏主存读取和打包阶段的延迟
```

需要知道原始 B 的布局、copy/pack 循环、是否已经由硬件预取器覆盖。

### 2.2 计算消费预取

```text
目标：packed/transposed B buffer
位置：FMOPA 对应 K-loop 内
目的：在计算消费前把 packed panel 带入目标 Cache
```

需要知道 packed buffer 的物理布局、K-loop、每次迭代消费的 cache line 数量和
计算窗口。本机当前 `prefetch-gemm-rhs` 实现的是第二类。

这两个策略不能共用一个模糊的 `object_id = B`。建议区分：

```text
B_LOGICAL
B_INPUT
B_PACK_SOURCE
B_PACKED
B_COMPUTE_VIEW
```

### 2.3 BMM 的 batch 维

BMM 还必须区分 batch 复用与 batch 独立：

```text
C[b,m,n] += A[b,m,k] * B[b,k,n]
```

需要从高层确认 B 是否 broadcast across batch，从 MemRef 层确认 batch stride，
从硬件层确认一个 batch 的 panel 是否可能跨 batch 留在 Cache。如果 B 在 batch
间广播，同一 B panel 可能具有更高复用；如果每个 batch 使用独立 B，则把预取
距离跨越 batch 边界通常是错误的。第一版 legality 应限制 future address 留在
当前 batch，除非计划明确声明 batch-level prefetch。

## 3. 各层能力总览

| 层级 | 代表文件 | 最强信息 | 能决定什么 | 主要缺失 |
|---|---|---|---|---|
| Python/Kernel | FlagGems/Triton 源码 | 算子意图、shape、dtype | 候选 kernel、逻辑对象 | 最终 schedule、地址、硬件代价 |
| TTIR | `tt.mlir` | pointer tensor、mask、dot | 逻辑访问方向和边界 | CPU 物理循环、packed buffer |
| TTShared/Linalg | `ttshared.mlir` | matmul operand、transpose、shape/stride | A/B/C 角色、reuse、流量 | 最终 tile、K-loop、地址 |
| Schedule 输入 | `00_input.mlir` | 高层 payload + Transform schedule | 选择正确插入时机 | bufferize 后的实际结果 |
| MemRef/SCF/Vector | 动态 snapshot | 循环、subview、stride、transfer | 实际生成 `memref.prefetch` | 完整逻辑角色、真实机器延迟 |
| SME/LLVM Dialect | `01/02/ll.mlir` | 指针计算、分支、SME intrinsic | 指令级放置和 LLVM fallback | A/B 高层身份、稳定结构 |
| LLVM IR | `ll.ir`/`.llir` | 最终 SSA pointer、LLVM intrinsic | LLIR override、最终地址修补 | GEMM 角色、跨版本稳定性 |
| Object/Assembly | `kernel.o` | `PRFM`、`FMOPA`、真实布局 | 验证指令存在及位置 | 无法再做可靠语义决策 |
| Hardware/Runtime | 计数器与计时 | latency、miss、带宽、MLP | 判断收益、校准 Cost Model | 不提供程序语义 |

## 4. Python/FlagGems/Triton Kernel 层

### 4.1 可以利用的数据

- 算子是 MM、BMM 还是 addmm；
- M/N/K、batch、dtype；
- transpose 标志；
- program/block shape；
- mask、边界和广播关系；
- autotune 候选配置。

### 4.2 可以利用的预取语义

- 识别逻辑 A、B、C；
- 识别 reduction 维 K；
- 判断 B 是否转置；
- 按问题规模估算逻辑工作集和数据流量；
- 过滤明显不值得预取的小 kernel。

### 4.3 这一层可以做出的决策

```text
kernel 是否进入预取分析
优先考虑 A 还是 B
是否属于输入打包预取或计算消费预取
生成初始 KernelSemantic
```

### 4.4 仍需要的下层或硬件信息

- Triton CPU 最终选择的 tile 和循环顺序；
- 是否产生临时 copy/transpose/packed buffer；
- 函数参数与实际 MemRef 的绑定；
- 指针对齐；
- Cache line、内存延迟、带宽和硬件预取器能力；
- SME streaming vector length 和吞吐。

### 4.5 是否适合直接插入预取

不适合作为当前工程的直接插入层。此处可以生成意图，但无法保证一个高层 B
地址最终对应哪一个实际 buffer，也无法可靠选择 CPU K-loop 中的放置点。

## 5. TTIR 层

代表文件：`tt.mlir`。

### 5.1 可以利用的数据

```mlir
tt.make_range
tt.splat
tt.addptr
tt.load
tt.trans
tt.dot
tt.store
```

这一层可以看到：

- pointer tensor 的基地址来自哪个函数参数；
- 每个 lane 的 offset 表达式；
- load mask 和边界；
- Triton cache/evict hint；
- `tt.dot` 的两个输入及 transpose。

### 5.2 可以利用的预取语义

- 从 `tt.dot` 识别 A/B/C；
- 从 `tt.addptr` 推导逻辑地址随 M/N/K 的变化方向；
- 从 `tt.load` 的 mask 推导尾块 legality；
- 判断访问是连续、跨步还是 gather-like；
- 区分 B 的原始布局与逻辑转置结果。

### 5.3 这一层可以做出的决策

```text
object_role = A_PANEL / B_PANEL
logical_access = A[m,k] / B[k,n]
boundary_policy = masked
初步 stride/contiguity
```

### 5.4 仍需要的下层或硬件信息

- `tt.dot` 如何被 CPU Transform 分块；
- K-loop 是否显式存在；
- load 是否变成 copy、transpose、subview 或 packed buffer；
- 每个迭代实际消费多少字节；
- 指令级计算窗口；
- 预取指令支持的 locality 和 cache 行为。

### 5.5 是否适合直接插入预取

理论上可以定义 Triton prefetch op，但当前公开 CPU pipeline 没有现成、稳定的
高层 prefetch lowering 合同。用无结果的普通 load 模拟预取还可能被优化删除，
也可能错误引入 fault/带宽。因此 TTIR 适合提取语义，不作为第一版插入点。

## 6. TTShared/Linalg 层

代表文件：`ttshared.mlir`。

### 6.1 可以利用的数据

- `linalg.matmul`/`linalg.batch_matmul`；
- operand 0/1/output 的明确角色；
- tensor/memref shape 和 dtype；
- `linalg.transpose`；
- `memref.reinterpret_cast` 的 shape、offset、stride；
- fill、epilogue add、materialize destination。

### 6.2 可以利用的预取语义

对于：

```mlir
linalg.matmul ins(%A, %B) outs(%C)
```

可以稳定得到：

```text
A -> (m,k)
B -> (k,n)
C -> (m,n)
k -> reduction dimension
```

还可以计算：

```text
A_panel_bytes
B_panel_bytes
C_tile_bytes
逻辑 FLOP = 2*M*N*K
逻辑 working set
transpose/layout penalty
角色驱动的 reuse proxy
```

### 6.3 这一层可以做出的决策

- 生成稳定的 `KernelSemantic`；
- 选择逻辑预取对象；
- 判断输入打包预取与计算预取是否都是候选；
- 产生距离候选集合，而不是最终固定距离；
- 给物理绑定提供 provenance 起点。

### 6.4 仍需要的下层或硬件信息

- logical tile 如何变成 physical tile；
- 循环顺序、步长、peeling 和 tail；
- 是否分配 packed A/B；
- packed buffer 的 shape/stride/alignment；
- `vector.transfer_read` 的宽度；
- 每个 K 迭代的 cycles；
- Cache 容量、line size、latency、MSHR/MLP。

### 6.5 是否适合直接插入预取

适合产生抽象 `PrefetchIntent`，但不适合直接生成当前第一版
`memref.prefetch`。如果在此处插入，需要新增高层 op 并保证它穿过 tile、
bufferize、transpose 和 vector lowering，工程风险较高。

## 7. `00_input.mlir`：Payload 与 Transform Schedule 层

### 7.1 可以利用的数据

该文件同时包含：

```text
高层 Linalg payload
完整 Transform Schedule
```

Schedule 暴露：

- tile 和 vectorize 的顺序；
- bufferization 的位置；
- `convert-linalg-to-loops`；
- `convert-bufferization-to-memref`；
- ArmSME lowering；
- Vector 和 LLVM lowering。

### 7.2 可以利用的预取语义

这一层最重要的不是新数据语义，而是“阶段语义”：

```text
在 bufferize 前：角色清楚，地址不足
在 bufferize 后：地址和循环出现
在 ArmSME 前：仍可使用 memref.prefetch
```

### 7.3 这一层可以做出的决策

- 确定 Pass 的执行窗口；
- 通过 `transform.apply_registered_pass` 加载项目插件；
- 插入 snapshot Pass；
- 保证原有 ArmSME/LLVM pipeline 继续运行。

### 7.4 仍需要的下层或硬件信息

- Transform 执行后实际生成的 MemRef/SCF 结构；
- 现场版本是否有不同的 schedule 名称或顺序；
- 动态插件 ABI 是否匹配；
- 后续 canonicalization 是否保留 prefetch。

### 7.5 是否适合直接插入预取

它是当前工程最合适的控制入口，但不是直接的地址分析层。正确做法是修改
`00_input.mlir` 的副本，在此安排 Pass 于 bufferize 与 ArmSME 之间执行。

## 8. MemRef/SCF/Vector 层

代表文件：运行时导出的 `bufferized_before_sme.mlir`。

### 8.1 可以利用的数据

- 真实 `scf.for` 的 lower/upper/step；
- 主循环、peeled loop 和 tail 分支；
- `memref.alloc`、`copy`、`reinterpret_cast`；
- packed/transposed buffer；
- `memref.subview` 的 offset、size、stride；
- `vector.transfer_read` 的实际消费宽度；
- A/B panel 与 accumulator 的物理访问位置。

公开 GEMM 中实际可见：

```mlir
scf.for %k = %c0 to %c64 step %c1 {
  %b_view = memref.subview %B_packed[%k, %n] ...
  %b_vec = vector.transfer_read %b_view[0, 0] ...
  ... outerproduct ...
}
```

### 8.2 可以利用的预取语义

- `scf.for` induction variable 给出物理 K anchor；
- subview offset 中出现 K-IV，说明该 view 随 reduction 前进；
- transfer width 给出本次迭代消费的最小字节数；
- alloc/copy/transpose 链给出逻辑对象到物理对象的 provenance；
- dominance 和 region 结构给出合法插入位置；
- upper bound 给出 `k + D < K` 的保护条件。

### 8.3 这一层可以做出的决策

这是第一版 Rewrite 可以完整决定的部分：

```text
physical_object = B_PACKED
anchor_loop = physical K-loop
placement = before current B transfer_read
future_index = k + distance * step
address = B_PACKED[future_index, n_offset]
guard = future_index < upper_bound
op = memref.prefetch
```

还可以根据 transfer width 与 cache line size 决定一轮预取几个地址。

### 8.4 仍需要的高层、下层或硬件信息

高层需要提供：

- 该物理 buffer 到底是 A 还是 B；
- 它是输入、packed 还是临时输出；
- 是否允许预取这个对象。

硬件需要提供：

- cache line size；
- L1/L2 容量和目标 locality 的含义；
- latency、带宽、MSHR/MLP；
- 硬件预取器是否已覆盖顺序流；
- SME SVL 和每次 transfer 的真实字节数。

低层需要验证：

- `memref.prefetch` 是否保留到 LLVM；
- 最终是否选择为 `PRFM`；
- 指令是否仍位于目标循环中。

### 8.5 是否适合直接插入预取

是。它同时具有循环和可 lower 的地址，是当前正式主路径。

## 9. SME/LLVM Dialect 层

代表文件：`01_after_transform_interpreter.mlir`、
`02_after_erase_schedule.mlir` 和 `ll.mlir`。

### 9.1 可以利用的数据

- `llvm.func` ABI；
- `llvm.getelementptr`、load/store、branch；
- 已展开的主循环和边界路径；
- `arm_sme.intr.mopa`、`ld1w`、`st1w`；
- `arm_sve.intr.whilelt` 和 predicate；
- 如果高层插入成功，可见 `llvm.intr.prefetch`。

### 9.2 可以利用的预取语义

- 从 MOPA 前的 load/GEP 识别真实消费地址；
- 计算预取与 MOPA 之间的静态指令距离；
- 检查预取是否被复制到主循环、remainder 和 tail；
- 检查 guard 是否正确 lower；
- 估算每个 K 迭代的 SME 指令数量。

### 9.3 这一层可以做出的决策

- 验证上一层的 materialization；
- 当 MemRef 插入不可用时，直接插 `llvm.intr.prefetch`；
- 对固定 kernel 手工扫描距离；
- 生成 LLIR override 候选。

### 9.4 仍需要的高层或硬件信息

- 哪条 GEP 属于 A/B/C；
- pointer 是否来自原始输入还是 packed buffer；
- alias 和 lifetime；
- LLVM 优化后真实调度；
- 目标 CPU 对 locality hint 的指令选择；
- 实际 latency、Cache miss 和带宽。

### 9.5 是否适合直接插入预取

可以作为 fallback，但不适合通用自动选择对象。低层结构容易随 LLVM 版本和
优化改变，必须依赖外部 BindingManifest 或非常保守的结构匹配。

## 10. LLVM IR 层

代表文件：`ll.ir` 或 Triton override 使用的 `.llir`。

### 10.1 可以利用的数据

- 标准 LLVM SSA 和 pointer；
- `getelementptr`、`phi`、`icmp`、branch；
- `llvm.prefetch`；
- AArch64 SME intrinsic；
- 最终函数名、参数 ABI 和 attributes。

### 10.2 可以利用的预取语义

- 基于 loop `phi` 构造未来迭代地址；
- 确认 prefetch 参数：read/write、locality、data/instruction cache；
- 检查 LLVM verifier、dominance 和类型合法性；
- 用同一份 LLIR 做可控 override A/B 对比。

### 10.3 这一层可以做出的决策

- 最终手工地址修补；
- 验证不同 distance 的最短闭环；
- 对单个固定 kernel 建立性能上限/下限证据。

### 10.4 仍需要的高层或硬件信息

- object role 和 K-loop 身份；
- GEP 中哪个 SSA 分量对应 K；
- Cache line、latency、带宽、硬件 prefetch；
- backend 是否把 intrinsic 选择成目标 `PRFM` hint；
- override 的 hash、kernel symbol 和 cache 隔离规则。

### 10.5 是否适合直接插入预取

适合手工实验和 fallback，不适合作为最终的通用自动化入口。

## 11. Object/Assembly 层

代表文件：`kernel.o` 及 `llvm-objdump -d` 输出。

### 11.1 可以利用的数据

- `PRFM pldl1keep/pldl2keep/...` 是否存在；
- `FMOPA/BFMOPA` 是否保留；
- `SMSTART/SMSTOP`；
- 分支和 PRFM 的机器码位置；
- 代码大小和寄存器/指令变化的间接迹象。

### 11.2 可以利用的预取语义

这一层只剩“机器行为语义”，不再有可靠 A/B/C 身份。可以验证：

```text
计划的 cache hint 是否成为预期 PRFM
每个机器循环有多少 PRFM
PRFM 是否位于 FMOPA 之前
是否因为边界分支产生额外控制开销
SME 指令是否被破坏
```

### 11.3 这一层可以做出的决策

- 接受或拒绝编译结果；
- 统计静态开销；
- 检查 baseline 与 variant 的二进制差异。

### 11.4 仍需要的硬件信息

- PRFM 是否命中目标 Cache；
- miss latency 是否真的被覆盖；
- prefetch 是否过早、过晚或污染 Cache；
- load stall、L1/L2 miss、带宽和 MLP；
- CPU 频率、绑核、NUMA 和线程环境。

机器码存在 `PRFM` 只能证明 materialization 成功，不能证明性能有效。

## 12. Hardware/Runtime 层

硬件信息不是任何 IR 自带的，必须独立测量或配置。

### 12.1 Cost Model 必需信息

```text
cache_line_bytes
L1/L2 size、associativity、共享范围
L1/L2/DRAM latency
sustained bandwidth
可并发 miss 数/MLP/MSHR proxy
hardware prefetcher 行为
SME streaming vector length
FMOPA throughput/latency
prefetch instruction cost
```

### 12.2 Runtime 验证信息

```text
kernel-only latency
warmup 与 measurement 次数
median、p25、p75
L1/L2/LLC miss
stall cycles
memory bandwidth
CPU 频率、绑核、线程数
correctness/max error
```

### 12.3 不能仅靠规格表确定的信息

- 硬件预取器是否已经覆盖 B 的顺序访问；
- packed B 是否仍停留在 L1/L2；
- `PRFM` 是否抢占 demand miss 的 MSHR；
- 多线程时共享 Cache 是否被污染；
- D=4 在目标 kernel 中对应多少真实 cycles。

这些必须通过现场 distance sweep 和计数器校准。

## 13. 跨层绑定：逻辑对象如何落到物理地址

建议新增独立 `BindingManifest`，不要把不稳定的 SSA 名称写进
`KernelSemantic` 或 `PrefetchPlan`。

示意：

```json
{
  "logical_object": "B_PANEL",
  "physical_objects": [
    {
      "stage": "input",
      "kind": "function_argument",
      "argument_index": 1
    },
    {
      "stage": "packed",
      "kind": "memref_alloc",
      "provenance": [
        "argument:1",
        "reinterpret_cast",
        "copy",
        "transpose"
      ],
      "consumer": "vector.transfer_read",
      "anchor_dimension": "K"
    }
  ]
}
```

匹配优先级应为：

```text
1. operand role + provenance 链
2. consumer/producer 结构
3. loop/index relationship
4. shape/stride
5. loc 作为辅助证据
6. SSA 文本名字不得作为合同
```

`loc` 可能在 lowering 中保留，但不能单独证明 `%x` 仍代表同一个物理对象。
shape 也不能单独使用，因为多个临时 buffer 可能具有相同 shape。

## 14. Prefetch 参数需要哪些信息

### 14.1 Distance

距离单位必须绑定物理 anchor loop：

```text
D_iter = ceil((L_effective - O_existing) / C_anchor)
```

其中：

- `L_effective`：目标层 miss 的有效延迟；
- `O_existing`：当前循环本来已有的 overlap；
- `C_anchor`：一个物理 K 迭代的 cycles。

还需要满足：

```text
future = iv + D_iter * step
future < upper_bound
```

只从高层 FLOP 推导 D 不够，因为 tile、SME SVL、peeling 和指令调度都会改变
`C_anchor`。

### 14.2 Granularity

```text
bytes_per_iteration = transfer_lanes * element_bytes
candidate_lines = ceil(bytes_per_iteration / cache_line_bytes)
```

但不能无条件预取所有 line。还要考虑：

- 相邻迭代是否重叠；
- 硬件预取器是否自动覆盖后续 line；
- 指令数和带宽预算；
- tile 是否跨页。

当前公开实现只预取 panel 起始地址，属于验证链路的最小策略。

### 14.3 Target cache/locality

需要结合：

```text
reuse distance
到消费点的时间
panel 工作集
Cache 容量/冲突
多线程共享情况
后端对 locality hint 的映射
```

高层只能提供 reuse 倾向，最终 target level 必须由硬件 profile 和实测校准。

### 14.4 Legality

Rewrite 至少必须证明：

- future iteration 在边界内；
- 预取地址来自有效 allocation；
- 生命周期覆盖预取点；
- 不跨越可能释放/重分配的 region；
- 不把输出写流错误当作只读输入；
- 动态 shape、tail 和 batch offset 正确；
- 插入后 verifier、LLVM translation 和目标代码生成通过。

## 15. 各中间合同的职责

### KernelSemantic

保存稳定的高层事实：

```text
kernel type
A/B/C logical role
M/N/K/batch
dtype
logical shape/layout
contraction/reuse/access class
```

不保存 SSA Value、Operation 指针或最终地址。

### BindingManifest

保存一次具体 lowering 中：

```text
logical object -> physical object
physical K-loop
producer/copy/transpose provenance
consumer access site
shape/stride/index evidence
```

它随编译实例和 pipeline 版本变化。

### HardwareProfile

保存：

```text
cache、latency、bandwidth、SME/SVL、MLP 和测量 provenance
```

### PrefetchPlan

只表达决策：

```text
logical/physical target
strategy
anchor loop identity
distance + unit
target cache/locality
granularity/max lines
placement
legality requirements
confidence/score
```

### MaterializationReport

记录实际改写结果：

```text
匹配到了哪个物理对象和循环
插入了几条 prefetch
哪些分支被覆盖
哪些 decision 被拒绝以及原因
LLVM prefetch/PRFM 数量
输入输出 hash
```

## 16. 当前项目的理论落点

当前正式路径应理解为：

```text
TTShared/Linalg
  提取“B 是什么、为什么值得考虑”

00_input Transform Schedule
  安排“什么时候运行 Rewrite”

MemRef/SCF/Vector snapshot
  绑定“B 现在是哪块 buffer、K-loop 和地址是什么”

HardwareProfile + Cost Model
  决定“提前几次、预取几行、目标哪级 Cache”

Pass Materialization
  生成带边界保护的 memref.prefetch

LLVM/Object
  验证 llvm.prefetch、PRFM 和 SME 指令

Runtime
  验证正确性并判断性能收益
```

所以第一版自动化不应该追求“在某一个 IR 层解决所有问题”，而应建立一个
跨层闭环：高层提供角色，中层提供地址，硬件提供代价，低层提供证据。
