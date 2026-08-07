# 现场预取配置（07_onsite_workflow/config）

供明天现场预取实验使用的宏观配置。IR 里只有微观信息（4x4 微块、stride），
大矩阵分块和硬件参数不进入 IR，全部写在这里。

## 文件

| 文件 | 内容 | 状态 |
|---|---|---|
| `onsite_hardware_profile.json` | 硬件画像（缓存/TLB/SVL/带宽/延迟），按 `03_legality_cost_model/schemas/hardware_profile_schema.json` 规范 | 模板值，`calibrated: false` |
| `onsite_bmm_case.json` | 现场 bmm 用例：M/N/K/batch、TILE、grid、复用度、每 program 数据量、预取目标参数 | 分块参数已确认 |

## 已确认（现场测试参数，2026-08-07）

```text
dtype = bf16
M = 8192, N = 2048, K = 64, batch = 4
TILE_M = TILE_N = TILE_K = 4, GROUP_M = 1
grid = [2048, 512, 4]，共 4,194,304 programs
K 循环 = 16 次迭代
A 行 stride = 128 B，B 行 stride = 4096 B
复用：A ×512（跨 pid_n），B ×2048（跨 pid_m）
cache line 假设 64 B：A 每行每迭代 8 B，8 迭代一条 line
```

## 明天现场核对清单（都待实测，改完置 `calibrated: true`）

1. `lscpu`：核数、CPU 型号；`cat /proc/cpuinfo` 确认 SME 特性（`sme` flag）
2. 缓存：`/sys/devices/system/cpu/cpu0/cache/index{0,1,2}/` 读 size/line/shared_cpu_list，
   确认 L1d/L2 容量与共享域，cache line 是否 64 B
3. `rdvl`（或小 C 程序）确认 SVL 是否 512 bit
4. 线程：launcher metadata 的 num_threads / `OMP_NUM_THREADS` / 绑核方式，
   填进 `onsite_bmm_case.json` 的 `runtime_onsite_fill`
5. 延迟/带宽：`hardware_calibration/run_calibration.sh`（SME2-public）实测后
   替换 execution_units 里的占位值
6. 基线 miss：`perf stat -e cache-misses,cache-references` 跑 baseline
   （0 PRFM），记录到 `runtime_onsite_fill.measured_miss_baseline`

## 预取目标（写入 case 配置，供 pass/实验使用）

```text
primary   = A_panel（source-A，4 条静态 prefetch，distance 8，issue-every 8）
secondary = B_panel（4 条静态，locality 单独实验）
locality  = L2=2（默认） / L1=3（对比）
```
