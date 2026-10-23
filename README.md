# H20 block-FP8 GEMM：35B / 122B 共用实现与实验记录

这个仓库为 H20 上的 dense FP8 linear 提供一套 Triton / Gluon kernel、统一的配置格式和调用入口。当前有两个经过测量的 profile：**35B TP1** 和 **122B TP4 单 rank**。模型名称沿用原实验记录；122B 的模型配置架构为 `Qwen3_5Moe`。routed experts 的 fused-MoE 不在范围内。

35B 的起点是缩短 SGLang DeepGEMM 全 M 预编译带来的启动时间，同时保持 prefill 性能。122B 的工作是在不同 TP 分片形状上重新调参，并针对小 M 的 padding、tail split-K 和 scale 访问开销继续改 kernel。两者共享数学定义和 kernel 库，**选参表与插值策略按 profile 隔离**，不把一个模型的最优参数直接套给另一个模型。

所有下文的加速比均为 **参考实现耗时 / 当前实现耗时**，大于 1 表示当前实现更快。跨 shape 汇总使用几何平均（GM），不代表真实流量加权收益。

## 快速使用

实验环境：NVIDIA H20 96 GB / 78 SM，CUDA 13.0，driver 550.54.15，PyTorch 2.13.0+cu130，Triton 3.7.1。SGLang 集成及 DeepGEMM 对照使用 SGLang 0.5.20、sgl-deep-gemm 0.2.0。尚未验证其他版本或 GPU。

```bash
# 使用已经安装好上述 GPU 依赖的环境；不会替换已有 torch / triton
python -m pip install --no-deps -e .
```

```python
from fp8_gemm_h20 import matmul, select

# A: [M,K]、B: [N,K]，均为 contiguous CUDA torch.float8_e4m3fn
# As: [M,K/128]、Bs: [N/128,K/128]，均为 CUDA float32
C = matmul(A, B, As, Bs, profile="35b_tp1")   # BF16 [M,N]
C = matmul(A, B, As, Bs, profile="122b_tp4")  # 根据当前模型二选一
entry, reason = select("122b_tp4", 200, 3072, 2048)  # 无 GPU 也能查看选参
```

计算是 `C = dequant(A, As) @ dequant(B, Bs).T`：activation 按 token × 128 K 分组，weight 按 128 N × 128 K 分组；每个 K 分组的 FP8 dot 乘对应 scale 后累加到 FP32，最后输出 BF16。不是 per-tensor FP8 GEMM。N、K 必须为 128 的倍数，公共接口仅覆盖配置中的 N/K 和 `1 ≤ M ≤ 16384`；不支持的 shape 显式报错。scale 支持 stride，scale-TMA 在 As 非行内连续时回到 v2。

**目录与唯一运行配置：**

| 路径 | 用途 |
|---|---|
| [`fp8_gemm_h20/dispatch.py`](fp8_gemm_h20/dispatch.py) | 两个 profile 共用的 tensor 校验与 `matmul` |
| [`fp8_gemm_h20/config.py`](fp8_gemm_h20/config.py) | 不依赖 CUDA 的配置校验、缓存和选参 |
| [`fp8_gemm_h20/configs/`](fp8_gemm_h20/configs/) | 相同 schema 的 `35b_tp1.json`、`122b_tp4.json`；运行时唯一选参来源 |
| [`fp8_gemm_h20/engine.py`](fp8_gemm_h20/engine.py)、[`kernels/`](fp8_gemm_h20/kernels/) | 一个延迟加载的 kernel 注册表；两个模型复用 c3/c11 实现 |
| [`scripts/validate.py`](scripts/validate.py) | 两个模型共用的精度、CUDA Graph、双 stream、stride scale、计时脚本 |
| [`sgl_patch/`](sgl_patch/) | 可选 SGLang adapter，`FP8MM_PROFILE` 选择模型 |
| [`runs/`](runs/)、[`candidates.jsonl`](candidates.jsonl) | 35B 历史原始表、A/B 和 candidate 日志 |
| [`experiments/122b_tp4/`](experiments/122b_tp4/) | 122B v1 / NCU / v2 原始数值、验证结果与来源哈希 |
| 根目录 `opt2.py` / `gl5.py` / `dispatch.py` | 旧实验脚本兼容入口，c3/c11 已转发到共用实现 |
| [`docs/history/35b-original.md`](docs/history/35b-original.md) | 原 README 快照，包含历史阶段的结论与限制 |

配置说明见 [配置格式与迁移规则](docs/configuration.md)。历史 `runs/dispatch.json` 留作证据；新 sweep 的输出必须经过检查后转换到 package profile，不能只覆盖历史表期待新运行时生效。

## 两个模型的形状

表中为 **每个 TP rank 的 (N,K)**，不是完整模型权重尺寸。35B TP1 有 6 种层用途、5 个唯一 N/K；原 120 个 benchmark 行包含 out/o 的重复形状，不能当作 120 个唯一 shape。122B TP4 合并相同 N/K 后有 5 组，单 GPU 测量一张分片，未进行四卡通信测试。

| dense linear | 35B TP1 (N,K) | 每次 forward 次数 | 122B TP4 (N,K) | 每次 forward 次数 |
|---|---|---:|---|---:|
| GDN qkvz | (12288,2048) | 30 | (5120,3072) | 36 |
| GDN out + attention o | (2048,4096) | 30 + 10 | (3072,2048) | 36 + 12 |
| attention qkv | (9216,2048) | 10 | (4608,3072) | 12 |
| shared gate_up | (1024,2048) | 40 | (512,3072) | 48 |
| shared down | (2048,512) | 40 | (3072,256) | 48 |

122B 尺寸核对了权重 safetensor header 和 SGLang sharding：TP4、attention DP 关闭、DCP=1、shared expert 按 TP 分片。改变这些条件必须重新核对尺寸。35B 历史还测过部分 TP2/4 分片，见 [`shapes.py`](shapes.py)；它们是研究测试集，不是新的已调优 runtime profile。

## 实验结果概览

| 模型 / 阶段 | 指标 | 结果 | 证据 |
|---|---|---|---|
| 35B c11 | cold server ready | 334.2 → **173.6 s**，减少 48% | [`startup_ready.txt`](runs/startup_ready.txt) |
| 35B c11 | warm server ready，两轮 | 183.6 / 178.6 → **117.7 / 114.2 s** | 同上 |
| 35B c11 | 8192-token prefill TTFT | 295.3 → **268.8 ms**，**1.098×**；交换 GPU 后 **1.096×** | [`round1`](runs/ab_e2e_round1.json)、[`round2`](runs/ab_e2e_round2_swapped.json) |
| 35B c11 | 全 120 行 kernel benchmark vs DG | **1.131× GM**，83/120 行获胜 | [`bench_prefill_c11.csv`](runs/bench_prefill_c11.csv) |
| 122B TP4 v1 | 全 110 点 kernel vs DG | **1.112× GM**，77/110 获胜 | [`v1/summary.json`](experiments/122b_tp4/v1/summary.json) |
| 122B TP4 v2 | 全 260 点 kernel vs DG | **1.121× GM**，201/260 获胜；vs v1 **1.038×** | [`v2/summary.json`](experiments/122b_tp4/v2/summary.json) |
| 122B TP4 v2 | 训练网格内 M≤512，70 点 | vs DG **1.176× GM**，53/70 获胜；vs v1 **1.070×** | 同上 |

**35B 的 TTFT / 启动收益来自历史 SGLang server A/B；122B 的数字全部是 kernel 级。没有测量 122B 的端到端 TTFT、启动时间或 TP4 通信收益。** 两组 M 集合与计时协议不同，不应将 GM 当成模型之间的排名。

## 35B：从启动预编译到 c11

### 优化动机与演进

原 DeepGEMM 路径会在启动阶段预编译 M=1..16384，增加版本发布等待时间。最初先优化 Triton 的 block-FP8 基线，随后用 Gluon 显式控制 Hopper TMA / WGMMA；最后接入 SGLang，检查 kernel 收益能否转化为启动和 prefill 收益。

| 阶段 | 手段 / 要解决的问题 | 结果与取舍 |
|---|---|---|
| c1 / c2 | 原 kernel 调参；TMA + persistent grid | 没有稳定收益，c2 仅 0.68–0.95× |
| **c3** | BLOCK_N 整除 128 时把 Bs 化为每 tile 一个 scalar；合并 scale、single-FMA promotion；按 shape 调参 | 相对原始 kernel 的模型形状 GM 为 1.138×，保留 |
| c5 / c7 | 手动 K unroll、跨迭代延迟 promotion | `tl.dot` 仍序列化，或 register 开销过大，未采用 |
| c8 → **c9** | Gluon TMA ring + mbarrier、SWAP_AB、小 M tail-wave split-K | 小 M 候选改善；随后发现 async 在机器码中被序列化 |
| c10 / e1 | 更早取 scale、更深 pipeline、每 CTA 多 tile | 没有解决当时主要瓶颈，未采用 |
| **c11** | 成对发出 WGMMA：`wait(1) → promote k → wait(0) → promote k+1` | 消除 ptxas C7514；与 c3 共同组成 80-entry 配置表 |
| c12 | warp-specialized TMA producer / MMA consumer | 7 个 shape 中 6 个慢于 c11（0.71–0.95×）；限制 maxnreg 也未改善 |

关键证据是 [`ptxas_c3_c9.txt`](runs/ptxas_c3_c9.txt) 和 [`ncu_r1_summary.txt`](runs/ncu_r1_summary.txt)：c9 的 PTX 虽写了 `wait_group 1`，SASS 仍因 accumulator 的非 WGMMA 访问而序列化，27% stall samples 位于 WARPGROUP.DEPBAR。c11 让同一 loop 内的两次 WGMMA 有明确重叠，但 **loop back-edge 仍没有未完成 WGMMA**。不能把它描述为任意深度的跨迭代计算流水线。

早期 v1 没有 NCU 权限时对“小 M 是 scale latency 还是 wave 利用率”的判断属于当时假设；后续 NCU 与 c11 实验补充、修正了它。完整旧记录保留在 [历史 README](docs/history/35b-original.md)。

### Kernel 结果与端到端结果

35B kernel protocol：6 层用途 × 20 M，3 轮 CUDA Graph timing 的中位数，JIT 不计入。M 包含 1、37、200、777 等非整齐点，到 16384 为止。

| M 范围 | 行数 | vs DG GM | 最差 / 最好 | DG 更快行数 |
|---|---:|---:|---|---:|
| M≤128 | 42 | 1.171× | 0.802 / 2.102 | 18 |
| 128<M≤512 | 24 | 0.993× | 0.777 / 1.298 | 15 |
| 512<M<4096 | 24 | 1.129× | 0.847 / 1.644 | 4 |
| M≥4096 | 30 | **1.196×** | 1.020 / 1.357 | 0 |
| 全部 | 120 | **1.131×** | 0.777 / 2.102 | 37 |

SGLang A/B 使用暖缓存、随机且每次不同的 token，防止 radix prefix reuse；每 case 交替请求两台 server，取 3 次中位数，第二轮交换 GPU。核心 TTFT：

| 输入 / 并发 | DG TTFT (ms) | ours TTFT (ms) | 第一轮 / 换卡后加速比 |
|---|---:|---:|---|
| 1024 tokens | 51.1 | 48.6 | 1.052 / 1.040 |
| 4096 tokens | 150.2 | 138.3 | 1.086 / 1.088 |
| 8192 tokens | 295.3 | 268.8 | 1.098 / 1.096 |
| 16384 tokens | 666.7 | 612.7 | 1.088 / 1.080 |
| 32768 tokens | 1658.9 | 1546.0 | 1.073 / 1.063 |
| 8192 tokens × 4 并发 | 1143.7 | 1031.0 | 1.109 / 1.098 |

32768-token 请求通过 chunking 执行，不表示 kernel profile 支持 M=32768。启动实验将权重预先放入共享内存，减少存储抖动；启动的是 `sglang.launch_server`，移除了 Dynamo、PD 与 speculative decoding。kernel-only 冷编译从 DG 207.5 s（FAST_WARMUP 143.9 s）到 ours 17.0 s，不能与 server ready 时间相加。

8192-token 时 dense GEMM 总耗时约 94.8 → 84.2 ms（1.126×）；MoE、GDN、attention 等没有被替换，所以整体 TTFT 收益约 1.10×。历史 eager Python 调用开销 Gluon 约 65 µs、DG 约 20 µs；CUDA Graph 隐藏这部分开销，这不是 eager 模式的收益承诺。

历史精度：585 个 kernel case，worst normalized max error 7.5e-3；4032 个 c11 配置加 graph / 双 stream 检查；真实权重 patch 测试最大相对差 1.05e-3。teacher-forced NLL 为 0.6580 → 0.6618（+0.58%，4 文本 / 348 tokens）。两实现 greedy 输出并非逐字一致，尚无下游任务集评测；不能据小样本 NLL 声称质量完全等价。

## 122B TP4：重新调参、NCU 诊断与 kernel v2

### v1：相同 kernel，新的 TP4 几何形状

2026-10-22 先核对实际权重分片，对 5 个唯一 N/K、16 个训练 M 调 c3/c11，另外 6 个 M 作检查，共 110 点。vs 未调优默认实现为 **1.452× GM**，vs DG 为 **1.112×**。M≥4096 的 20 点全胜 DG，但 M≤512 仅 40/70 点胜出，说明大 M 调参成功不能推导出小 M 同样合适。

330 个精度检查和 graph / 双 stream 检查通过；M=8192 按 dense 层次数加权，DG 78.538 ms → tuned 70.618 ms。这仍只是 dense GEMM 时间，不是模型 forward / TTFT。

### NCU：为什么小 M 仍然输

2026-10-23 在 GPU6、7 上分离干净 CUDA Graph timing 与 NCU profiling。诊断保存了 19 份 profile，并试验 277 个有效配置；仓库收录 [数值摘要](experiments/122b_tp4/ncu/ncu_summary.csv) 和 [消融结论](experiments/122b_tp4/ncu/analysis.json)。

| 观察 | 证据 / 原因 | 对应调整 |
|---|---|---|
| M200、N3072、K2048 的 M tile 太大 | 64×64 改 16×128 + SWAP，CTA 192→312，token padding 利用率 78.1%→96.2%，SM active-cycle max/min 1.58→1.09 | 更细 token tile、SWAP_AB 和 wave 分布联合选参 |
| M37 同形状的 tail split-K 过度 | 仅 SPLIT 4→1，CTA 342→144，约 8.14→6.18 µs | split-K 不是越多越好，要考虑 K 工作量和 resident CTA |
| M200 qkvz 多参数共同影响 | tile / swap / split / mode / group 联合变更才有效；单改 MODE 或 SPLIT 未复现收益 | 保留消融，不把联合收益归因于单一技巧 |
| scale load 有等待 | M37 的 SASS long-scoreboard samples 集中在等待 scale LDG 的 FMUL | v2 尝试更早加载 scale、部分 shape 用 scale-TMA |
| narrow-N、short-K 调参收益有限 | 部分 case 原 sweep 只有约 1% 改善 | 新增固定 K=256 kernel 与更多 WGMMA 调度变体 |
| DG 的 tile 更贴合 78 SM | 部分形状采用 N tile=80，当前 kernel 的 weight-scale 规则要求 tile 整除 128 | 不能直接把 80 填入当前参数表；涉及 scale / tile 结构改变 |

较高 occupancy 不必然较快；某个旧 kernel 的 occupancy 为 15.53%，DG 为 13.30%，旧 kernel 仍较慢。短 kernel 的多 pass NCU L2 指标可能出现不可信数值，stall sample 占比也不是时间占比；性能结论采用未挂 profiler 的计时。

### v2：实际采用与未采用的手段

| 实现 / 手段 | 解决的问题 | 110 个训练点的最终选择数 |
|---|---|---:|
| 原 c11 / `gluon` | 保留已稳定的大 M / 合适小 M 路径 | 34 |
| 原 c3 / `triton` | 部分矩形和短 K 的常规 Triton 路径更好 | 18 |
| `v2` | M/N/K constexpr；SPLIT=1 编译期去分支；scale prefetch；可选 K unroll；MODE=3 四次 async WGMMA；更细 BT/BW | 53 |
| `scale_tma` | 将 activation scale 先搬入 shared memory，缓解部分 scale LDG 等待 | 4 |
| `shortk` | K=256 展开成两次独立 128-K dot / scale promotion，消除通用 K-loop 和 workspace | 1 |
| warp-specialized producer / consumer | 试过独立 TMA producer 和计算 warp，以及寄存器上限 | **0**，保留实验代码但不进入 dispatch registry |

主搜索 3105 次配置检查通过；scale-TMA 83 次 refinement 只有 4 个点被采用。更多 stage、更复杂 pipeline、warp specialization 都不是普遍收益，最终按 shape 的实测结果选择。

### Dispatch 也是优化的一部分

kernel 表冻结后，最初“只要 M tile 数一样就借配置”的路由让 M65 / M129 出现约 18% 退化。改为仅允许 **距离 ≤8 token 且 `ceil(M/BT)` 相同** 的 anchor；否则回到 v1 的最近 M 配置。

随后换卡复测确认：将 M512 的 short-K 配置借给 M511 会退化约 8%。最终 short-K **只在 exact M 命中时使用**。M510/511 回到原 kernel，M512 的收益仍保留。完整更正记录见 [`POLICY_CHANGE.txt`](experiments/122b_tp4/v2/POLICY_CHANGE.txt)。两次更正期间 110-entry kernel 参数表没有变化。

评估集合也随之如实划分：110 个调参点；最初 90 个加后续 40 个 off-grid 点因参与策略修正，归为 **130 个策略回归点**；最后 M={47,79,194,509}×5 是 **20 个未参与选参或策略修正的独立留出点**。不把全部 150 个 off-grid 点冒称为独立 holdout。

### v2 最终对比

| 范围 | 点数 | vs 同轮 v1 GM | vs DG GM | 胜 DG 点数 | 相对 v1 退化 >3% |
|---|---:|---:|---:|---:|---:|
| 调参网格 | 110 | 1.046× | 1.162× | 92 | 0 |
| 其中 M≤512 | 70 | **1.070×** | **1.176×** | **53** | 0 |
| 其中 M≥4096 | 20 | 1.000× | 1.156× | 20 | 0 |
| 策略回归点 | 130 | 1.031× | 1.101× | 96 | 0 |
| 最终独立留出点 | 20 | 1.042× | 1.036× | 13 | 0 |
| 全部 | 260 | **1.038×** | **1.121×** | **201** | **0** |
| 独立 cold-cache 检查 | 25 | 1.024× | 1.121× | 24 | 0 |

历史 v1 小 M 是 **40/70** 胜 DG；在 v2 同轮重测中，v1 为 **42/70**、v2 为 **53/70**。两者的差异包含计时波动，不能写成同一轮的 40→53。v2 仍有 **17 个小 M 点慢于 DG，其中 9 个慢超过 3%**；全 260 点相对 v1 最差约退化 1.34%。

| (M,N,K) | 同轮 v1 (µs) | v2 (µs) | DG (µs) | v2 vs v1 |
|---|---:|---:|---:|---:|
| (37,3072,2048) | 8.128 | **5.416** | 6.433 | 1.501× |
| (200,3072,2048) | 17.528 | **12.994** | 14.386 | 1.349× |
| (200,5120,3072) | 37.523 | **31.059** | 34.116 | 1.208× |
| (128,512,3072) | 6.344 | 5.586 | **5.420** | 1.136× |
| (64,4608,3072) | 10.830 | 10.080 | **9.893** | 1.074× |
| (512,3072,256) | 7.077 | 6.498 | **6.320** | 1.089× |

[完整 260 点 CSV](experiments/122b_tp4/v2/comparison.csv) 的 `holdout` 字段表示原先是否在 kernel 搜索网格外；最终独立 holdout 的定义以上述四个 M 为准。

Warm-cache protocol：每实现交替测 5 轮 × 80 ms CUDA Graph，取中位数；JIT、host dispatch、DG scale 布局转换不计入。21 点换卡复核使用 9×100 ms。cold-cache 25 点在每次 graph 前对 256 MiB buffer 做读写驱逐，共 40 samples，驱逐不计入 event 时间；这不是服务流量模拟。策略更正后重新验证 / 计时发生变化的选择，未变化项保留其已有测量，没有冒称所有数据在最后一轮从头重跑。

精度：260×4 分布 = **1040 次检查**，worst normalized max error **0.003728**；发布包另换 seed 检查 260 次，worst **0.003794**。28 条代表路径在两个独立 stream 各 replay 100 次通过；memcheck / synccheck 各检查 14 条路径，0 error。精度指标为 `max(abs(out-ref))/max(abs(ref))`，不是逐元素 relative error。

## SGLang 接入

```bash
export FP8MM_HOME=/path/to/fp8_gemm_h20
export PYTHONPATH="$FP8MM_HOME/sgl_patch:$PYTHONPATH"
export FP8MM_PROFILE=35b_tp1             # 或 122b_tp4
export FP8MM_ENABLE=1
export FP8MM_DISABLE_DG_PRECOMPILE=1
python -m sglang.launch_server ...       # 使用你自己的模型和 TP 参数
# 回退：unset FP8MM_ENABLE，重启 worker
```

adapter 只替换原本选择 DeepGEMM 的 dense linear，使用 SGLang 原 activation quantizer，随后调用统一 `matmul`，最后处理 bias / reshape。预量化输入、非 BF16 输入、非 FP32 weight scale、UE8M0 scale、非 [128,128] block、不支持的 M/N/K 等走原函数；公共 kernel 包本身没有 DG 隐式回退。

35B 旧版 adapter 有历史真实权重及 server A/B；统一版 adapter 的模型选择 / fallback 经自动检查，但 **本次整理没有重跑 server A/B，122B 的 server 接入尚未端到端验证**。`FP8MM_DISABLE_DG_PRECOMPILE=1` 是进程级设置，未被本 profile 覆盖的层如果回退 DG，仍可能在首次请求 JIT。`dynamo.sglang` / PD 路径也尚未验证。

## 复现与本次整理的验证

```bash
# CPU 配置检查：不需要 torch / CUDA
python -m unittest discover -s tests -v

# 只使用授权的 GPU6、7；每张卡同一时间一个 benchmark
CUDA_VISIBLE_DEVICES=6 python -m scripts.validate \
  --profile 35b_tp1 --timing --output validation_35b.json
CUDA_VISIBLE_DEVICES=7 python -m scripts.validate \
  --profile 122b_tp4 --timing --output validation_122b.json

# 可选：加 --deepgemm 对照 DG；加 --reference /path/to/frozen/reference
# 35B reference 是旧仓库根目录；122B reference 是含 fp8_gemm_122b/ 的目录。
# 脚本记录分布精度、逐位迁移对照、双 stream graph、stride scale 与计时 samples。

# 历史 35B 专项复现（默认 FP8MM_PROFILE=35b_tp1）
IMPL=dispatch python test_correct.py
python test_c11_paths.py
python bench_prefill.py /tmp/bench_prefill.csv
```

本次统一将原 35B c3/c11 和 122B baseline 收敛为同一实现，保留 v2 / scale-TMA / short-K 各自不同的调度结构。实际迁移检查与结果见 [`experiments/unification/`](experiments/unification/)。历史实验数字没有被本次短程重测替换；新输出用于判断配置 / 代码搬迁有没有改变行为。

本仓库保留可审计的 JSON / CSV / 日志摘要；大型 `.ncu-rep`、完整 tuning 过程与模型权重不纳入源码。122B 历史来源及原配置哈希在 [`provenance.json`](experiments/122b_tp4/provenance.json)。当前 profile 转换了字段格式，因此不能拿新 JSON 的文件哈希与旧文件直接比较，应对比选中的 impl/config。
