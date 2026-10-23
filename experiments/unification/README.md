# 统一实现的迁移验证（2026-10-23）

这里只记录代码 / 配置整理后的检查，不把新计时改写为历史模型性能结论。

| 检查 | 35B TP1 | 122B TP4 |
|---|---:|---:|
| 实际执行 GPU | 6 | 7 |
| shape 点数（唯一 N/K） | 100 | 260 |
| 分布 / 精度次数 | 4 / 400 | 4 / 1040 |
| 与原实现输出逐位一致 | 400/400 | 1040/1040 |
| 相对 FP32 参考的最大 normalized error | 0.003797 | 0.003810 |
| 双 stream graph 代表路径 | 13 | 27 |
| 每条路径每 stream replay 次数 | 100 | 100 |
| 非连续 scale 代表路径 | 13 | 27 |
| 迁移前后计时点数 | 35 | 40 |
| GM（原版耗时 / 统一版耗时） | 0.9991× | 0.9992× |
| 最差比值 | 0.9814× | 0.9784× |
| 耗时增加 >3% 点数 | 0 | 0 |

协议：`scripts/validate.py`，非零随机输入，交替 5 轮 × 60 ms CUDA Graph timing；JIT、CPU dispatch 不计入。不是重新搜索参数，没再跑端到端 server A/B。35B 原版为 Git `80a3f43a0d6f64ea00bebb9105a46f91baff5d2c`；122B 原版为冻结的 `20261023_kernel_v2/release/fp8_gemm_122b`。原版和统一版在同一进程、同一卡运行；两张卡各自一项任务。

- [`validation_35b.json`](validation_35b.json) / [`validation_122b.json`](validation_122b.json)：完整精度、逐位对照、各轮 timing、graph / scale 结果。
- [`summary.json`](summary.json)：上表机器可读汇总。
- [`cpu_tests.log`](cpu_tests.log)：10 项 CPU 检查，含 122B 五个 N/K 的 M=1..16384 全范围策略检查、全部配置迁移检查和 SGLang adapter mock 检查。
- [`runtime_checks.json`](runtime_checks.json)：输入类型、scale 形状、跨设备、非连续输入、未知 shape 拒绝检查；current device=0 时输入在 device=1，输出仍在 device=1 且正确。
- [`wheel_check.json`](wheel_check.json)：独立 wheel 解包目录中测试两个 profile 的 7 个 profile/kernel 组合，不依赖仓库根目录或 SGLang overlay。

本次复用了原 kernel 的指令实现：package 内 6 个 kernel 文件与冻结 release 的对应源码 SHA256 一致（包括未纳入选择器的实验 ws）。c3/c11 的原根目录文件变为兼容 wrapper，共用这套代码。旧研究脚本仍留在根目录；它们不是 wheel 的 runtime 依赖。

重现迁移对照需准备上述冻结 reference，然后运行 README 中带 `--reference` 的命令。`scripts/check_runtime.py` 要求恰好两张可见卡：`CUDA_VISIBLE_DEVICES=6,7 python -m scripts.check_runtime`。
