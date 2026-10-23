# 122B TP4 evidence index

这些是已经完成的 2026-10-22 / 23 单 rank kernel 实验，不是服务压测。

- `v1/`：原 c3/c11 调参，110 点对比、80-entry 原始表、精度 / integration 摘要。
- `ncu/`：小 M 诊断、clean timing 消融与 NCU 指标摘要。没有将 profiler 内的耗时用于最终速度比较。
- `v2/`：260 点最终 comparison、110-entry 冻结表、汇总、独立发布包检查、双 stream graph、cold-cache 检查、历史 sanitizer 日志。
- `provenance.json`：35B 基础仓库提交与 122B 原 kernel / config SHA256。

`v2/comparison.csv` 的 `holdout=True` 是旧的“非 kernel 调参点”标记。最终 130 点已用于策略回归，只有 M={47,79,194,509}×5 是最后的独立留出集。详见 `v2/POLICY_CHANGE.txt`。

大型 NCU 二进制、数千条 sweep 中间记录、已中止的扩展 M sweep、模型权重没有复制进 Git。未完成的 sweep 不计入结果。`summary.json` 中的历史路径只用于来源说明，不是运行依赖；公共 package 不读取本目录。
