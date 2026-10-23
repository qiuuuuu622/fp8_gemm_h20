# Profile schema v1

`fp8_gemm_h20/configs/<profile>.json` 是运行时唯一配置。公共接口用显式 profile 避免模型/TP 混淆；默认 `35b_tp1` 保持原使用习惯。

```json
{
  "schema_version": 1,
  "profile": "35b_tp1",
  "hardware": "NVIDIA H20 (SM90, 78 SMs)",
  "block_size": [128, 128],
  "m_range": [1, 16384],
  "policy": {"kind": "log_nearest"},
  "shapes": [{"name": "shared.down", "N": 2048, "K": 512, "count": 40}],
  "entries": {
    "2048,512": {
      "128": {"impl": "triton", "config": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3, "SPLIT_K": 1}}
    }
  }
}
```

上面仅为结构示例；真实最优值见配置文件，不应复制这个缩减示例用于生产。

两个 profile 的 `entries[N,K][M]` 都是 `{impl, config}`；122B 新表额外保留 `anchor_us`，用于复现已验证的同 tile 借参排序。impl 统一为 `triton/gluon/v2/scale_tma/shortk`，具体 kernel 参数不强行重命名，以便逐项比对原实验。训练测量的 `anchor_us` 不是在线预测延迟。

35B 的 `log_nearest` 先 exact 命中，再按 `abs(log(anchor/M))` 选择。122B 使用：

```json
{"kind": "nearby_equal_tiles", "max_distance": 8, "exact_only": ["shortk"]}
```

1. exact M 命中；
2. 排除 exact-only kernel，要求距离 ≤8 且 `ceil(M/token_tile)` 与 anchor 相同，按 `anchor_us`、log 距离、anchor 排序；
3. 没有候选则用 `fallback_entries` 的 log-nearest，即 2026-10-22 v1 表。

fallback 仍是自有 Triton/Gluon kernel；没有隐藏 DeepGEMM。配置缓存不自动热更新：编辑配置后应重启 Python 进程。`select` 返回拷贝，调用方不会意外改掉缓存。package import 与选参不初始化 GPU；真正调用 kernel 时才加载 torch / Triton。

`35b_tp1` 是 80 entries / 5 唯一 N/K；`122b_tp4` 是 110 entries + 80 fallback entries / 5 N/K。校验允许历史表中的 SPLIT 大于 K tile 数：底层 wrapper 按原行为将不能整除 K tile 数的 split 归一为 1。不能简单删除这些条目，否则会改变历史选参。MODE=3 的四 tile 分组限制另外检查。

根目录的旧 `dispatch.py` 保留 `pick`、`_TABLE`、`IMPLS` 和 `DISPATCH_LOOKUP=ceil`，供历史脚本使用。它默认 35B，对未调优的 TP 分片仍使用历史 c3 fallback；新公共 API 对 profile 外 shape 报错，SGLang adapter 则调用原函数。旧 `runs/opt2_configs.json` 仅用于 c3 独立实验的兼容入口，不参与公共 profile 的选参。

添加 profile 时需要：核对真实 sharding；记录 shape/count、环境、测量方法；保存完整配置和基线；增加 correctness / boundary / graph 检查，再加入 `config.py` 的 profile 注册。不要从总参数量推导 tensor shape，也不要把 35B TP2/4 的历史测试误认为已完成的新 profile。
