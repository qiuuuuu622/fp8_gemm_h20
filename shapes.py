# Qwen3.6-35B-A3B dense FP8 linears (TP=1), block [128,128]. (name, N, K)
SHAPES = [
    ("gdn.in_proj_qkvz", 12288, 2048),
    ("gdn.out_proj", 2048, 4096),
    ("attn.qkv_proj", 9216, 2048),
    ("attn.o_proj", 2048, 4096),
    ("shared.gate_up", 1024, 2048),
    ("shared.down", 2048, 512),
]
# prefill token counts
MS = [256, 1024, 2048, 4096, 8192, 16384]

# Extended set: TP=2/4 splits of the same layers + ragged prefill M
EXT_SHAPES = [
    ("qkvz.tp2", 6144, 2048), ("qkvz.tp4", 3072, 2048),
    ("out.tp2", 2048, 2048), ("out.tp4", 2048, 1024),
    ("qkv.tp2", 4608, 2048), ("qkv.tp4", 2304, 2048),
    ("sh_gu.tp2", 512, 2048), ("sh_down.tp2", 2048, 256),
]
EXT_MS = [100, 333, 1000, 3000, 5000, 12000]
