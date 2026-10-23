"""Compatibility wrapper for the historical c3 experiment API."""
import json
from pathlib import Path
from fp8_gemm_h20.kernels.baseline_triton import (
    DEFAULT_CONFIG as DEFAULT,
    _block_fp8_gemm_triton_h20_kernel as _w8a8_block_fp8_matmul_hopper_v2,
    block_fp8_gemm_triton_h20,
)
_CFGS = json.loads((Path(__file__).parent / "runs/opt2_configs.json").read_text())

def pick_cfg(M, N, K):
    table = _CFGS.get(f"{N},{K}")
    return table[min(table, key=lambda m: abs(int(m) - M))] if table else DEFAULT

def matmul(A, B, As, Bs, cfg=None, block=(128, 128)):
    return block_fp8_gemm_triton_h20(A, B, As, Bs, cfg or pick_cfg(A.shape[0], B.shape[0], A.shape[1]), block)
