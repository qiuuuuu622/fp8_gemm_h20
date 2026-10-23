"""Compatibility wrapper for the historical c11 experiment API."""
from fp8_gemm_h20.kernels.baseline_gluon import (
    DEFAULT_CONFIG as DEFAULT,
    _block_fp8_gemm_gluon_h20_kernel as _w8a8_block_fp8_matmul_hopper_gl5,
    _mma, _promote, _bufs, _desc,
    block_fp8_gemm_gluon_h20,
)

def pick_cfg(M, N, K):
    return DEFAULT

def matmul(A, B, As, Bs, cfg=None, block=(128, 128)):
    return block_fp8_gemm_gluon_h20(A, B, As, Bs, cfg, block)
