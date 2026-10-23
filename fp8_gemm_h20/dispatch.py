"""Validated public entry point; kernel code and configs are model-independent."""
from .config import _select
from .engine import run

def matmul(A, B, As, Bs, *, profile="35b_tp1"):
    """FP8 A[M,K], B[N,K], FP32 block scales -> BF16 C[M,N].

Unknown shapes and M outside 1..16384 raise ValueError. There is no hidden
DeepGEMM fallback; the optional SGLang adapter handles its own fallback.
"""
    import torch
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be rank-2 tensors")
    m, k = A.shape
    n, bk = B.shape
    if k != bk or k % 128 or n % 128:
        raise ValueError("A/B dimensions must match and N,K must be multiples of 128")
    if A.dtype != torch.float8_e4m3fn or B.dtype != torch.float8_e4m3fn:
        raise ValueError("A and B must be float8_e4m3fn")
    if not A.is_cuda or any(t.device != A.device for t in (B, As, Bs)):
        raise ValueError("All tensors must reside on the same CUDA device")
    if not A.is_contiguous() or not B.is_contiguous():
        raise ValueError("A and B must be contiguous row-major tensors")
    if As.dtype != torch.float32 or Bs.dtype != torch.float32:
        raise ValueError("Scales must be float32")
    if tuple(As.shape) != (m, k // 128) or tuple(Bs.shape) != (n // 128, k // 128):
        raise ValueError("Expected As[M,K/128] and Bs[N/128,K/128]")
    entry, _ = _select(profile, m, n, k)
    with torch.cuda.device(A.device):
        return run(entry, (A, B, As, Bs))
