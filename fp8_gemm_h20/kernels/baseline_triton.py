# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Triton block-FP8 GEMM tuned for H20 (part of the ``--fp8-gemm-backend gluon_h20`` path).

C[M, N] (bf16) = A[M, K] (fp8, 1x128 scales As[M, K/128]) @ B[N, K]^T (fp8, 128x128 scales Bs[N/128, K/128]).
BLOCK_K == 128 (one K tile per scale group). When BLOCK_N divides 128 the weight scale is a single scalar
per (tile, K group), so promotion is one FMA per element: ``acc += dot * (a_s * b_s)[:, None]``.
"""

from typing import Dict, Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _block_fp8_gemm_triton_h20_kernel(
    A, B, C, As, Bs, M, N, K, group_n,
    stride_am, stride_bn, stride_cm,
    stride_As_m, stride_As_k, stride_Bs_n, stride_Bs_k,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr, SCALAR_BS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    split = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k_tiles = K // BLOCK_SIZE_K
    k_per_split = k_tiles // SPLIT_K
    k0 = split * k_per_split

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + offs_am[:, None] * stride_am + (k0 * BLOCK_SIZE_K + offs_k)[None, :]
    b_ptrs = B + offs_bn[None, :] * stride_bn + (k0 * BLOCK_SIZE_K + offs_k)[:, None]
    As_ptrs = As + offs_am * stride_As_m + k0 * stride_As_k
    if SCALAR_BS:
        Bs_ptrs = Bs + (pid_n * BLOCK_SIZE_N // group_n) * stride_Bs_n + k0 * stride_Bs_k
    else:
        Bs_ptrs = Bs + (offs_bn // group_n) * stride_Bs_n + k0 * stride_Bs_k

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(k_per_split):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)
        if SCALAR_BS:
            acc += tl.dot(a, b) * (a_s * b_s)[:, None]
        else:
            acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        As_ptrs += stride_As_k
        Bs_ptrs += stride_Bs_k

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + split * M * N + stride_cm * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)


DEFAULT_CONFIG = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=32, num_warps=4, num_stages=3, SPLIT_K=1)


def block_fp8_gemm_triton_h20(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    config: Optional[Dict] = None,
    block_size=(128, 128),
) -> torch.Tensor:
    """A[M,K] fp8, B[N,K] fp8, As[M,K/128] f32 (row-major), Bs[N/128,K/128] f32 -> C[M,N] bf16."""
    M, K = A.shape
    N = B.shape[0]
    assert K % block_size[1] == 0 and A.is_contiguous() and B.is_contiguous()
    c = dict(config or DEFAULT_CONFIG)
    split_k = c.pop("SPLIT_K")
    if (K // block_size[1]) % split_k:
        split_k = 1
    out = torch.empty(
        (split_k, M, N), device=A.device, dtype=torch.bfloat16 if split_k == 1 else torch.float32
    )
    grid = (triton.cdiv(M, c["BLOCK_SIZE_M"]) * triton.cdiv(N, c["BLOCK_SIZE_N"]), split_k)
    _block_fp8_gemm_triton_h20_kernel[grid](
        A, B, out, As, Bs, M, N, K, block_size[0],
        A.stride(0), B.stride(0), N,
        As.stride(0), As.stride(1), Bs.stride(0), Bs.stride(1),
        BLOCK_SIZE_K=block_size[1], SPLIT_K=split_k,
        SCALAR_BS=block_size[0] % c["BLOCK_SIZE_N"] == 0, **c,
    )
    return out[0] if split_k == 1 else out.sum(0).to(torch.bfloat16)
