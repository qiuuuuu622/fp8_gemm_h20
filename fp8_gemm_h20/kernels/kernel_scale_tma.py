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
"""Gluon (Triton >= 3.7) block-FP8 GEMM for Hopper, tuned on H20 (``--fp8-gemm-backend gluon_h20``).

C[M, N] (bf16) = A[M, K] (fp8, 1x128 scales As[M, K/128]) @ B[N, K]^T (fp8, 128x128 scales Bs[N/128, K/128]).

- TMA ring of NUM_STAGES A/B tiles guarded by mbarriers.
- MODE 1 (pairwise wgmma): each loop iteration issues two async wgmmas (k, k+1), then
  wait(1) -> promote k while k+1 runs -> wait(0) -> promote k+1. No wgmma is in flight across the
  loop back-edge, which avoids ptxas C7514 ("wgmma.mma_async instructions are serialized"),
  the reason a single-in-flight async loop ran fully serialized. MODE 2 is the synchronous reference.
- SWAP_AB puts weight rows on the wgmma M side (>= 64) and tokens on the N side (>= 8) for small M.
- Tail split-K: CTAs [0, n_dp) own full tiles; the remaining partial-wave tiles are split along K over
  SPLIT CTAs. Partials go to an fp32 workspace; the last-arriving CTA sums them in a fixed order
  (deterministic) and resets its counter, so CUDA-graph replay is safe.
"""

from typing import Dict, Optional

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


@gluon.constexpr_function
def _instr_n(BLOCK_M, BLOCK_N, num_warps):
    m_reps = triton.cdiv(BLOCK_M, 16)
    n_reps = triton.cdiv(num_warps, m_reps)
    max_n = max(BLOCK_N // n_reps, 8)
    n = 256
    while n > max_n or BLOCK_N % n != 0:
        n -= 8
    return n


@gluon.constexpr_function
def _warps(BLOCK_M, num_warps):
    w = [4, 1]
    while w[0] * w[1] != num_warps:
        if BLOCK_M > 16 * w[0]:
            w[0] *= 2
        else:
            w[1] *= 2
    return w


@gluon.constexpr_function
def _mma_layout(BLOCK_M, BLOCK_N, num_warps):
    return gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=_warps(BLOCK_M, num_warps),
                                     instr_shape=[16, _instr_n(BLOCK_M, BLOCK_N, num_warps), 32])


@gluon.constexpr_function
def _pick(cond, a, b):
    return a if cond else b


@gluon.jit
def _mma(x_bufs, w_bufs, s, zero, SWAP_AB: gl.constexpr):
    if SWAP_AB:
        t = warpgroup_mma(w_bufs.index(s), x_bufs.index(s).permute((1, 0)), zero, use_acc=False, is_async=True)
    else:
        t = warpgroup_mma(x_bufs.index(s), w_bufs.index(s).permute((1, 0)), zero, use_acc=False, is_async=True)
    return t


@gluon.jit
def _promote(acc, d, sc, SWAP_AB: gl.constexpr):
    if SWAP_AB:
        return acc + d * sc[None, :]
    else:
        return acc + d * sc[:, None]


@gluon.jit
def _scale_column(buf, index, BT: gl.constexpr, target_layout: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1])
    indices = gl.full((BT, 1), 0, gl.int32, layout) + index
    column = buf.gather(indices, axis=1).reshape((BT,))
    return gl.convert_layout(column, target_layout)


@gluon.jit
def _block_fp8_gemm_gluon_h20_kernel(
    x_desc, w_desc, as_desc, C, As, Bs, WS, CNT, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, stride_cm,
    stride_As_m, stride_As_k, stride_Bs_n, stride_Bs_k, n_dp,
    SPLIT: gl.constexpr, GROUP_SIZE_M: gl.constexpr, NUM_STAGES: gl.constexpr,
    SWAP_AB: gl.constexpr, MODE: gl.constexpr, PREFETCH: gl.constexpr, UNROLL: gl.constexpr, num_warps: gl.constexpr,
):
    BLOCK_T: gl.constexpr = x_desc.block_type.shape[0]
    BLOCK_K: gl.constexpr = x_desc.block_type.shape[1]
    BLOCK_W: gl.constexpr = w_desc.block_type.shape[0]
    MMA_M: gl.constexpr = _pick(SWAP_AB, BLOCK_W, BLOCK_T)
    MMA_N: gl.constexpr = _pick(SWAP_AB, BLOCK_T, BLOCK_W)
    dtype: gl.constexpr = x_desc.dtype

    pid = gl.program_id(0)
    k_tiles = K // BLOCK_K
    if SPLIT == 1:
        is_tail = False
        q = 0
        tile = pid
        split = 0
        k_per = K // BLOCK_K
        k_lo = 0
    else:
        is_tail = pid >= n_dp
        q = pid - n_dp
        tile = gl.where(is_tail, n_dp + q // SPLIT, pid)
        split = gl.where(is_tail, q % SPLIT, 0)
        k_per = gl.where(is_tail, k_tiles // SPLIT, k_tiles)
        k_lo = split * k_per

    num_pid_t = gl.cdiv(M, BLOCK_T)
    num_pid_w = gl.cdiv(N, BLOCK_W)
    num_pid_in_group = GROUP_SIZE_M * num_pid_w
    group_id = tile // num_pid_in_group
    first_pid_t = group_id * GROUP_SIZE_M
    group_size = min(num_pid_t - first_pid_t, GROUP_SIZE_M)
    pid_t = first_pid_t + (tile % group_size)
    pid_w = (tile % num_pid_in_group) // group_size
    off_t = pid_t * BLOCK_T
    off_w = pid_w * BLOCK_W

    x_bufs = gl.allocate_shared_memory(dtype, [NUM_STAGES] + x_desc.block_type.shape, x_desc.layout)
    w_bufs = gl.allocate_shared_memory(dtype, [NUM_STAGES] + w_desc.block_type.shape, w_desc.layout)
    ready = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(NUM_STAGES):
        mbarrier.init(ready.index(i), count=1)
    nbytes: gl.constexpr = x_desc.block_type.nbytes + w_desc.block_type.nbytes

    mma_layout: gl.constexpr = _mma_layout(MMA_M, MMA_N, num_warps)
    tok_layout: gl.constexpr = gl.SliceLayout(_pick(SWAP_AB, 0, 1), mma_layout)
    wrow_layout: gl.constexpr = gl.SliceLayout(_pick(SWAP_AB, 1, 0), mma_layout)
    toks = off_t + gl.arange(0, BLOCK_T, layout=tok_layout)
    wrows = off_w + gl.arange(0, BLOCK_W, layout=wrow_layout)
    as_ptrs = As + gl.minimum(toks, M - 1) * stride_As_m + k_lo * stride_As_k
    bs_ptr = Bs + (off_w // 128) * stride_Bs_n + k_lo * stride_Bs_k
    zero = gl.zeros((MMA_M, MMA_N), dtype=gl.float32, layout=mma_layout)
    acc = zero
    scales_smem = gl.allocate_shared_memory(gl.float32, as_desc.block_type.shape, as_desc.layout)
    scales_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(scales_bar, count=1)
    mbarrier.expect(scales_bar, as_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(as_desc, [off_t, 0], scales_bar, scales_smem)
    mbarrier.wait(scales_bar, 0)

    for s in gl.static_range(NUM_STAGES):
        if s < k_per:
            mbarrier.expect(ready.index(s), nbytes)
            tma.async_copy_global_to_shared(x_desc, [off_t, (k_lo + s) * BLOCK_K], ready.index(s), x_bufs.index(s))
            tma.async_copy_global_to_shared(w_desc, [off_w, (k_lo + s) * BLOCK_K], ready.index(s), w_bufs.index(s))

    if MODE == 1 and UNROLL:
        for i in gl.static_range(0, K // BLOCK_K, 2):
            s0 = i % NUM_STAGES
            s1 = (i + 1) % NUM_STAGES
            if PREFETCH:
                sc0 = _scale_column(scales_smem, k_lo + i, BLOCK_T, tok_layout) * gl.load(bs_ptr + i * stride_Bs_k)
                sc1 = _scale_column(scales_smem, k_lo + i + 1, BLOCK_T, tok_layout) * gl.load(bs_ptr + (i + 1) * stride_Bs_k)
            mbarrier.wait(ready.index(s0), (i // NUM_STAGES) & 1)
            t0 = _mma(x_bufs, w_bufs, s0, zero, SWAP_AB)
            mbarrier.wait(ready.index(s1), ((i + 1) // NUM_STAGES) & 1)
            t1 = _mma(x_bufs, w_bufs, s1, zero, SWAP_AB)
            if not PREFETCH:
                sc0 = _scale_column(scales_smem, k_lo + i, BLOCK_T, tok_layout) * gl.load(bs_ptr + i * stride_Bs_k)
                sc1 = _scale_column(scales_smem, k_lo + i + 1, BLOCK_T, tok_layout) * gl.load(bs_ptr + (i + 1) * stride_Bs_k)
            d0, _ = warpgroup_mma_wait(num_outstanding=1, deps=(t0, t1))
            acc = _promote(acc, d0, sc0, SWAP_AB)
            if i + NUM_STAGES < k_per:
                mbarrier.expect(ready.index(s0), nbytes)
                tma.async_copy_global_to_shared(x_desc, [off_t, (k_lo + i + NUM_STAGES) * BLOCK_K], ready.index(s0), x_bufs.index(s0))
                tma.async_copy_global_to_shared(w_desc, [off_w, (k_lo + i + NUM_STAGES) * BLOCK_K], ready.index(s0), w_bufs.index(s0))
            d1 = warpgroup_mma_wait(num_outstanding=0, deps=(t1, ))
            acc = _promote(acc, d1, sc1, SWAP_AB)
            if i + 1 + NUM_STAGES < k_per:
                mbarrier.expect(ready.index(s1), nbytes)
                tma.async_copy_global_to_shared(x_desc, [off_t, (k_lo + i + 1 + NUM_STAGES) * BLOCK_K], ready.index(s1), x_bufs.index(s1))
                tma.async_copy_global_to_shared(w_desc, [off_w, (k_lo + i + 1 + NUM_STAGES) * BLOCK_K], ready.index(s1), w_bufs.index(s1))
    elif MODE == 1:
        for i in range(0, k_per, 2):
            s0 = i % NUM_STAGES
            s1 = (i + 1) % NUM_STAGES
            if PREFETCH:
                sc0 = _scale_column(scales_smem, k_lo + i, BLOCK_T, tok_layout) * gl.load(bs_ptr + i * stride_Bs_k)
                sc1 = _scale_column(scales_smem, k_lo + i + 1, BLOCK_T, tok_layout) * gl.load(bs_ptr + (i + 1) * stride_Bs_k)
            mbarrier.wait(ready.index(s0), (i // NUM_STAGES) & 1)
            t0 = _mma(x_bufs, w_bufs, s0, zero, SWAP_AB)
            mbarrier.wait(ready.index(s1), ((i + 1) // NUM_STAGES) & 1)
            t1 = _mma(x_bufs, w_bufs, s1, zero, SWAP_AB)
            if not PREFETCH:
                sc0 = _scale_column(scales_smem, k_lo + i, BLOCK_T, tok_layout) * gl.load(bs_ptr + i * stride_Bs_k)
                sc1 = _scale_column(scales_smem, k_lo + i + 1, BLOCK_T, tok_layout) * gl.load(bs_ptr + (i + 1) * stride_Bs_k)
            d0, _ = warpgroup_mma_wait(num_outstanding=1, deps=(t0, t1))
            acc = _promote(acc, d0, sc0, SWAP_AB)
            if i + NUM_STAGES < k_per:
                mbarrier.expect(ready.index(s0), nbytes)
                tma.async_copy_global_to_shared(x_desc, [off_t, (k_lo + i + NUM_STAGES) * BLOCK_K], ready.index(s0), x_bufs.index(s0))
                tma.async_copy_global_to_shared(w_desc, [off_w, (k_lo + i + NUM_STAGES) * BLOCK_K], ready.index(s0), w_bufs.index(s0))
            d1 = warpgroup_mma_wait(num_outstanding=0, deps=(t1, ))
            acc = _promote(acc, d1, sc1, SWAP_AB)
            if i + 1 + NUM_STAGES < k_per:
                mbarrier.expect(ready.index(s1), nbytes)
                tma.async_copy_global_to_shared(x_desc, [off_t, (k_lo + i + 1 + NUM_STAGES) * BLOCK_K], ready.index(s1), x_bufs.index(s1))
                tma.async_copy_global_to_shared(w_desc, [off_w, (k_lo + i + 1 + NUM_STAGES) * BLOCK_K], ready.index(s1), w_bufs.index(s1))
    elif MODE == 3:
        for i in range(0, k_per, 4):
            s0 = (i + 0) % NUM_STAGES
            s1 = (i + 1) % NUM_STAGES
            s2 = (i + 2) % NUM_STAGES
            s3 = (i + 3) % NUM_STAGES
            sc0 = _scale_column(scales_smem, k_lo + i + 0, BLOCK_T, tok_layout) * gl.load(bs_ptr + (i + 0) * stride_Bs_k)
            sc1 = _scale_column(scales_smem, k_lo + i + 1, BLOCK_T, tok_layout) * gl.load(bs_ptr + (i + 1) * stride_Bs_k)
            sc2 = _scale_column(scales_smem, k_lo + i + 2, BLOCK_T, tok_layout) * gl.load(bs_ptr + (i + 2) * stride_Bs_k)
            sc3 = _scale_column(scales_smem, k_lo + i + 3, BLOCK_T, tok_layout) * gl.load(bs_ptr + (i + 3) * stride_Bs_k)
            mbarrier.wait(ready.index(s0), ((i + 0) // NUM_STAGES) & 1)
            t0 = _mma(x_bufs, w_bufs, s0, zero, SWAP_AB)
            mbarrier.wait(ready.index(s1), ((i + 1) // NUM_STAGES) & 1)
            t1 = _mma(x_bufs, w_bufs, s1, zero, SWAP_AB)
            mbarrier.wait(ready.index(s2), ((i + 2) // NUM_STAGES) & 1)
            t2 = _mma(x_bufs, w_bufs, s2, zero, SWAP_AB)
            mbarrier.wait(ready.index(s3), ((i + 3) // NUM_STAGES) & 1)
            t3 = _mma(x_bufs, w_bufs, s3, zero, SWAP_AB)
            d0, _, _, _ = warpgroup_mma_wait(num_outstanding=3, deps=(t0, t1, t2, t3,))
            acc = _promote(acc, d0, sc0, SWAP_AB)
            if i + 0 + NUM_STAGES < k_per:
                mbarrier.expect(ready.index(s0), nbytes)
                tma.async_copy_global_to_shared(x_desc, [off_t, (k_lo + i + 0 + NUM_STAGES) * BLOCK_K], ready.index(s0), x_bufs.index(s0))
                tma.async_copy_global_to_shared(w_desc, [off_w, (k_lo + i + 0 + NUM_STAGES) * BLOCK_K], ready.index(s0), w_bufs.index(s0))
            d1, _, _ = warpgroup_mma_wait(num_outstanding=2, deps=(t1, t2, t3,))
            acc = _promote(acc, d1, sc1, SWAP_AB)
            if i + 1 + NUM_STAGES < k_per:
                mbarrier.expect(ready.index(s1), nbytes)
                tma.async_copy_global_to_shared(x_desc, [off_t, (k_lo + i + 1 + NUM_STAGES) * BLOCK_K], ready.index(s1), x_bufs.index(s1))
                tma.async_copy_global_to_shared(w_desc, [off_w, (k_lo + i + 1 + NUM_STAGES) * BLOCK_K], ready.index(s1), w_bufs.index(s1))
            d2, _ = warpgroup_mma_wait(num_outstanding=1, deps=(t2, t3,))
            acc = _promote(acc, d2, sc2, SWAP_AB)
            if i + 2 + NUM_STAGES < k_per:
                mbarrier.expect(ready.index(s2), nbytes)
                tma.async_copy_global_to_shared(x_desc, [off_t, (k_lo + i + 2 + NUM_STAGES) * BLOCK_K], ready.index(s2), x_bufs.index(s2))
                tma.async_copy_global_to_shared(w_desc, [off_w, (k_lo + i + 2 + NUM_STAGES) * BLOCK_K], ready.index(s2), w_bufs.index(s2))
            d3 = warpgroup_mma_wait(num_outstanding=0, deps=(t3,))
            acc = _promote(acc, d3, sc3, SWAP_AB)
            if i + 3 + NUM_STAGES < k_per:
                mbarrier.expect(ready.index(s3), nbytes)
                tma.async_copy_global_to_shared(x_desc, [off_t, (k_lo + i + 3 + NUM_STAGES) * BLOCK_K], ready.index(s3), x_bufs.index(s3))
                tma.async_copy_global_to_shared(w_desc, [off_w, (k_lo + i + 3 + NUM_STAGES) * BLOCK_K], ready.index(s3), w_bufs.index(s3))
    else:
        for i in range(0, k_per):
            s0 = i % NUM_STAGES
            mbarrier.wait(ready.index(s0), (i // NUM_STAGES) & 1)
            t0 = _mma(x_bufs, w_bufs, s0, zero, SWAP_AB)
            sc0 = _scale_column(scales_smem, k_lo + i, BLOCK_T, tok_layout) * gl.load(bs_ptr + i * stride_Bs_k)
            d0 = warpgroup_mma_wait(num_outstanding=0, deps=(t0, ))
            acc = _promote(acc, d0, sc0, SWAP_AB)
            if i + NUM_STAGES < k_per:
                mbarrier.expect(ready.index(s0), nbytes)
                tma.async_copy_global_to_shared(x_desc, [off_t, (k_lo + i + NUM_STAGES) * BLOCK_K], ready.index(s0), x_bufs.index(s0))
                tma.async_copy_global_to_shared(w_desc, [off_w, (k_lo + i + NUM_STAGES) * BLOCK_K], ready.index(s0), w_bufs.index(s0))
    for i in gl.static_range(NUM_STAGES):
        mbarrier.invalidate(ready.index(i))

    mbarrier.invalidate(scales_bar)
    if SWAP_AB:
        c_ptrs = C + toks[None, :] * stride_cm + wrows[:, None]
        mask = (toks[None, :] < M) & (wrows[:, None] < N)
    else:
        c_ptrs = C + toks[:, None] * stride_cm + wrows[None, :]
        mask = (toks[:, None] < M) & (wrows[None, :] < N)
    if is_tail:
        r = gl.arange(0, MMA_M, layout=gl.SliceLayout(1, mma_layout))
        cc = gl.arange(0, MMA_N, layout=gl.SliceLayout(0, mma_layout))
        off = r[:, None] * MMA_N + cc[None, :]
        t_local = (q // SPLIT).to(gl.int64)
        ws_tile = WS + t_local * (SPLIT * MMA_M * MMA_N)
        gl.store(ws_tile + split * (MMA_M * MMA_N) + off, acc, cache_modifier=".cg")
        gl.barrier()
        cnt = gl.atomic_add(CNT + t_local, 1, sem="acq_rel", scope="gpu")
        if cnt == SPLIT - 1:
            gl.barrier()
            tot = gl.zeros((MMA_M, MMA_N), dtype=gl.float32, layout=mma_layout)
            for s in gl.static_range(SPLIT):
                tot += gl.load(ws_tile + s * (MMA_M * MMA_N) + off, cache_modifier=".cg")
            gl.store(c_ptrs, tot.to(gl.bfloat16), mask=mask)
            gl.atomic_xchg(CNT + t_local, 0, sem="relaxed", scope="gpu")
    else:
        gl.store(c_ptrs, acc.to(gl.bfloat16), mask=mask)


DEFAULT_CONFIG = dict(
    BLOCK_T=64, BLOCK_W=64, GROUP_SIZE_M=8, num_warps=4, NUM_STAGES=4, SWAP_AB=False, SPLIT=1, MODE=1
)

_NUM_SMS: Dict[int, int] = {}


def _num_sms(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _NUM_SMS.get(idx)
    if n is None:
        n = _NUM_SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return n


# Split-K workspace + tile counters, keyed by (device, stream): kernels on different streams never
# share a buffer. Counters self-reset after each fixup, so CUDA-graph replay is safe. Within one
# stream launches are serialized, so reuse across calls is safe. Growing allocates on the current
# stream (the caching allocator ties the old block to that stream, so in-flight kernels keep it).
_BUFS: Dict = {}


def _bufs(dev: torch.device, ws_elems: int, n_cnt: int):
    key = (dev, torch.cuda.current_stream(dev).cuda_stream)
    ws, cnt = _BUFS.get(key, (None, None))
    if ws is None or ws.numel() < ws_elems:
        ws = torch.empty(max(ws_elems, 1 << 20), device=dev, dtype=torch.float32)
    if cnt is None or cnt.numel() < n_cnt:
        cnt = torch.zeros(max(n_cnt, 4096), device=dev, dtype=torch.int32)
    _BUFS[key] = (ws, cnt)
    return ws, cnt


_LAYOUTS: Dict = {}


def _desc(t: torch.Tensor, block):
    # No per-tensor cache: a TensorDescriptor references its base tensor, so caching by data_ptr
    # would pin activation buffers in a long-running server. Only the shared layout is cached.
    key = (tuple(block), t.dtype)
    lay = _LAYOUTS.get(key)
    if lay is None:
        dtype = gl.float32 if t.dtype == torch.float32 else gl.float8e4nv
        lay = _LAYOUTS[key] = gl.NVMMASharedLayout.get_default_for(list(block), dtype)
    return TensorDescriptor.from_tensor(t, list(block), lay)


def block_fp8_gemm_scale_tma(
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
    if As.stride(1) != 1:
        from .kernel_v2 import block_fp8_gemm_v2
        return block_fp8_gemm_v2(A, B, As, Bs, config, block_size)
    c = dict(DEFAULT_CONFIG, **(config or {}))
    BT, BW, BK = c["BLOCK_T"], c["BLOCK_W"], block_size[1]
    assert K % BK == 0 and block_size[0] % BW == 0
    k_tiles = K // BK
    split = c["SPLIT"]
    if k_tiles % split:
        split = 1
    tiles = triton.cdiv(M, BT) * triton.cdiv(N, BW)
    # tail = last partial wave; only split it when it would leave SMs idle
    tail = tiles % _num_sms(A.device) if split > 1 else 0
    mode = c.get("MODE", 1)
    unroll = c.get("UNROLL", False)
    assert not unroll or split == 1
    assert mode != 3 or (c["NUM_STAGES"] >= 4 and k_tiles % (4 * split) == 0)
    if mode == 1 and ((k_tiles % 2) or (tail and (k_tiles // split) % 2)):
        mode = 2  # pairwise loop needs an even K-step count
    n_dp = tiles - tail
    mma_m, mma_n = (BW, BT) if c["SWAP_AB"] else (BT, BW)
    ws, cnt = _bufs(A.device, tail * split * mma_m * mma_n, max(tail, 1))
    C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
    _block_fp8_gemm_gluon_h20_kernel[(n_dp + tail * split,)](
        _desc(A, (BT, BK)), _desc(B, (BW, BK)), _desc(As, (BT, triton.next_power_of_2(K // BK))), C, As, Bs, ws, cnt, M, N, K, N,
        As.stride(0), As.stride(1), Bs.stride(0), Bs.stride(1), n_dp,
        SPLIT=split, GROUP_SIZE_M=c["GROUP_SIZE_M"], NUM_STAGES=c["NUM_STAGES"],
        SWAP_AB=c["SWAP_AB"], MODE=mode, PREFETCH=c.get("PREFETCH", True), UNROLL=unroll, num_warps=c["num_warps"],
    )
    return C
