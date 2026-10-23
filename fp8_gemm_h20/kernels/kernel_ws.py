"""Hopper FP8 GEMM with separate TMA producer and WGMMA consumer warps.

Uses ready/empty mbarrier rings. Scale groups are promoted independently into
FP32. One CTA owns an output tile; no split-K workspace is used.
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma, warpgroup_mma_wait, fence_async_shared
from .kernel_v2 import _desc, _mma_layout, _mma, _promote, _pick


@gluon.jit
def _producer(xd, wd, ab, bb, ready, empty, mt, nt, K: gl.constexpr, STAGES: gl.constexpr):
    for i in range(K // 128):
        s = i % STAGES
        phase = (i // STAGES) & 1
        mbarrier.wait(empty.index(s), phase ^ 1)
        mbarrier.expect(ready.index(s), xd.block_type.nbytes + wd.block_type.nbytes)
        tma.async_copy_global_to_shared(xd, [mt, i * 128], ready.index(s), ab.index(s))
        tma.async_copy_global_to_shared(wd, [nt, i * 128], ready.index(s), bb.index(s))


@gluon.jit
def _consumer(ab, bb, ready, empty, C, SA, SB, mt, nt,
              M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
              SAM: gl.constexpr, SAK: gl.constexpr, SBN: gl.constexpr, SBK: gl.constexpr,
              BT: gl.constexpr, BW: gl.constexpr, SWAP: gl.constexpr, STAGES: gl.constexpr):
    MM: gl.constexpr = _pick(SWAP, BW, BT)
    MN: gl.constexpr = _pick(SWAP, BT, BW)
    layout: gl.constexpr = _mma_layout(MM, MN, 4)
    rm = mt + gl.arange(0, BT, layout=gl.SliceLayout(_pick(SWAP, 0, 1), layout))
    rn = nt + gl.arange(0, BW, layout=gl.SliceLayout(_pick(SWAP, 1, 0), layout))
    sap = SA + gl.minimum(rm, M - 1) * SAM
    sbp = SB + (nt // 128) * SBN
    zero = gl.zeros((MM, MN), gl.float32, layout)
    acc = zero
    for i in range(0, K // 128, 2):
        s0 = i % STAGES
        s1 = (i + 1) % STAGES
        sc0 = gl.load(sap + i * SAK) * gl.load(sbp + i * SBK)
        sc1 = gl.load(sap + (i + 1) * SAK) * gl.load(sbp + (i + 1) * SBK)
        mbarrier.wait(ready.index(s0), (i // STAGES) & 1)
        t0 = _mma(ab, bb, s0, zero, SWAP)
        mbarrier.wait(ready.index(s1), ((i + 1) // STAGES) & 1)
        t1 = _mma(ab, bb, s1, zero, SWAP)
        d0, _ = warpgroup_mma_wait(num_outstanding=1, deps=(t0, t1))
        fence_async_shared()
        mbarrier.arrive(empty.index(s0), count=1)
        acc = _promote(acc, d0, sc0, SWAP)
        d1 = warpgroup_mma_wait(num_outstanding=0, deps=(t1,))
        fence_async_shared()
        mbarrier.arrive(empty.index(s1), count=1)
        acc = _promote(acc, d1, sc1, SWAP)
    if SWAP:
        gl.store(C + rm[None, :] * N + rn[:, None], acc.to(gl.bfloat16), (rm[None, :] < M) & (rn[:, None] < N))
    else:
        gl.store(C + rm[:, None] * N + rn[None, :], acc.to(gl.bfloat16), (rm[:, None] < M) & (rn[None, :] < N))


@gluon.jit
def _ws(xd, wd, C, SA, SB, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
        SAM: gl.constexpr, SAK: gl.constexpr, SBN: gl.constexpr, SBK: gl.constexpr,
        BT: gl.constexpr, BW: gl.constexpr, SWAP: gl.constexpr, STAGES: gl.constexpr, GROUP: gl.constexpr):
    pid = gl.program_id(0)
    nm = gl.cdiv(M, BT)
    nn = gl.cdiv(N, BW)
    group = pid // (GROUP * nn)
    first = group * GROUP
    size = min(nm - first, GROUP)
    mt = (first + pid % size) * BT
    nt = ((pid % (GROUP * nn)) // size) * BW
    ab = gl.allocate_shared_memory(xd.dtype, [STAGES] + xd.block_type.shape, xd.layout)
    bb = gl.allocate_shared_memory(wd.dtype, [STAGES] + wd.block_type.shape, wd.layout)
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    for s in gl.static_range(STAGES):
        mbarrier.init(ready.index(s), count=1)
        mbarrier.init(empty.index(s), count=1)
    gl.warp_specialize([
        (_consumer, (ab, bb, ready, empty, C, SA, SB, mt, nt, M, N, K, SAM, SAK, SBN, SBK, BT, BW, SWAP, STAGES)),
        (_producer, (xd, wd, ab, bb, ready, empty, mt, nt, K, STAGES)),
    ], [1], [24])
    for s in gl.static_range(STAGES):
        mbarrier.invalidate(ready.index(s))
        mbarrier.invalidate(empty.index(s))


def warp_specialized(A, B, As, Bs, config):
    m, k = A.shape
    n = B.shape[0]
    c = dict(config)
    assert k % 256 == 0 and 128 % c['BW'] == 0 and c['STAGES'] >= 2
    out = torch.empty((m, n), device=A.device, dtype=torch.bfloat16)
    _ws[(triton.cdiv(m, c['BT']) * triton.cdiv(n, c['BW']),)](
        _desc(A, (c['BT'], 128)), _desc(B, (c['BW'], 128)), out, As, Bs,
        m, n, k, *As.stride(), *Bs.stride(), num_warps=4, **c)
    return out
