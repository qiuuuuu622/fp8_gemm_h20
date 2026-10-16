# Candidate c12 (parent c11): warp-specialized producer/consumer (Gluon gl.warp_specialize).
#  - worker partition (1 warp): TMA producer, waits empty[s] -> expect/load into slot s -> ready[s]
#  - default partition (num_warps): pairwise async wgmma (c11 scheme) + scale promotion, arrives empty[s]
#    once the wgmma reading slot s has retired; epilogue + tail split-K fixup as in c9/c11.
#  Producer never stalls on MMA completion, so the ring can run NUM_STAGES deep at low CTA count.
import json, os
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, warpgroup_mma_wait
from gl1 import _mma_layout
from gl2 import _pick, _desc, _bufs, NUM_SMS
from gl5 import _mma, _promote


@gluon.jit
def _tile(pid, M, N, K, n_dp, BLOCK_T: gl.constexpr, BLOCK_W: gl.constexpr, BLOCK_K: gl.constexpr,
          SPLIT: gl.constexpr, GROUP_SIZE_M: gl.constexpr):
    k_tiles = K // BLOCK_K
    is_tail = pid >= n_dp
    q = pid - n_dp
    tile = gl.where(is_tail, n_dp + q // SPLIT, pid)
    split = gl.where(is_tail, q % SPLIT, 0)
    k_per = gl.where(is_tail, k_tiles // SPLIT, k_tiles)
    num_pid_t = gl.cdiv(M, BLOCK_T)
    num_pid_w = gl.cdiv(N, BLOCK_W)
    num_pid_in_group = GROUP_SIZE_M * num_pid_w
    group_id = tile // num_pid_in_group
    first_pid_t = group_id * GROUP_SIZE_M
    group_size = min(num_pid_t - first_pid_t, GROUP_SIZE_M)
    pid_t = first_pid_t + (tile % group_size)
    pid_w = (tile % num_pid_in_group) // group_size
    return pid_t * BLOCK_T, pid_w * BLOCK_W, split * k_per, k_per, is_tail, q, split


@gluon.jit
def _producer(x_desc, w_desc, x_bufs, w_bufs, ready, empty, M, N, K, n_dp,
              SPLIT: gl.constexpr, GROUP_SIZE_M: gl.constexpr, NUM_STAGES: gl.constexpr):
    BLOCK_T: gl.constexpr = x_desc.block_type.shape[0]
    BLOCK_K: gl.constexpr = x_desc.block_type.shape[1]
    BLOCK_W: gl.constexpr = w_desc.block_type.shape[0]
    off_t, off_w, k_lo, k_per, is_tail, q, split = _tile(gl.program_id(0), M, N, K, n_dp, BLOCK_T, BLOCK_W,
                                                         BLOCK_K, SPLIT, GROUP_SIZE_M)
    nbytes: gl.constexpr = x_desc.block_type.nbytes + w_desc.block_type.nbytes
    for i in range(k_per):
        s = i % NUM_STAGES
        # empty barriers start "phase 1 complete" -> first pass through the ring does not block
        mbarrier.wait(empty.index(s), ((i // NUM_STAGES) & 1) ^ 1)
        mbarrier.expect(ready.index(s), nbytes)
        tma.async_copy_global_to_shared(x_desc, [off_t, (k_lo + i) * BLOCK_K], ready.index(s), x_bufs.index(s))
        tma.async_copy_global_to_shared(w_desc, [off_w, (k_lo + i) * BLOCK_K], ready.index(s), w_bufs.index(s))


@gluon.jit
def _consumer(x_bufs, w_bufs, ready, empty, C, As, Bs, WS, CNT, M, N, K, n_dp, stride_cm,
              stride_As_m, stride_As_k, stride_Bs_n, stride_Bs_k,
              BLOCK_T: gl.constexpr, BLOCK_W: gl.constexpr, BLOCK_K: gl.constexpr,
              SPLIT: gl.constexpr, GROUP_SIZE_M: gl.constexpr, NUM_STAGES: gl.constexpr,
              SWAP_AB: gl.constexpr, num_warps: gl.constexpr):
    MMA_M: gl.constexpr = _pick(SWAP_AB, BLOCK_W, BLOCK_T)
    MMA_N: gl.constexpr = _pick(SWAP_AB, BLOCK_T, BLOCK_W)
    off_t, off_w, k_lo, k_per, is_tail, q, split = _tile(gl.program_id(0), M, N, K, n_dp, BLOCK_T, BLOCK_W,
                                                         BLOCK_K, SPLIT, GROUP_SIZE_M)
    mma_layout: gl.constexpr = _mma_layout(MMA_M, MMA_N, num_warps)
    tok_layout: gl.constexpr = gl.SliceLayout(_pick(SWAP_AB, 0, 1), mma_layout)
    wrow_layout: gl.constexpr = gl.SliceLayout(_pick(SWAP_AB, 1, 0), mma_layout)
    toks = off_t + gl.arange(0, BLOCK_T, layout=tok_layout)
    wrows = off_w + gl.arange(0, BLOCK_W, layout=wrow_layout)
    as_ptrs = As + gl.minimum(toks, M - 1) * stride_As_m + k_lo * stride_As_k
    bs_ptr = Bs + (off_w // 128) * stride_Bs_n + k_lo * stride_Bs_k
    zero = gl.zeros((MMA_M, MMA_N), dtype=gl.float32, layout=mma_layout)
    acc = zero

    n_pair = k_per // 2
    for p in range(n_pair):
        i = 2 * p
        s0 = i % NUM_STAGES
        s1 = (i + 1) % NUM_STAGES
        mbarrier.wait(ready.index(s0), (i // NUM_STAGES) & 1)
        t0 = _mma(x_bufs, w_bufs, s0, zero, SWAP_AB)
        mbarrier.wait(ready.index(s1), ((i + 1) // NUM_STAGES) & 1)
        t1 = _mma(x_bufs, w_bufs, s1, zero, SWAP_AB)
        sc0 = gl.load(as_ptrs + i * stride_As_k) * gl.load(bs_ptr + i * stride_Bs_k)
        sc1 = gl.load(as_ptrs + (i + 1) * stride_As_k) * gl.load(bs_ptr + (i + 1) * stride_Bs_k)
        d0, _ = warpgroup_mma_wait(num_outstanding=1, deps=(t0, t1))
        mbarrier.arrive(empty.index(s0))
        acc = _promote(acc, d0, sc0, SWAP_AB)
        d1 = warpgroup_mma_wait(num_outstanding=0, deps=(t1, ))
        mbarrier.arrive(empty.index(s1))
        acc = _promote(acc, d1, sc1, SWAP_AB)
    if k_per % 2 == 1:
        i = k_per - 1
        s0 = i % NUM_STAGES
        mbarrier.wait(ready.index(s0), (i // NUM_STAGES) & 1)
        t0 = _mma(x_bufs, w_bufs, s0, zero, SWAP_AB)
        sc0 = gl.load(as_ptrs + i * stride_As_k) * gl.load(bs_ptr + i * stride_Bs_k)
        d0 = warpgroup_mma_wait(num_outstanding=0, deps=(t0, ))
        mbarrier.arrive(empty.index(s0))
        acc = _promote(acc, d0, sc0, SWAP_AB)

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


@gluon.jit
def _w8a8_block_fp8_matmul_hopper_gl6(
    x_desc, w_desc, C, As, Bs, WS, CNT, M, N, K, stride_cm,
    stride_As_m, stride_As_k, stride_Bs_n, stride_Bs_k, n_dp,
    SPLIT: gl.constexpr, GROUP_SIZE_M: gl.constexpr, NUM_STAGES: gl.constexpr,
    SWAP_AB: gl.constexpr, PRODUCER_REGS: gl.constexpr, num_warps: gl.constexpr,
):
    BLOCK_T: gl.constexpr = x_desc.block_type.shape[0]
    BLOCK_K: gl.constexpr = x_desc.block_type.shape[1]
    BLOCK_W: gl.constexpr = w_desc.block_type.shape[0]
    dtype: gl.constexpr = x_desc.dtype
    x_bufs = gl.allocate_shared_memory(dtype, [NUM_STAGES] + x_desc.block_type.shape, x_desc.layout)
    w_bufs = gl.allocate_shared_memory(dtype, [NUM_STAGES] + w_desc.block_type.shape, w_desc.layout)
    ready = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(NUM_STAGES):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(empty.index(i), count=1)
    gl.warp_specialize([
        (_consumer, (x_bufs, w_bufs, ready, empty, C, As, Bs, WS, CNT, M, N, K, n_dp, stride_cm,
                     stride_As_m, stride_As_k, stride_Bs_n, stride_Bs_k,
                     BLOCK_T, BLOCK_W, BLOCK_K, SPLIT, GROUP_SIZE_M, NUM_STAGES, SWAP_AB, num_warps)),
        (_producer, (x_desc, w_desc, x_bufs, w_bufs, ready, empty, M, N, K, n_dp,
                     SPLIT, GROUP_SIZE_M, NUM_STAGES)),
    ], [1], [PRODUCER_REGS])
    for i in gl.static_range(NUM_STAGES):
        mbarrier.invalidate(ready.index(i))
        mbarrier.invalidate(empty.index(i))


DEFAULT = dict(BLOCK_T=64, BLOCK_W=64, GROUP_SIZE_M=8, num_warps=4, NUM_STAGES=4, SWAP_AB=False, SPLIT=1)
_CFG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "gl6_configs.json")
_CFGS = json.load(open(_CFG_FILE)) if os.path.exists(_CFG_FILE) else {}


def pick_cfg(M, N, K):
    t = _CFGS.get(f"{N},{K}")
    if not t:
        return DEFAULT
    return t[min(t, key=lambda m: abs(int(m) - M))]


def matmul(A, B, As, Bs, cfg=None, block=(128, 128)):
    """A[M,K] fp8, B[N,K] fp8, As[M,K/128] f32, Bs[N/128,K/128] f32 -> C[M,N] bf16."""
    M, K = A.shape
    N = B.shape[0]
    c = dict(DEFAULT, **(cfg or pick_cfg(M, N, K)))
    BT, BW, BK = c["BLOCK_T"], c["BLOCK_W"], block[1]
    assert K % BK == 0 and block[0] % BW == 0
    k_tiles = K // BK
    split = c["SPLIT"]
    if k_tiles % split:
        split = 1
    tiles = triton.cdiv(M, BT) * triton.cdiv(N, BW)
    tail = tiles % NUM_SMS if split > 1 else 0
    n_dp = tiles - tail
    mma_m, mma_n = (BW, BT) if c["SWAP_AB"] else (BT, BW)
    ws, cnt = _bufs(A.device, tail * split * mma_m * mma_n, max(tail, 1))
    C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
    _w8a8_block_fp8_matmul_hopper_gl6[(n_dp + tail * split,)](
        _desc(A, (BT, BK)), _desc(B, (BW, BK)), C, As, Bs, ws, cnt, M, N, K, N,
        As.stride(0), As.stride(1), Bs.stride(0), Bs.stride(1), n_dp,
        SPLIT=split, GROUP_SIZE_M=c["GROUP_SIZE_M"], NUM_STAGES=c["NUM_STAGES"],
        SWAP_AB=c["SWAP_AB"], PRODUCER_REGS=c.get("PRODUCER_REGS", 40), num_warps=c["num_warps"],
    )
    return C
