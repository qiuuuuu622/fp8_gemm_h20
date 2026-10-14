# Candidate c9 (parent c8): Gluon async-wgmma kernel + SWAP_AB + data-parallel/split-K-tail scheduling.
#  - CTAs [0, n_dp) own full tiles; the remaining "tail" tiles (< 1 wave) are split along K across
#    SPLIT CTAs each, partials go to an fp32 workspace, last-arriving CTA sums in fixed order (deterministic).
#  - SWAP_AB puts weight rows on the wgmma M side (>=64) and tokens on the N side (>=8) for small M.
import json, os
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma, mbarrier, warpgroup_mma, warpgroup_mma_wait,
)
from gl1 import _mma_layout

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


@gluon.constexpr_function
def _pick(cond, a, b):
    return a if cond else b


@gluon.jit
def _w8a8_block_fp8_matmul_hopper_gl2(
    x_desc, w_desc, C, As, Bs, WS, CNT, M, N, K, stride_cm,
    stride_As_m, stride_As_k, stride_Bs_n, stride_Bs_k, n_dp,
    SPLIT: gl.constexpr, GROUP_SIZE_M: gl.constexpr, NUM_STAGES: gl.constexpr,
    SWAP_AB: gl.constexpr, num_warps: gl.constexpr,
):
    BLOCK_T: gl.constexpr = x_desc.block_type.shape[0]
    BLOCK_K: gl.constexpr = x_desc.block_type.shape[1]
    BLOCK_W: gl.constexpr = w_desc.block_type.shape[0]
    MMA_M: gl.constexpr = _pick(SWAP_AB, BLOCK_W, BLOCK_T)
    MMA_N: gl.constexpr = _pick(SWAP_AB, BLOCK_T, BLOCK_W)
    dtype: gl.constexpr = x_desc.dtype

    pid = gl.program_id(0)
    k_tiles = K // BLOCK_K
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

    for s in gl.static_range(NUM_STAGES):
        if s < k_per:
            mbarrier.expect(ready.index(s), nbytes)
            tma.async_copy_global_to_shared(x_desc, [off_t, (k_lo + s) * BLOCK_K], ready.index(s), x_bufs.index(s))
            tma.async_copy_global_to_shared(w_desc, [off_w, (k_lo + s) * BLOCK_K], ready.index(s), w_bufs.index(s))

    mbarrier.wait(ready.index(0), 0)
    if SWAP_AB:
        tok = warpgroup_mma(w_bufs.index(0), x_bufs.index(0).permute((1, 0)), zero, use_acc=False, is_async=True)
    else:
        tok = warpgroup_mma(x_bufs.index(0), w_bufs.index(0).permute((1, 0)), zero, use_acc=False, is_async=True)
    prev_scale = gl.load(as_ptrs) * gl.load(bs_ptr)
    for i in range(1, k_per):
        s_cur = i % NUM_STAGES
        mbarrier.wait(ready.index(s_cur), (i // NUM_STAGES) & 1)
        if SWAP_AB:
            tok_i = warpgroup_mma(w_bufs.index(s_cur), x_bufs.index(s_cur).permute((1, 0)), zero,
                                  use_acc=False, is_async=True)
        else:
            tok_i = warpgroup_mma(x_bufs.index(s_cur), w_bufs.index(s_cur).permute((1, 0)), zero,
                                  use_acc=False, is_async=True)
        d_prev, _ = warpgroup_mma_wait(num_outstanding=1, deps=(tok, tok_i))
        if SWAP_AB:
            acc = acc + d_prev * prev_scale[None, :]
        else:
            acc = acc + d_prev * prev_scale[:, None]
        # wgmma i-1 retired -> its slot takes tile i-1+NUM_STAGES
        inext = i + NUM_STAGES - 1
        if inext < k_per:
            s_n = inext % NUM_STAGES
            mbarrier.expect(ready.index(s_n), nbytes)
            tma.async_copy_global_to_shared(x_desc, [off_t, (k_lo + inext) * BLOCK_K], ready.index(s_n), x_bufs.index(s_n))
            tma.async_copy_global_to_shared(w_desc, [off_w, (k_lo + inext) * BLOCK_K], ready.index(s_n), w_bufs.index(s_n))
        prev_scale = gl.load(as_ptrs + i * stride_As_k) * gl.load(bs_ptr + i * stride_Bs_k)
        tok = tok_i
    d_last = warpgroup_mma_wait(num_outstanding=0, deps=(tok, ))
    if SWAP_AB:
        acc = acc + d_last * prev_scale[None, :]
    else:
        acc = acc + d_last * prev_scale[:, None]
    for i in gl.static_range(NUM_STAGES):
        mbarrier.invalidate(ready.index(i))

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


DEFAULT = dict(BLOCK_T=64, BLOCK_W=64, GROUP_SIZE_M=8, num_warps=4, NUM_STAGES=4, SWAP_AB=False, SPLIT=1)
_CFG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "gl2_configs.json")
_CFGS = json.load(open(_CFG_FILE)) if os.path.exists(_CFG_FILE) else {}
# Split-K workspace + tile counters, keyed by (device, stream): kernels on different streams never
# share a buffer. Counters self-reset after each fixup, so CUDA-graph replay is safe. Within one stream
# launches are serialized, so reuse across calls is safe. Growing allocates on the current stream
# (caching allocator ties the old block to that stream, so in-flight kernels keep their memory).
_BUFS = {}


def _bufs(dev, ws_elems, n_cnt):
    key = (dev, torch.cuda.current_stream(dev).cuda_stream)
    ws, cnt = _BUFS.get(key, (None, None))
    if ws is None or ws.numel() < ws_elems:
        ws = torch.empty(max(ws_elems, 1 << 20), device=dev, dtype=torch.float32)
    if cnt is None or cnt.numel() < n_cnt:
        cnt = torch.zeros(max(n_cnt, 4096), device=dev, dtype=torch.int32)
    _BUFS[key] = (ws, cnt)
    return ws, cnt


def pick_cfg(M, N, K):
    t = _CFGS.get(f"{N},{K}")
    if not t:
        return DEFAULT
    return t[min(t, key=lambda m: abs(int(m) - M))]


_DESC = {}


def _desc(t, block):
    key = (t.data_ptr(), tuple(t.shape), tuple(block))
    d = _DESC.get(key)
    if d is None:
        lay = gl.NVMMASharedLayout.get_default_for(list(block), gl.float8e4nv)
        d = TensorDescriptor.from_tensor(t, list(block), lay)
        if len(_DESC) > 4096:
            _DESC.clear()
        _DESC[key] = d
    return d


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
    # tail = last partial wave; only split it when it would leave SMs idle
    tail = tiles % NUM_SMS if split > 1 else 0
    n_dp = tiles - tail
    mma_m, mma_n = (BW, BT) if c["SWAP_AB"] else (BT, BW)
    ws, cnt = _bufs(A.device, tail * split * mma_m * mma_n, max(tail, 1))
    C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
    _w8a8_block_fp8_matmul_hopper_gl2[(n_dp + tail * split,)](
        _desc(A, (BT, BK)), _desc(B, (BW, BK)), C, As, Bs, ws, cnt, M, N, K, N,
        As.stride(0), As.stride(1), Bs.stride(0), Bs.stride(1), n_dp,
        SPLIT=split, GROUP_SIZE_M=c["GROUP_SIZE_M"], NUM_STAGES=c["NUM_STAGES"],
        SWAP_AB=c["SWAP_AB"], num_warps=c["num_warps"],
    )
    return C
