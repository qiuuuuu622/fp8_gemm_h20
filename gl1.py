# Candidate c8 (parent c3): Gluon explicit-async kernel (Triton 3.7 gluon).
#  - TMA ring of NUM_STAGES A/B buffers, mbarrier ready/empty phases
#  - async wgmma with num_outstanding=1: promotion (scale-FMA) of step k overlaps wgmma of step k+1
#  - fp32 promote accumulator kept separate from the wgmma accumulator (DeepGEMM-style)
import json, os
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma, mbarrier, fence_async_shared, warpgroup_mma_init, warpgroup_mma, warpgroup_mma_wait,
)


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


@gluon.jit
def _w8a8_block_fp8_matmul_hopper_gl(
    a_desc, b_desc, C, As, Bs, M, N, K, stride_cm,
    stride_As_m, stride_As_k, stride_Bs_n, stride_Bs_k,
    GROUP_SIZE_M: gl.constexpr, NUM_STAGES: gl.constexpr, num_warps: gl.constexpr,
):
    BLOCK_M: gl.constexpr = a_desc.block_type.shape[0]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    BLOCK_N: gl.constexpr = b_desc.block_type.shape[0]
    dtype: gl.constexpr = a_desc.dtype

    pid = gl.program_id(0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    k_tiles = K // BLOCK_K

    a_bufs = gl.allocate_shared_memory(dtype, [NUM_STAGES] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [NUM_STAGES] + b_desc.block_type.shape, b_desc.layout)
    ready = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(NUM_STAGES):
        mbarrier.init(ready.index(i), count=1)

    mma_layout: gl.constexpr = _mma_layout(BLOCK_M, BLOCK_N, num_warps)
    row_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)
    rows = off_m + gl.arange(0, BLOCK_M, layout=row_layout)
    as_ptrs = As + gl.minimum(rows, M - 1) * stride_As_m
    bs_ptr = Bs + (off_n // 128) * stride_Bs_n
    zero = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
    acc = zero

    # prologue: fill all NUM_STAGES slots (tile s -> slot s)
    for s in gl.static_range(NUM_STAGES):
        if s < k_tiles:
            mbarrier.expect(ready.index(s), a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(a_desc, [off_m, s * BLOCK_K], ready.index(s), a_bufs.index(s))
            tma.async_copy_global_to_shared(b_desc, [off_n, s * BLOCK_K], ready.index(s), b_bufs.index(s))

    # step 0
    mbarrier.wait(ready.index(0), 0)
    tok = warpgroup_mma(a_bufs.index(0), b_bufs.index(0).permute((1, 0)), zero, use_acc=False, is_async=True)
    prev_scale = gl.load(as_ptrs) * gl.load(bs_ptr)
    for k in range(1, k_tiles):
        # after wait(1) below, wgmma k-1 is retired -> its slot (k-1)%S can take tile k-1+S
        kn = k + NUM_STAGES - 1
        s_cur = k % NUM_STAGES
        mbarrier.wait(ready.index(s_cur), (k // NUM_STAGES) & 1)
        # wait for wgmma k-1 to have <=0 outstanding only after issuing k: issue k first
        tok_k = warpgroup_mma(a_bufs.index(s_cur), b_bufs.index(s_cur).permute((1, 0)), zero,
                              use_acc=False, is_async=True)
        d_prev, _ = warpgroup_mma_wait(num_outstanding=1, deps=(tok, tok_k))
        acc = acc + d_prev * prev_scale[:, None]
        if kn < k_tiles:
            s_n = kn % NUM_STAGES
            mbarrier.expect(ready.index(s_n), a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(a_desc, [off_m, kn * BLOCK_K], ready.index(s_n), a_bufs.index(s_n))
            tma.async_copy_global_to_shared(b_desc, [off_n, kn * BLOCK_K], ready.index(s_n), b_bufs.index(s_n))
        prev_scale = gl.load(as_ptrs + k * stride_As_k) * gl.load(bs_ptr + k * stride_Bs_k)
        tok = tok_k
    d_last = warpgroup_mma_wait(num_outstanding=0, deps=(tok, ))
    acc = acc + d_last * prev_scale[:, None]
    for i in gl.static_range(NUM_STAGES):
        mbarrier.invalidate(ready.index(i))

    col_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)
    cols = off_n + gl.arange(0, BLOCK_N, layout=col_layout)
    c_ptrs = C + rows[:, None] * stride_cm + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    gl.store(c_ptrs, acc.to(gl.bfloat16), mask=mask)


DEFAULT = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=4, NUM_STAGES=4)
_CFG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "gl1_configs.json")
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
    BM, BN, BK = c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"], block[1]
    assert K % BK == 0 and block[0] % BN == 0
    a_l = gl.NVMMASharedLayout.get_default_for([BM, BK], gl.float8e4nv)
    b_l = gl.NVMMASharedLayout.get_default_for([BN, BK], gl.float8e4nv)
    a_desc = TensorDescriptor.from_tensor(A, [BM, BK], a_l)
    b_desc = TensorDescriptor.from_tensor(B, [BN, BK], b_l)
    C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _w8a8_block_fp8_matmul_hopper_gl[grid](
        a_desc, b_desc, C, As, Bs, M, N, K, N,
        As.stride(0), As.stride(1), Bs.stride(0), Bs.stride(1),
        GROUP_SIZE_M=c["GROUP_SIZE_M"], NUM_STAGES=c["NUM_STAGES"], num_warps=c["num_warps"],
    )
    return C
