# Candidate c3: baseline pointer kernel with
#  - scalar per-tile weight scale (BLOCK_N divides group_n -> one Bs value per K group)
#  - fused scale: acc = dot * (a_s * b_s)[:, None] + acc  (single FMA per element)
#  - one K tile per scale group (BLOCK_K == group_k), so scale ptrs advance every iter
#  - optional SPLIT_K with fp32 atomic-free stacked partials
import json, os
import torch
import triton
import triton.language as tl


@triton.jit
def _w8a8_block_fp8_matmul_hopper_v2(
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


DEFAULT = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=32, num_warps=4, num_stages=3, SPLIT_K=1)
_CFG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "opt2_configs.json")
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
    assert K % block[1] == 0 and A.is_contiguous() and B.is_contiguous()
    c = dict(cfg or pick_cfg(M, N, K))
    sk = c.pop("SPLIT_K")
    if (K // block[1]) % sk:
        sk = 1
    out = torch.empty((sk, M, N), device=A.device, dtype=torch.bfloat16 if sk == 1 else torch.float32)
    grid = (triton.cdiv(M, c["BLOCK_SIZE_M"]) * triton.cdiv(N, c["BLOCK_SIZE_N"]), sk)
    _w8a8_block_fp8_matmul_hopper_v2[grid](
        A, B, out, As, Bs, M, N, K, block[0],
        A.stride(0), B.stride(0), N,
        As.stride(0), As.stride(1), Bs.stride(0), Bs.stride(1),
        BLOCK_SIZE_K=block[1], SPLIT_K=sk,
        SCALAR_BS=block[0] % c["BLOCK_SIZE_N"] == 0, **c,
    )
    return out[0] if sk == 1 else out.sum(0).to(torch.bfloat16)
