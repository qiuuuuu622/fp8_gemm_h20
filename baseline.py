# Baseline: _w8a8_block_fp8_matmul_hopper as provided by the user (verbatim kernel body).
import torch
import triton
import triton.language as tl


@triton.jit
def _w8a8_block_fp8_matmul_hopper(
    A, B, C, As, Bs, M, N, K, group_n, group_k,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    stride_As_m, stride_As_k, stride_Bs_k, stride_Bs_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, needs_masking: tl.constexpr,
    SWAP_AB: tl.constexpr = False, SPLIT_K: tl.constexpr = 1,
):
    pid = tl.program_id(axis=0)
    split = tl.program_id(axis=1)
    tiles_per_split = tl.cdiv(tl.cdiv(K, BLOCK_SIZE_K), SPLIT_K)
    first_tile = split * tiles_per_split
    C += split * M * N
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    a_ptrs += first_tile * BLOCK_SIZE_K * stride_ak
    b_ptrs += first_tile * BLOCK_SIZE_K * stride_bk
    As_ptrs += (first_tile // n_tiles_k_per_group_k) * stride_As_k
    Bs_ptrs += (first_tile // n_tiles_k_per_group_k) * stride_Bs_k

    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(
        first_tile, tl.minimum(first_tile + tiles_per_split, tl.cdiv(K, BLOCK_SIZE_K))
    ):
        if needs_masking:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)
        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        if SWAP_AB:
            accumulator += tl.dot(tl.trans(b), tl.trans(a)) * b_s[:, None] * a_s[None, :]
        else:
            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

    if SWAP_AB:
        accumulator = tl.trans(accumulator)
    c = accumulator.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def run(A, B, As, Bs, cfg, block=(128, 128)):
    """A[M,K] fp8, B[N,K] fp8, As[M,K/128] f32, Bs[N/128,K/128] f32 -> C[M,N] bf16."""
    M, K = A.shape
    N = B.shape[0]
    cfg = dict(cfg)
    split_k = cfg.pop("SPLIT_K", 1)
    out = torch.empty((split_k, M, N), device=A.device,
                      dtype=torch.bfloat16 if split_k == 1 else torch.float32)
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]), split_k)
    _w8a8_block_fp8_matmul_hopper[grid](
        A, B, out, As, Bs, M, N, K, block[0], block[1],
        A.stride(0), A.stride(1), B.stride(1), B.stride(0), N, 1,
        As.stride(0), As.stride(1), Bs.stride(1), Bs.stride(0),
        needs_masking=K % cfg["BLOCK_SIZE_K"] != 0, SPLIT_K=split_k, **cfg,
    )
    return out[0] if split_k == 1 else out.sum(0).to(torch.bfloat16)
