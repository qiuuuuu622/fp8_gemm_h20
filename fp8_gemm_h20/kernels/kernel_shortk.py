"""Fixed K=256 block-FP8 kernel: two MMA operations, no runtime K loop.

The two 128-element scale groups remain independent; FP32 promotion precedes
their sum. No reduction workspace or cross-CTA synchronization is needed.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _short(A, B, SA, SB, C, M: tl.constexpr, N: tl.constexpr,
           SAM: tl.constexpr, SAK: tl.constexpr, SBN: tl.constexpr, SBK: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, SWAP: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    group = pid // (GROUP * nn)
    first = group * GROUP
    size = tl.minimum(nm - first, GROUP)
    pm = first + pid % size
    pn = (pid % (GROUP * nn)) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, 128)
    # Masked loads avoid reading other rows for the last, partially filled tile.
    a0 = tl.load(A + rm[:, None] * 256 + rk[None, :], rm[:, None] < M, 0.0)
    a1 = tl.load(A + rm[:, None] * 256 + rk[None, :] + 128, rm[:, None] < M, 0.0)
    b0 = tl.load(B + rn[:, None] * 256 + rk[None, :], rn[:, None] < N, 0.0)
    b1 = tl.load(B + rn[:, None] * 256 + rk[None, :] + 128, rn[:, None] < N, 0.0)
    sa0 = tl.load(SA + rm * SAM, rm < M, 0)
    sa1 = tl.load(SA + rm * SAM + SAK, rm < M, 0)
    sb0 = tl.load(SB + (rn // 128) * SBN, rn < N, 0)
    sb1 = tl.load(SB + (rn // 128) * SBN + SBK, rn < N, 0)
    if SWAP:
        d0 = tl.dot(b0, tl.trans(a0))
        d1 = tl.dot(b1, tl.trans(a1))
        out = tl.trans(d0 * (sb0[:, None] * sa0[None, :]) + d1 * (sb1[:, None] * sa1[None, :]))
    else:
        d0 = tl.dot(a0, tl.trans(b0))
        d1 = tl.dot(a1, tl.trans(b1))
        out = d0 * (sa0[:, None] * sb0[None, :]) + d1 * (sa1[:, None] * sb1[None, :])
    tl.store(C + rm[:, None] * N + rn[None, :], out.to(tl.bfloat16), (rm[:, None] < M) & (rn[None, :] < N))


def shortk(A, B, As, Bs, config):
    m, k = A.shape
    n = B.shape[0]
    assert k == 256 and A.is_contiguous() and B.is_contiguous()
    out = torch.empty((m, n), device=A.device, dtype=torch.bfloat16)
    c = dict(config)
    _short[(triton.cdiv(m, c['BM']) * triton.cdiv(n, c['BN']),)](
        A, B, As, Bs, out, m, n, *As.stride(), *Bs.stride(), **c)
    return out
