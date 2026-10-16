# 2 math warpgroups (num_warps=8, wgmma M=128 via BLOCK_W=128 swap) on small-M losers vs DeepGEMM
import itertools, torch, triton, deep_gemm, gl2
from bench import make, ref
SH = [(2048, 4096, 100), (1024, 2048, 333), (512, 2048, 100), (2304, 2048, 100), (12288, 2048, 256)]
bench = lambda f: triton.testing.do_bench_cudagraph(f, rep=50) * 1e3
for N, K, M in SH:
    A, B, As, Bs = make(M, N, K); r = ref(A, B, As, Bs).float()
    Asd = deep_gemm.get_mn_major_tma_aligned_tensor(As); C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    tdg = bench(lambda: deep_gemm.fp8_gemm_nt((A, Asd), (B, Bs), C))
    res = []
    for bt, bw, nw, st, sp, sw in itertools.product([16, 32, 64], [64, 128], [4, 8], [3, 4, 6], [1, 2, 4], [True, False]):
        if (not sw and bt < 64) or (nw == 8 and (bw if sw else bt) < 128) or (sw and bt > 64):
            continue
        c = dict(BLOCK_T=bt, BLOCK_W=bw, SWAP_AB=sw, SPLIT=sp, NUM_STAGES=st, num_warps=nw, GROUP_SIZE_M=1)
        try:
            o = gl2.matmul(A, B, As, Bs, c).float()
            if ((o - r).abs().max() / r.abs().max()).item() > 1e-2: continue
            res.append((bench(lambda: gl2.matmul(A, B, As, Bs, c)), c))
        except Exception:
            pass
    res.sort(key=lambda x: x[0])
    b8 = min((x for x in res if x[1]["num_warps"] == 8), key=lambda x: x[0], default=(float("nan"), None))
    print(f"N={N} K={K} M={M} dg={tdg:.1f} best={res[0][0]:.1f} ({tdg/res[0][0]:.3f}) {res[0][1]} | best_w8={b8[0]:.1f} ({tdg/b8[0]:.3f})", flush=True)
