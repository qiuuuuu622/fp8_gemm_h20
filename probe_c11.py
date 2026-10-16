# c11 (gl5): pairwise wgmma, no in-flight wgmma across loop back-edge (fix ptxas C7514) vs c9 / DeepGEMM
import torch, triton, deep_gemm
import gl2, gl5
from bench import make, ref

CASES = [
    (2048, 4096, 100, dict(BLOCK_T=16, BLOCK_W=64, SWAP_AB=True, SPLIT=1, NUM_STAGES=3, num_warps=4, GROUP_SIZE_M=1)),
    (1024, 2048, 333, dict(BLOCK_T=16, BLOCK_W=64, SWAP_AB=True, SPLIT=2, NUM_STAGES=3, num_warps=4, GROUP_SIZE_M=1)),
    (512, 2048, 100, dict(BLOCK_T=16, BLOCK_W=64, SWAP_AB=True, SPLIT=4, NUM_STAGES=3, num_warps=4, GROUP_SIZE_M=1)),
    (12288, 2048, 256, dict(BLOCK_T=64, BLOCK_W=128, SWAP_AB=False, SPLIT=1, NUM_STAGES=3, num_warps=4, GROUP_SIZE_M=1)),
    (12288, 2048, 8192, dict(BLOCK_T=64, BLOCK_W=64, SWAP_AB=False, SPLIT=1, NUM_STAGES=3, num_warps=4, GROUP_SIZE_M=1)),
]
bench = lambda f: triton.testing.do_bench_cudagraph(f, rep=50) * 1e3
for N, K, M, c in CASES:
    A, B, As, Bs = make(M, N, K)
    r = ref(A, B, As, Bs).float()
    Asd = deep_gemm.get_mn_major_tma_aligned_tensor(As)
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    tdg = bench(lambda: deep_gemm.fp8_gemm_nt((A, Asd), (B, Bs), C))
    t9 = bench(lambda: gl2.matmul(A, B, As, Bs, c))
    line = f"N={N} K={K} M={M} dg={tdg:.1f} c9={t9:.1f}({tdg / t9:.3f})"
    for mode in (1, 2):
        for st in (3, 4, 6):
            cc = {**c, "MODE": mode, "NUM_STAGES": st}
            o = gl5.matmul(A, B, As, Bs, cc).float()
            e = ((o - r).abs().max() / r.abs().max()).item()
            t = bench(lambda: gl5.matmul(A, B, As, Bs, cc))
            flag = "" if e < 1e-2 else " BAD"
            line += f" | m{mode}s{st}={t:.1f}({tdg / t:.3f}){flag}"
    print(line, flush=True)
k = list(gl5._w8a8_block_fp8_matmul_hopper_gl5.device_caches[0][0].values())[0]
open("/tmp/gl5.ptx", "w").write(k.asm["ptx"])
