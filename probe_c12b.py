# c12 with maxnreg cap (setmaxnreg needs a kernel-wide reg budget): sweep maxnreg on the c12 winners
import torch, triton, deep_gemm, re
import gl5, gl6
from bench import make, ref

CASES = [
    (512, 2048, 100, dict(BLOCK_T=16, BLOCK_W=64, SWAP_AB=True, SPLIT=1, NUM_STAGES=6, num_warps=4, GROUP_SIZE_M=1)),
    (1024, 2048, 333, dict(BLOCK_T=16, BLOCK_W=64, SWAP_AB=True, SPLIT=2, NUM_STAGES=4, num_warps=4, GROUP_SIZE_M=1)),
    (2048, 4096, 100, dict(BLOCK_T=16, BLOCK_W=64, SWAP_AB=True, SPLIT=1, NUM_STAGES=6, num_warps=4, GROUP_SIZE_M=1)),
    (12288, 2048, 256, dict(BLOCK_T=64, BLOCK_W=128, SWAP_AB=False, SPLIT=1, NUM_STAGES=4, num_warps=4, GROUP_SIZE_M=1)),
]
bench = lambda f: triton.testing.do_bench_cudagraph(f, rep=50) * 1e3
orig = gl6._w8a8_block_fp8_matmul_hopper_gl6
for N, K, M, c in CASES:
    A, B, As, Bs = make(M, N, K)
    r = ref(A, B, As, Bs).float()
    Asd = deep_gemm.get_mn_major_tma_aligned_tensor(As)
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    tdg = bench(lambda: deep_gemm.fp8_gemm_nt((A, Asd), (B, Bs), C))
    line = f"N={N} K={K} M={M} dg={tdg:.1f}"
    for mr in (None, 128, 168, 232):
        for preg in (24, 40):
            # gluon launch kwarg maxnreg caps all partitions; worker_num_regs then redistributes
            gl6._w8a8_block_fp8_matmul_hopper_gl6 = orig if mr is None else (lambda m: type("K", (), {
                "__getitem__": lambda s, g: (lambda *a, **kw: orig[g](*a, maxnreg=m, **kw))})())(mr)
            cc = {**c, "PRODUCER_REGS": preg}
            try:
                o = gl6.matmul(A, B, As, Bs, cc).float()
                e = ((o - r).abs().max() / r.abs().max()).item()
                t = bench(lambda: gl6.matmul(A, B, As, Bs, cc))
                line += f" | mr{mr}/p{preg}={t:.1f}({tdg / t:.3f}){'' if e < 1e-2 else ' BAD'}"
            except Exception as ex:
                line += f" | mr{mr}/p{preg}=ERR {type(ex).__name__}:{str(ex)[:60]}"
    gl6._w8a8_block_fp8_matmul_hopper_gl6 = orig
    print(line, flush=True)
