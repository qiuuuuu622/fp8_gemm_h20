# c12 (gl6, warp-specialized TMA producer) vs c11 (gl5) vs DeepGEMM on the remaining gap shapes.
import itertools, re, torch, triton, deep_gemm
import gl5, gl6
from bench import make, ref

SH = [(512, 2048, 100), (1024, 2048, 333), (12288, 2048, 256), (2048, 4096, 100), (2048, 512, 256),
      (9216, 2048, 333), (12288, 2048, 8192)]
bench = lambda f: triton.testing.do_bench_cudagraph(f, rep=50) * 1e3


def space(M):
    small = M <= 512
    for bt, bw, sw, sp, st, nw in itertools.product([16, 32, 64] if small else [64, 128], [64, 128],
                                                    [True, False] if small else [False], [1, 2, 4] if small else [1, 2],
                                                    [3, 4, 6, 8], [4, 8]):
        if (not sw and bt < 64) or (sw and bt > 64) or (nw == 8 and (bw if sw else bt) < 128):
            continue
        yield dict(BLOCK_T=bt, BLOCK_W=bw, SWAP_AB=sw, SPLIT=sp, NUM_STAGES=st, num_warps=nw, GROUP_SIZE_M=1)


errs = {}
for N, K, M in SH:
    A, B, As, Bs = make(M, N, K)
    r = ref(A, B, As, Bs).float()
    Asd = deep_gemm.get_mn_major_tma_aligned_tensor(As)
    C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    tdg = bench(lambda: deep_gemm.fp8_gemm_nt((A, Asd), (B, Bs), C))
    best = {}
    for name, mod in (("c11", gl5), ("c12", gl6)):
        res = []
        for c in space(M):
            cc = {**c, "MODE": 1} if name == "c11" else c
            try:
                o = mod.matmul(A, B, As, Bs, cc).float()
                e = ((o - r).abs().max() / r.abs().max()).item()
                if e > 1e-2:
                    errs.setdefault(name + "_BAD", (round(e, 3), cc))
                    continue
                res.append((bench(lambda: mod.matmul(A, B, As, Bs, cc)), cc))
            except Exception as ex:
                errs.setdefault(f"{name}_{type(ex).__name__}", str(ex)[-300:])
        best[name] = min(res, key=lambda x: x[0]) if res else (float("inf"), None)
    print(f"N={N} K={K} M={M} dg={tdg:.1f} | c11={best['c11'][0]:.1f} ({tdg / best['c11'][0]:.3f}) | "
          f"c12={best['c12'][0]:.1f} ({tdg / best['c12'][0]:.3f}) {best['c12'][1]}", flush=True)
for k, v in errs.items():
    print("ERR", k, v)
ks = list(gl6._w8a8_block_fp8_matmul_hopper_gl6.device_caches[0][0].values())
if ks:
    open("/tmp/gl6.ptx", "w").write(ks[0].asm["ptx"])
