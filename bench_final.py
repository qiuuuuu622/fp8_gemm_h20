# Final clean comparison (same 120 cases as bench_dg.py): dispatch(c3|c9) vs c3-only vs DeepGEMM vs baseline.
# Protocol: run alone on one GPU, warm every kernel (incl. DeepGEMM JIT) before timing,
# per case time impls back-to-back, median of 3 x do_bench_cudagraph(rep=100).
import sys, csv, statistics
import torch, triton
import deep_gemm
import baseline, opt2, dispatch
from bench import make, ref, DEFAULT_CFG
from shapes import SHAPES, MS, EXT_SHAPES, EXT_MS

CASES = [(s, m) for s in SHAPES for m in sorted(set(MS + EXT_MS))] + \
        [(s, m) for s in EXT_SHAPES for m in EXT_MS]


def med_us(f):
    f()
    torch.cuda.synchronize()
    return statistics.median(triton.testing.do_bench_cudagraph(f, rep=100) for _ in range(3)) * 1e3


def main():
    rows = []
    for (name, N, K), M in CASES:
        A, B, As, Bs = make(M, N, K)
        r = ref(A, B, As, Bs).float()
        As_dg = deep_gemm.get_mn_major_tma_aligned_tensor(As)
        C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        dg = lambda: deep_gemm.fp8_gemm_nt((A, As_dg), (B, Bs), C)
        dg()
        err = lambda o: ((o.float() - r).abs().max() / r.abs().max()).item()
        e_dsp = err(dispatch.matmul(A, B, As, Bs))
        dg(); torch.cuda.synchronize(); e_dg = err(C)
        impl, _ = dispatch.pick(M, N, K)
        t_dsp = med_us(lambda: dispatch.matmul(A, B, As, Bs))
        t_c3 = med_us(lambda: opt2.matmul(A, B, As, Bs))
        t_dg = med_us(dg)
        t_b = med_us(lambda: baseline.run(A, B, As, Bs, DEFAULT_CFG))
        tf = lambda us: round(2 * M * N * K / us / 1e6, 1)
        row = dict(layer=name, N=N, K=K, M=M, impl=impl, base_tf=tf(t_b), c3_tf=tf(t_c3), dsp_tf=tf(t_dsp),
                   dg_tf=tf(t_dg), dsp_vs_dg=round(t_dg / t_dsp, 3), c3_vs_dg=round(t_dg / t_c3, 3),
                   dsp_vs_base=round(t_b / t_dsp, 3), err_dsp=f"{e_dsp:.1e}", err_dg=f"{e_dg:.1e}")
        rows.append(row)
        print(row, flush=True)
    out = sys.argv[1] if len(sys.argv) > 1 else "runs/bench_final.csv"
    with open(out, "w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
