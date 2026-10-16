# Upper bound on the e2e gain from dense GEMM alone: per-layer dense FP8 GEMM time at prefill M,
# DeepGEMM vs ours, times the model's layer counts (Qwen3.6-35B-A3B: 40 layers = 30 GDN + 10 full-attn).
# Compares the saved time with the measured TTFT delta.
import sys, torch, triton, deep_gemm, dispatch
from bench import make

GDN, ATT = 30, 10  # full_attention_interval=4 over 40 layers
SHAPES = {  # (N, K, per-model count)
    "gdn.in_proj_qkvz": (12288, 2048, GDN), "gdn.out_proj": (2048, 4096, GDN),
    "attn.qkv_proj": (9216, 2048, ATT), "attn.o_proj": (2048, 4096, ATT),
    "shared.gate_up": (1024, 2048, GDN + ATT), "shared.down": (2048, 512, GDN + ATT),
}
bench = lambda f: triton.testing.do_bench_cudagraph(f, rep=50) * 1e3
for M in [int(x) for x in (sys.argv[1:] or ["4096", "8192"])]:
    tdg = tours = 0.0
    for name, (N, K, cnt) in SHAPES.items():
        A, B, As, Bs = make(M, N, K)
        Asd = deep_gemm.get_mn_major_tma_aligned_tensor(As)
        C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        a = bench(lambda: deep_gemm.fp8_gemm_nt((A, Asd), (B, Bs), C))
        b = bench(lambda: dispatch.matmul(A, B, As, Bs))
        tdg += a * cnt
        tours += b * cnt
        print(f"  M={M} {name:18s} x{cnt:2d}  dg={a:8.1f}us ours={b:8.1f}us  ({a / b:.3f}x)")
    print(f"M={M}: dense GEMM per forward  dg={tdg / 1e3:.2f}ms ours={tours / 1e3:.2f}ms  saved={(tdg - tours) / 1e3:.2f}ms "
          f"({tdg / tours:.3f}x on the GEMM part)")
