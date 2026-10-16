# python profile/ncu_harness.py <impl:ours|dg> N K M   (2 warmup + 1 profiled launch; use -s 2 -c 1)
import sys, torch, deep_gemm, dispatch
from bench import make
impl, N, K, M = sys.argv[1], *map(int, sys.argv[2:5])
A, B, As, Bs = make(M, N, K)
Asd = deep_gemm.get_mn_major_tma_aligned_tensor(As)
C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
f = (lambda: dispatch.matmul(A, B, As, Bs)) if impl == "ours" else (lambda: deep_gemm.fp8_gemm_nt((A, Asd), (B, Bs), C))
for _ in range(3): f()
torch.cuda.synchronize()
