# python profile/ncu_h2.py <c3|c9|dg> N K M [json-cfg]
import sys, json, torch, deep_gemm, opt2, gl2
from bench import make
impl, N, K, M = sys.argv[1], *map(int, sys.argv[2:5])
cfg = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
A, B, As, Bs = make(M, N, K)
Asd = deep_gemm.get_mn_major_tma_aligned_tensor(As)
C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
f = {"c3": lambda: opt2.matmul(A, B, As, Bs, cfg), "c9": lambda: gl2.matmul(A, B, As, Bs, cfg),
     "dg": lambda: deep_gemm.fp8_gemm_nt((A, Asd), (B, Bs), C)}[impl]
for _ in range(3): f()
torch.cuda.synchronize()
