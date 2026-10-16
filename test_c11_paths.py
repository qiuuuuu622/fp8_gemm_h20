# c11 (gl5) path coverage: every SWAP_AB x SPLIT x tile combo on ragged M, plus repeated calls (counter reset),
# CUDA-graph replay, and two streams running split-K GEMMs concurrently (per-stream buffers).
import itertools, torch
import gl5 as gl2
from bench import ref
from test_correct import make

torch.manual_seed(2)
CFGS = [dict(BLOCK_T=t, BLOCK_W=w, SWAP_AB=s, SPLIT=k, NUM_STAGES=3, num_warps=4, GROUP_SIZE_M=g, MODE=md)
        for t, w, s, k, g, md in itertools.product([16, 32, 64], [64, 128], [True, False], [1, 2, 4], [1, 8], [1, 2])
        if s or t == 64]
NK = [(512, 2048), (2048, 4096), (2048, 512), (2304, 2048), (2048, 256), (12288, 2048)]
MS = [1, 7, 17, 64, 100, 333, 1000]
fails = []
n = 0


def err(o, r):
    return ((o.float() - r).abs().max() / r.abs().max().clamp(min=1e-30)).item()


for (N, K), M in itertools.product(NK, MS):
    A, B, As, Bs = make(M, N, K, "outlier" if M % 2 else "normal")
    r = ref(A, B, As, Bs).float()
    for c in CFGS:
        o1 = gl2.matmul(A, B, As, Bs, c)
        o2 = gl2.matmul(A, B, As, Bs, c)  # 2nd call: counters must have self-reset
        e = err(o1, r)
        n += 1
        if not torch.isfinite(o1).all() or e > 1e-2 or not torch.equal(o1, o2):
            fails.append((N, K, M, c, round(e, 4), torch.equal(o1, o2)))
print(f"path cases={n} fails={len(fails)}")
for f in fails[:10]:
    print("FAIL", f)

# CUDA graph replay with split-K tail
A, B, As, Bs = make(100, 2048, 4096, "normal")
c = dict(BLOCK_T=16, BLOCK_W=64, SWAP_AB=True, SPLIT=4, NUM_STAGES=3, num_warps=4, GROUP_SIZE_M=1)
r = ref(A, B, As, Bs).float()
gl2.matmul(A, B, As, Bs, c)
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    gl2.matmul(A, B, As, Bs, c)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = gl2.matmul(A, B, As, Bs, c)
ok = True
for _ in range(50):
    g.replay()
    torch.cuda.synchronize()
    ok &= err(out, r) < 1e-2
print("graph replay x50:", "PASS" if ok else "FAIL")

# two streams concurrently, different shapes, both split-K -> must not corrupt each other
A1, B1, As1, Bs1 = make(100, 2048, 4096, "normal")
A2, B2, As2, Bs2 = make(64, 512, 2048, "outlier")
r1, r2 = ref(A1, B1, As1, Bs1).float(), ref(A2, B2, As2, Bs2).float()
c2 = dict(BLOCK_T=16, BLOCK_W=64, SWAP_AB=True, SPLIT=4, NUM_STAGES=3, num_warps=4, GROUP_SIZE_M=1)
s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
bad = 0
for _ in range(200):
    with torch.cuda.stream(s1):
        o1 = gl2.matmul(A1, B1, As1, Bs1, c)
    with torch.cuda.stream(s2):
        o2 = gl2.matmul(A2, B2, As2, Bs2, c2)
    torch.cuda.synchronize()
    bad += (err(o1, r1) > 1e-2) + (err(o2, r2) > 1e-2)
print("2-stream concurrent x200:", "PASS" if bad == 0 else f"FAIL bad={bad}")
print("buffers keyed:", len(gl2._BUFS))
print("PASS" if not fails and ok and bad == 0 else "FAILED")
