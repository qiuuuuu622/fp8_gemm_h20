# Correctness + perf harness. Usage: python bench.py [impl ...]   impls: base, base_tuned, opt, deepgemm
import sys, json, csv, os, itertools
import torch, triton
from shapes import SHAPES, MS, EXT_SHAPES, EXT_MS
import baseline

torch.manual_seed(0)
DEV = "cuda"
FP8 = torch.float8_e4m3fn
DEFAULT_CFG = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=32, num_warps=4, num_stages=3)
CFG_DIR = "/sgl-workspace/sglang/python/sglang/kernels/ops/quantization/configs"


def make(M, N, K):
    a = torch.randn(M, K, device=DEV)
    b = torch.randn(N, K, device=DEV) * 0.05
    As = a.view(M, K // 128, 128).abs().amax(-1).clamp(min=1e-4) / 448.0
    A = (a.view(M, K // 128, 128) / As[..., None]).view(M, K).to(FP8)
    bb = b.view(N // 128, 128, K // 128, 128)
    Bs = bb.abs().amax((1, 3)).clamp(min=1e-4) / 448.0
    B = (bb / Bs[:, None, :, None]).view(N, K).to(FP8)
    return A, B, As.contiguous(), Bs.contiguous()


def ref(A, B, As, Bs):
    M, K = A.shape
    N = B.shape[0]
    a = A.float().view(M, K // 128, 128) * As[..., None]
    b = B.float().view(N // 128, 128, K // 128, 128) * Bs[:, None, :, None]
    return (a.view(M, K) @ b.view(N, K).t()).to(torch.bfloat16)


def sgl_tuned_cfg(N, K, M):
    f = f"{CFG_DIR}/N={N},K={K},device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json"
    if not os.path.exists(f):
        return DEFAULT_CFG
    t = {int(k): v for k, v in json.load(open(f)).items()}
    return t[min(t, key=lambda x: abs(x - M))]


def get_impl(name):
    if name == "base":
        return lambda A, B, As, Bs: baseline.run(A, B, As, Bs, DEFAULT_CFG)
    if name == "base_tuned":
        return lambda A, B, As, Bs: baseline.run(A, B, As, Bs, sgl_tuned_cfg(B.shape[0], A.shape[1], A.shape[0]))
    if name == "deepgemm":
        import deep_gemm
        def f(A, B, As, Bs):
            C = torch.empty(A.shape[0], B.shape[0], device=DEV, dtype=torch.bfloat16)
            deep_gemm.fp8_gemm_nt((A, As), (B, Bs), C)
            return C
        return f
    if name == "opt":
        import opt
        return opt.matmul
    if name == "opt2":
        import opt2
        return opt2.matmul
    raise ValueError(name)


def bench(fn, *args):
    return triton.testing.do_bench_cudagraph(lambda: fn(*args), rep=50)


def main(impls):
    ext = os.environ.get("EXT") == "1"
    shapes, ms = (EXT_SHAPES + SHAPES, EXT_MS) if ext else (SHAPES, MS)
    fns = {n: get_impl(n) for n in impls}
    rows = []
    for (name, N, K), M in itertools.product(shapes, ms):
        A, B, As, Bs = make(M, N, K)
        r = ref(A, B, As, Bs).float()
        row = dict(layer=name, M=M, N=N, K=K)
        for n, fn in fns.items():
            try:
                out = fn(A, B, As, Bs).float()
                err = ((out - r).abs().max() / r.abs().max()).item()
                ok = err < 2e-2
                us = bench(fn, A, B, As, Bs) * 1e3 if ok else float("nan")
            except Exception as e:  # record, don't hide
                err, ok, us = float("nan"), False, float("nan")
                print(f"[{n}] {name} M={M}: {type(e).__name__}: {str(e)[:200]}")
            row[f"{n}_us"] = round(us, 2)
            row[f"{n}_tflops"] = round(2 * M * N * K / us / 1e6, 1) if ok else 0
            row[f"{n}_ok"] = ok
        rows.append(row)
        print(row, flush=True)
    out = f"runs/bench_{'ext_' if ext else ''}{'_'.join(impls)}.csv"
    os.makedirs("runs", exist_ok=True)
    with open(out, "w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:] or ["base", "base_tuned", "deepgemm"])
