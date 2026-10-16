# P0: cold-start compile cost for the /prefill_model dense FP8 shapes.
#   python startup_cost.py dg_full   -> DeepGEMM, sglang default precompile (M = 1..16384 per shape, chunk 8192)
#   python startup_cost.py dg_fast   -> DeepGEMM, SGLANG_JIT_DEEPGEMM_FAST_WARMUP M list
#   python startup_cost.py ours      -> dispatch table (c3/c11): compile every distinct (impl, cfg) once
# Each run uses a fresh, empty JIT cache dir so the numbers are true cold-start costs.
import os, sys, tempfile, time, json

mode = sys.argv[1]
cache = tempfile.mkdtemp(prefix=f"coldcache_{mode}_")
os.environ["DG_JIT_CACHE_DIR"] = cache
os.environ["DG_JIT_USE_NVRTC"] = "0"  # sglang default (SGL_DG_USE_NVRTC=0 -> nvcc)
os.environ["TRITON_CACHE_DIR"] = cache
import torch  # noqa: E402

from shapes import SHAPES  # noqa: E402

NK = [(N, K) for _, N, K in SHAPES]
CHUNK = 8192


def dg_m_list(fast):
    if not fast:  # compile_utils.update_deep_gemm_config, non-fast branch
        return list(range(1, 2 * 8192 + 1)) if CHUNK > 8192 else list(range(1, 16 * 1024 + 1))
    ms = list(range(1, 1025))
    next_m, step = 1024, 2
    while next_m < CHUNK:
        ms += list(range(next_m, 2 * next_m, step))
        next_m, step = next_m * 2, step * 2
    ms.append(CHUNK)
    return sorted(set(ms))


def count_files(d):
    return sum(len(f) for _, _, f in os.walk(d))


t_all = time.perf_counter()
rows = []
if mode.startswith("dg"):
    import deep_gemm
    ms = dg_m_list(mode == "dg_fast")
    has_mode = hasattr(deep_gemm, "get_compile_mode") and hasattr(deep_gemm, "set_compile_mode")
    for N, K in NK:
        t0 = time.perf_counter()
        max_m = max(ms)
        A = torch.empty(max_m, K, device="cuda", dtype=torch.float8_e4m3fn)
        As = torch.ones(max_m, K // 128, device="cuda", dtype=torch.float32)
        B = torch.empty(N, K, device="cuda", dtype=torch.float8_e4m3fn)
        Bs = torch.ones(N // 128, K // 128, device="cuda", dtype=torch.float32)
        C = torch.empty(max_m, N, device="cuda", dtype=torch.bfloat16)
        if has_mode:
            old = deep_gemm.get_compile_mode()
            deep_gemm.set_compile_mode(1)
        for m in ms:
            deep_gemm.fp8_gemm_nt((A[:m], As[:m]), (B, Bs), C[:m])
        if has_mode:
            deep_gemm.set_compile_mode(old)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rows.append(dict(N=N, K=K, n_m=len(ms), sec=round(dt, 1), cache_files=count_files(cache)))
        print(rows[-1], flush=True)
        del A, As, B, Bs, C
        torch.cuda.empty_cache()
else:
    import dispatch
    from bench import make
    for N, K in NK:
        t0 = time.perf_counter()
        seen = {}
        for m, e in dispatch._TABLE.get(f"{N},{K}", {}).items():
            key = (e["impl"], json.dumps(e["cfg"], sort_keys=True))
            seen.setdefault(key, int(m))
        for (impl, cfg), m in seen.items():
            A, B, As, Bs = make(m, N, K)
            dispatch.IMPLS[impl](A, B, As, Bs, json.loads(cfg))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rows.append(dict(N=N, K=K, n_cfg=len(seen), sec=round(dt, 1), cache_files=count_files(cache)))
        print(rows[-1], flush=True)
total = time.perf_counter() - t_all
print(json.dumps(dict(mode=mode, total_sec=round(total, 1), cache_dir=cache, rows=rows)), flush=True)
json.dump(dict(mode=mode, total_sec=round(total, 1), rows=rows), open(f"runs/startup_cost_{mode}.json", "w"), indent=1)
