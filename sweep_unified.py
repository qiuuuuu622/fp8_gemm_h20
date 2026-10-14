# Unified sweep: per (N,K,M) pick the faster of c3 (opt2) and c9 (gl2) with its best config.
# Stage 1: rep=20 over the space; Stage 2: re-time each impl's top-2 with rep=100, keep the fastest.
# Usage: python sweep_unified.py <out.json> <shard> <nshard>
import itertools, json, sys
import torch, triton
import opt2, gl2
from bench import make, ref
from shapes import SHAPES, EXT_SHAPES

NK = [(N, K) for _, N, K in SHAPES + EXT_SHAPES]
TUNE_M = [16, 32, 64, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192, 16384]


def grid(space):
    return [dict(zip(space, v)) for v in itertools.product(*space.values())]


C3_SMALL = grid(dict(BLOCK_SIZE_M=[32, 64], BLOCK_SIZE_N=[32, 64, 128], GROUP_SIZE_M=[1, 8], num_warps=[4],
                     num_stages=[2, 3, 4], SPLIT_K=[1, 2]))
C3_LARGE = [c for c in grid(dict(BLOCK_SIZE_M=[64, 128], BLOCK_SIZE_N=[64, 128], GROUP_SIZE_M=[1, 8, 32],
                                 num_warps=[4, 8], num_stages=[3, 4], SPLIT_K=[1]))
            if not (c["BLOCK_SIZE_M"] == 128 and c["BLOCK_SIZE_N"] == 128 and c["num_warps"] == 4)]
C9_SMALL = [c for c in grid(dict(BLOCK_T=[16, 32, 64], BLOCK_W=[64, 128], SWAP_AB=[True, False], SPLIT=[1, 2, 4],
                                 NUM_STAGES=[3, 4], num_warps=[4], GROUP_SIZE_M=[1, 8]))
            if c["SWAP_AB"] or c["BLOCK_T"] == 64]
C9_LARGE = [c for c in grid(dict(BLOCK_T=[64, 128], BLOCK_W=[64, 128], SWAP_AB=[False], SPLIT=[1, 2],
                                 NUM_STAGES=[3, 4], num_warps=[4, 8], GROUP_SIZE_M=[1, 8]))
            if not (c["num_warps"] == 8 and c["BLOCK_T"] < 128)]
IMPL = {"c3": opt2.matmul, "c9": gl2.matmul}


def time_cfg(impl, cfg, args, r, rep):
    f = IMPL[impl]
    o = f(*args, cfg).float()
    if ((o - r).abs().max() / r.abs().max()).item() > 2e-2:
        return None
    return triton.testing.do_bench_cudagraph(lambda: f(*args, cfg), rep=rep)


def main():
    out, shard, nshard = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    jobs = [(nk, m) for m in TUNE_M for nk in NK]  # M-major interleave balances shards
    res, detail = {}, []
    for i, ((N, K), M) in enumerate(jobs):
        if i % nshard != shard:
            continue
        A, B, As, Bs = make(M, N, K)
        r = ref(A, B, As, Bs).float()
        small = M <= 512
        cands = {"c3": C3_SMALL if small else C3_LARGE, "c9": C9_SMALL if small else C9_LARGE}
        top = {}
        for impl, space in cands.items():
            ts = []
            for c in space:
                try:
                    t = time_cfg(impl, c, (A, B, As, Bs), r, 20)
                except Exception:
                    continue
                if t is not None:
                    ts.append((t, c))
            ts.sort(key=lambda x: x[0])
            # stage 2: re-time top-2 with longer rep to cut noise
            top[impl] = min(((time_cfg(impl, c, (A, B, As, Bs), r, 100), c) for _, c in ts[:2]),
                            key=lambda x: x[0], default=(float("inf"), None))
        win = min(top, key=lambda k: top[k][0])
        res.setdefault(f"{N},{K}", {})[str(M)] = {"impl": win, "cfg": top[win][1]}
        row = dict(N=N, K=K, M=M, c3_us=round(top["c3"][0] * 1e3, 2), c9_us=round(top["c9"][0] * 1e3, 2), win=win)
        detail.append(row)
        print(row, flush=True)
        json.dump(res, open(out, "w"), indent=1)
        json.dump(detail, open(out.replace(".json", "_detail.json"), "w"))


if __name__ == "__main__":
    main()
