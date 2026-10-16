# End-to-end A/B: DeepGEMM server vs fp8mm server, same args, CUDA graphs on (breakable prefill + decode graphs).
# Workload per case: N requests with a fixed random-token prompt of length L, max_new_tokens=O, ignore_eos,
# radix cache defeated by a unique prefix per request. Metrics: TTFT (prefill-dominated), per-request prefill
# throughput, TPOT, end-to-end. Runs servers alternately per case, R repeats, reports medians.
# python ab_e2e.py <port_a> <name_a> <port_b> <name_b> <out.json>
import json, random, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

pa, na, pb, nb, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
# (prompt_len, max_new_tokens, concurrency, n_requests)
CASES = [
    (1024, 1, 1, 8), (4096, 1, 1, 8), (8192, 1, 1, 6), (16384, 1, 1, 4), (32768, 1, 1, 3),
    (8192, 1, 4, 8),
    (2048, 128, 1, 4), (8192, 128, 1, 4),
    (1024, 256, 16, 32),
]
R = 3
VOCAB = (1000, 150000)


def stream(port, ids, n_out):
    body = json.dumps({"input_ids": ids, "stream": True,
                       "sampling_params": {"max_new_tokens": n_out, "temperature": 0, "ignore_eos": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            n += 1
    e2e = time.perf_counter() - t0
    return ttft, e2e


def run_case(port, L, O, conc, nreq, seed):
    rng = random.Random(seed)
    reqs = [[rng.randint(*VOCAB) for _ in range(L)] for _ in range(nreq)]  # unique -> no prefix reuse
    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(lambda ids: stream(port, ids, O), reqs))
    wall = time.perf_counter() - t0
    ttft = [r[0] for r in res]
    tpot = [(r[1] - r[0]) / (O - 1) for r in res] if O > 1 else []
    return dict(ttft_ms=statistics.median(ttft) * 1e3, wall_s=wall,
                prefill_tok_s=L * nreq / sum(ttft) if conc == 1 else None,
                tpot_ms=statistics.median(tpot) * 1e3 if tpot else None,
                req_s=nreq / wall)


# warm both servers on every case shape (first-hit costs excluded)
for port in (pa, pb):
    for L, O, conc, nreq in CASES:
        run_case(port, L, min(O, 2), 1, 1, seed=999)

results = []
for ci, (L, O, conc, nreq) in enumerate(CASES):
    per = {na: [], nb: []}
    for r in range(R):
        order = [(pa, na), (pb, nb)] if r % 2 == 0 else [(pb, nb), (pa, na)]
        for port, name in order:
            per[name].append(run_case(port, L, O, conc, nreq, seed=1000 * ci + r))
    row = dict(L=L, O=O, conc=conc, nreq=nreq)
    for name in (na, nb):
        for k in per[name][0]:
            vals = [x[k] for x in per[name] if x[k] is not None]
            row[f"{name}_{k}"] = round(statistics.median(vals), 3) if vals else None
    results.append(row)
    a_t, b_t = row[f"{na}_ttft_ms"], row[f"{nb}_ttft_ms"]
    msg = f"L={L:6d} O={O:4d} c={conc:2d} | TTFT {na}={a_t:9.1f} {nb}={b_t:9.1f} ms ({nb}/{na} speedup {a_t / b_t:.3f})"
    if O > 1:
        msg += f" | TPOT {na}={row[f'{na}_tpot_ms']:.2f} {nb}={row[f'{nb}_tpot_ms']:.2f} ms"
    if conc > 1:
        msg += f" | req/s {na}={row[f'{na}_req_s']:.2f} {nb}={row[f'{nb}_req_s']:.2f}"
    print(msg, flush=True)
json.dump(results, open(out, "w"), indent=1)
print("wrote", out)
