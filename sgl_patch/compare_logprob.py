# Is a greedy divergence a near-tie (numerics) or a real error?
# For each prompt: run ref greedily, then score ref's tokens under both servers (prefill-only, input logprobs)
# and report per-token |dlogp| plus the top-1/top-2 margin at the first position where test's argmax differs.
# python compare_logprob.py <port_ref> <port_test>
import json, sys, urllib.request

from compare_gen import PROMPTS, gen


def score(port, text):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", method="POST",
                                 headers={"Content-Type": "application/json"},
                                 data=json.dumps({"text": text, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                                                  "return_logprob": True, "logprob_start_len": 0,
                                                  "top_logprobs_num": 2}).encode())
    m = json.loads(urllib.request.urlopen(req, timeout=300).read())["meta_info"]
    return m["input_token_logprobs"], m["input_top_logprobs"]


ref_port, test_port = sys.argv[1], sys.argv[2]
worst_all = 0.0
for p in PROMPTS:
    full = p + gen(ref_port, p)["text"]
    lr, tr = score(ref_port, full)
    lt, tt = score(test_port, full)
    d = [abs(a[0] - b[0]) for a, b in zip(lr[1:], lt[1:]) if a[0] is not None and b[0] is not None]
    worst = max(d) if d else 0.0
    worst_all = max(worst_all, worst)
    # first position where the two servers' top-1 token differs, and how close ref's top-2 was there
    flip = None
    for i, (a, b) in enumerate(zip(tr, tt)):
        if a and b and a[0][1] != b[0][1]:
            flip = (i, a[0][0] - a[1][0], b[0][0] - b[1][0])
            break
    fl = f"argmax flip @tok{flip[0]} ref_margin={flip[1]:.3f} test_margin={flip[2]:.3f}" if flip else "no argmax flip"
    print(f"{p[:34]!r:38} mean|dlogp|={sum(d) / max(len(d), 1):.4f} max|dlogp|={worst:.3f}  {fl}")
print(f"worst max|dlogp| {worst_all:.3f}")
