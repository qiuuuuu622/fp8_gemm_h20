# Greedy-decode the same prompts on two running servers and compare token-by-token.
# python compare_gen.py <port_ref> <port_test>
import json, sys, urllib.request

PROMPTS = [
    "The capital of France is",
    "Explain in two sentences what a block-wise FP8 GEMM is.",
    "写一段关于杭州西湖的简短介绍。",
    "def quicksort(arr):",
    "List the first ten prime numbers:",
    "Q: A train travels 120 km in 1.5 hours. What is its average speed?\nA:",
    "Translate to English: 今天天气很好，我们去公园散步吧。",
    "In the year 2050, renewable energy",
]


def gen(port, prompt, n=64):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", method="POST",
                                 headers={"Content-Type": "application/json"},
                                 data=json.dumps({"text": prompt, "sampling_params":
                                                  {"max_new_tokens": n, "temperature": 0}}).encode())
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


ref_port, test_port = sys.argv[1], sys.argv[2]
exact = 0
for p in PROMPTS:
    a, b = gen(ref_port, p), gen(test_port, p)
    ia, ib = a["output_ids"], b["output_ids"]
    same = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), min(len(ia), len(ib)))
    exact += ia == ib
    print(f"{'EXACT ' if ia == ib else f'diverge@{same:2d}'} | {p[:40]!r}")
    if ia != ib:
        print(f"    ref : {a['text'][:110]!r}\n    test: {b['text'][:110]!r}")
print(f"exact match {exact}/{len(PROMPTS)}")
