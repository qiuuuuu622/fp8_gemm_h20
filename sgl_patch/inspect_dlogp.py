# Where does the large |dlogp| come from? Print the top positions with both servers' logprob for the scored token.
# python inspect_dlogp.py <port_ref> <port_test> <prompt_index>
import sys

from compare_gen import PROMPTS, gen
from compare_logprob import score

ref_port, test_port, i = sys.argv[1], sys.argv[2], int(sys.argv[3])
p = PROMPTS[i]
full = p + gen(ref_port, p)["text"]
lr, tr = score(ref_port, full)
lt, tt = score(test_port, full)
rows = []
for k, (a, b) in enumerate(zip(lr, lt)):
    if k == 0 or a[0] is None or b[0] is None:
        continue
    rows.append((abs(a[0] - b[0]), k, a[0], b[0], a[1], tr[k][0][1] if tr[k] else None, tt[k][0][1] if tt[k] else None))
rows.sort(reverse=True)
print(f"prompt {i}: {p!r}  scored tokens={len(rows)}")
for d, k, la, lb, tok, top_r, top_t in rows[:8]:
    print(f"  pos {k:3d} tok {tok:6d}  logp ref={la:8.3f} test={lb:8.3f}  |d|={d:.3f}  "
          f"top1 ref={top_r} test={top_t}")
big = [r for r in rows if r[0] > 0.5]
print(f"positions with |d|>0.5: {len(big)}; of those with ref logp > -2 (confident tokens): "
      f"{sum(1 for r in big if r[2] > -2)}")
