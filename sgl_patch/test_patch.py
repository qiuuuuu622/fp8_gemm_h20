# Unit check of the patch on real /prefill_model weights, run as two processes (patched / unpatched):
#   FP8MM_ENABLE=1 PYTHONPATH=/root/fp8mm/sgl_patch python test_patch.py out_ours.pt
#   python test_patch.py out_dg.pt
# then: python test_patch.py --compare out_ours.pt out_dg.pt
import json, os, re, sys
import torch

if sys.argv[1] == "--compare":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    worst = 0.0
    for k in a:
        e = ((a[k].float() - b[k].float()).abs().max() / b[k].float().abs().max()).item()
        worst = max(worst, e)
        print(f"{k:55s} rel_diff={e:.2e}")
    print("worst", f"{worst:.2e}", "PASS" if worst < 2e-2 else "FAIL")
    sys.exit(0)

from safetensors import safe_open
from sglang.srt.layers.quantization import fp8_utils  # patched on import when FP8MM_ENABLE=1

root = "/prefill_model"
idx = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
want = {}
for k in idx:
    m = re.match(r"model\.language_model\.layers\.(\d+)\.(.+)\.weight$", k)
    if not m or "experts." in k and "shared" not in k:
        continue
    name = re.sub(r"layers\.\d+\.", "layers.X.", k)
    if name not in want:
        want[name] = k
lin = fp8_utils.dispatch_w8a8_block_fp8_linear()
print("linear fn:", lin.__name__, file=sys.stderr)
torch.manual_seed(0)
out = {}
for name, k in sorted(want.items()):
    with safe_open(f"{root}/{idx[k]}", "pt", device="cuda") as s:
        w = s.get_tensor(k)
        if w.dtype != torch.float8_e4m3fn:
            continue
        ws = s.get_tensor(k.replace(".weight", ".weight_scale_inv")).float()
    N, K = w.shape
    for M in (7, 100, 333, 4096):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        out[f"{name.split('layers.X.')[-1]} M={M}"] = lin(x, w, [128, 128], ws).cpu()
torch.save(out, sys.argv[1])
print("stats:", getattr(lin, "fp8mm_stats", "n/a (original)"), file=sys.stderr)
