# Correctness: opt2 vs fp32-dequant reference AND vs baseline kernel, incl. ragged M, TP-split shapes,
# outlier/skewed scales. Usage: python test_correct.py
import itertools, torch
import os, importlib
import baseline, opt2
IMPL = importlib.import_module(os.environ.get("IMPL", "opt2"))
from bench import ref, FP8, DEFAULT_CFG

torch.manual_seed(1)
DEV = "cuda"
# model shapes (TP=1) + TP=2/4 splits (column-parallel splits N, row-parallel splits K)
NK = [(12288, 2048), (2048, 4096), (9216, 2048), (1024, 2048), (2048, 512),
      (6144, 2048), (2048, 2048), (4608, 2048), (512, 2048), (2048, 256),
      (3072, 2048), (2048, 1024), (2304, 2048), (256, 2048), (2048, 128)]
MS = [1, 7, 31, 64, 100, 255, 333, 1000, 1023, 2049, 3000, 4097, 8191]


def make(M, N, K, mode):
    a = torch.randn(M, K, device=DEV)
    b = torch.randn(N, K, device=DEV) * 0.05
    if mode == "outlier":  # heavy-tailed activations, a few hot channels
        a[:, torch.randint(0, K, (8,))] *= 100
        a *= torch.rand(M, 1, device=DEV) * 10
    if mode == "skew_w":  # weight blocks with very different magnitudes
        b *= torch.logspace(-3, 1, N // 128, device=DEV).repeat_interleave(128)[:, None]
    As = a.view(M, K // 128, 128).abs().amax(-1).clamp(min=1e-6) / 448.0
    A = (a.view(M, K // 128, 128) / As[..., None]).view(M, K).to(FP8)
    bb = b.view(N // 128, 128, K // 128, 128)
    Bs = bb.abs().amax((1, 3)).clamp(min=1e-6) / 448.0
    B = (bb / Bs[:, None, :, None]).view(N, K).to(FP8)
    return A, B, As.contiguous(), Bs.contiguous()


def main():
    worst_ref = worst_base = 0.0
    fails = []
    n = 0
    for (N, K), M, mode in itertools.product(NK, MS, ["normal", "outlier", "skew_w"]):
        A, B, As, Bs = make(M, N, K, mode)
        r = ref(A, B, As, Bs).float()
        o = IMPL.matmul(A, B, As, Bs).float()
        bo = baseline.run(A, B, As, Bs, DEFAULT_CFG).float()
        scale = r.abs().max().clamp(min=1e-30)
        e_ref = ((o - r).abs().max() / scale).item()
        e_base = ((o - bo).abs().max() / scale).item()
        b_ref = ((bo - r).abs().max() / scale).item()
        finite = torch.isfinite(o).all().item()
        n += 1
        worst_ref, worst_base = max(worst_ref, e_ref), max(worst_base, e_base)
        # opt must be no worse than baseline vs ref (within bf16 rounding slack)
        if not finite or e_ref > max(2 * b_ref, 1e-2):
            fails.append((N, K, M, mode, e_ref, b_ref, e_base))
    print(f"cases={n} worst_rel_err_vs_ref={worst_ref:.2e} worst_rel_diff_vs_base={worst_base:.2e}")
    for f in fails:
        print("FAIL N=%d K=%d M=%d %s opt_vs_ref=%.2e base_vs_ref=%.2e opt_vs_base=%.2e" % f)
    print("PASS" if not fails else f"FAILED {len(fails)}")


if __name__ == "__main__":
    main()
