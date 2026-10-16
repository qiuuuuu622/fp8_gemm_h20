# Opt-in runtime patch: route SGLang's block-FP8 dense linear (DeepGEMM path) to the fp8mm dispatch kernels.
# Enabled only when FP8MM_ENABLE=1; loaded via sitecustomize from PYTHONPATH (no SGLang source edits).
#   FP8MM_ENABLE=1            replace dispatch_w8a8_block_fp8_linear() result with fp8mm_w8a8_block_fp8_linear
#   FP8MM_DISABLE_DG_PRECOMPILE=1   also skip DeepGEMM's all-M JIT precompile (it is unused once patched)
#   FP8MM_HOME=/root/fp8mm    where dispatch.py / runs/dispatch.json live
# Shapes our table does not cover (or non-bf16 / unaligned) fall through to the original DeepGEMM function.
import os
import sys

if os.environ.get("FP8MM_ENABLE") == "1":
    sys.path.insert(0, os.environ.get("FP8MM_HOME", "/root/fp8mm"))

    if os.environ.get("FP8MM_DISABLE_DG_PRECOMPILE") == "1":
        os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"

    import importlib.abc
    import importlib.machinery

    _TARGET = "sglang.srt.layers.quantization.fp8_utils"

    def _install(mod):
        import torch
        import dispatch as fp8mm_dispatch

        orig_dispatch = mod.dispatch_w8a8_block_fp8_linear
        table = fp8mm_dispatch._TABLE
        quant = mod.sglang_per_token_group_quant_fp8
        stats = {"ours": 0, "fallback": 0}

        def make_linear(orig_linear):
            def fp8mm_w8a8_block_fp8_linear(input, weight, block_size, weight_scale, input_scale=None, bias=None):
                N, K = weight.shape
                ok = (
                    input_scale is None
                    and input.dtype == torch.bfloat16
                    and weight_scale.dtype == torch.float32
                    and list(block_size) == [128, 128]
                    and f"{N},{K}" in table
                )
                if not ok:
                    stats["fallback"] += 1
                    return orig_linear(input, weight, block_size, weight_scale, input_scale, bias)
                stats["ours"] += 1
                x2 = input.view(-1, K)
                # row-major [M, K/128] fp32 scales (our kernels' layout), not DeepGEMM's MN-major TMA layout
                q, s = quant(x2, block_size[1], column_major_scales=False, scale_tma_aligned=False)
                out = fp8mm_dispatch.matmul(q, weight, s, weight_scale)
                if bias is not None:
                    out = out + bias
                return out.view(*input.shape[:-1], N)

            fp8mm_w8a8_block_fp8_linear.fp8mm_stats = stats
            return fp8mm_w8a8_block_fp8_linear

        def patched_dispatch():
            orig = orig_dispatch()
            if orig is not mod.deepgemm_w8a8_block_fp8_linear_with_fallback:
                return orig  # only replace the DeepGEMM path
            return make_linear(orig)

        mod.dispatch_w8a8_block_fp8_linear = patched_dispatch
        print(f"[fp8mm] patched {_TARGET}.dispatch_w8a8_block_fp8_linear "
              f"(table NK={sorted(table)}, dg_precompile={os.environ.get('SGLANG_JIT_DEEPGEMM_PRECOMPILE', '1')})",
              file=sys.stderr, flush=True)

    class _Loader(importlib.abc.Loader):
        """Delegates to the real loader, then installs the patch on the freshly executed module."""

        def __init__(self, inner):
            self.inner = inner

        def create_module(self, spec):
            return self.inner.create_module(spec)

        def exec_module(self, module):
            self.inner.exec_module(module)
            _install(module)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return None
            spec.loader = _Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, _Finder())
