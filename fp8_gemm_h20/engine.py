"""One kernel registry shared by all model profiles."""
from functools import lru_cache
from importlib import import_module

KERNELS = {
    "triton": ("baseline_triton", "block_fp8_gemm_triton_h20"),
    "gluon": ("baseline_gluon", "block_fp8_gemm_gluon_h20"),
    "v2": ("kernel_v2", "block_fp8_gemm_v2"),
    "scale_tma": ("kernel_scale_tma", "block_fp8_gemm_scale_tma"),
    "shortk": ("kernel_shortk", "shortk"),
}

@lru_cache(maxsize=None)
def kernel(name):
    module, function = KERNELS[name]
    return getattr(import_module(f".kernels.{module}", __package__), function)

def run(entry, args):
    return kernel(entry["impl"])(*args, dict(entry["config"]))
