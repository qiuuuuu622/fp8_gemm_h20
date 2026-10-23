"""Legacy 35B research API. New applications should import fp8_gemm_h20.

FP8MM_PROFILE switches profiles. DISPATCH_LOOKUP=ceil is retained only for
historical 35B experiments; the public package uses the frozen profile policy.
"""
import os
from fp8_gemm_h20 import matmul as _matmul, select, supports
from fp8_gemm_h20.config import get_profile
from fp8_gemm_h20.engine import kernel

_PROFILE = os.environ.get("FP8MM_PROFILE", "35b_tp1")
_ALIASES = {"triton": "c3", "gluon": "c11"}
_TABLE = {nk: {m: {"impl": _ALIASES.get(e["impl"], e["impl"]), "cfg": e["config"]}
               for m, e in rows.items()}
          for nk, rows in get_profile(_PROFILE)["entries"].items()}
_LOOKUP = os.environ.get("DISPATCH_LOOKUP", "nearest")

def pick(M, N, K):
    if not supports(_PROFILE, M, N, K):
        if _PROFILE != "35b_tp1":
            raise ValueError("Unsupported shape for 122b_tp4")
        return "c3", None
    if _LOOKUP == "ceil" and _PROFILE == "35b_tp1":
        table = _TABLE[f"{N},{K}"]
        ge = [int(m) for m in table if int(m) >= M]
        anchor = str(min(ge)) if ge else max(table, key=int)
        return table[anchor]["impl"], table[anchor]["cfg"]
    entry, _ = select(_PROFILE, M, N, K)
    return _ALIASES.get(entry["impl"], entry["impl"]), entry["config"]

def matmul(A, B, As, Bs):
    if supports(_PROFILE, A.shape[0], B.shape[0], A.shape[1]) and _LOOKUP != "ceil":
        return _matmul(A, B, As, Bs, profile=_PROFILE)
    impl, cfg = pick(A.shape[0], B.shape[0], A.shape[1])
    if impl == "c3":
        import opt2
        return opt2.matmul(A, B, As, Bs, cfg)
    return kernel({"c11": "gluon"}.get(impl, impl))(A, B, As, Bs, cfg)

# Historical startup_cost.py calls this registry directly.
def _legacy_kernel(name):
    def invoke(A, B, As, Bs, cfg=None):
        return kernel(name)(A, B, As, Bs, cfg)
    return invoke

IMPLS = {"c3": _legacy_kernel("triton"), "c11": _legacy_kernel("gluon")}

def _legacy_c9(A, B, As, Bs, cfg=None):
    from gl2 import matmul as c9
    return c9(A, B, As, Bs, cfg)

IMPLS["c9"] = _legacy_c9
