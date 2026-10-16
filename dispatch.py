# Unified entry: per-(N,K,M) table picks c3 (opt2, Triton tl.dot), c9 (gl2) or c11 (gl5, Gluon pairwise wgmma).
# Table: runs/dispatch.json  {"N,K": {"M": {"impl": "c3"|"c9"|"c11", "cfg": {...}}}}; lookup = log-nearest M.
import json, math, os
import opt2, gl2, gl5

IMPLS = {"c3": opt2.matmul, "c9": gl2.matmul, "c11": gl5.matmul}
_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "dispatch.json")
_TABLE = json.load(open(_FILE)) if os.path.exists(_FILE) else {}
_LOOKUP = os.environ.get("DISPATCH_LOOKUP", "nearest")  # nearest (log) | ceil


def pick(M, N, K):
    t = _TABLE.get(f"{N},{K}")
    if not t:
        return "c3", None  # untuned shape: c3 with its own fallback config
    if _LOOKUP == "ceil":
        # smallest tuned M >= actual M: its tile counts cover M without under-provisioning the grid
        ge = [int(m) for m in t if int(m) >= M]
        key = str(min(ge)) if ge else max(t, key=int)
    else:
        lm = math.log(max(M, 1))
        key = min(t, key=lambda m: abs(math.log(int(m)) - lm))
    e = t[key]
    return e["impl"], e["cfg"]


def matmul(A, B, As, Bs):
    """A[M,K] fp8, B[N,K] fp8, As[M,K/128] f32, Bs[N/128,K/128] f32 -> C[M,N] bf16."""
    impl, cfg = pick(A.shape[0], B.shape[0], A.shape[1])
    return IMPLS[impl](A, B, As, Bs, cfg)
