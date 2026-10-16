# Chain to the system sitecustomize, then load the opt-in fp8mm patch (no-op unless FP8MM_ENABLE=1).
import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _p in sys.path:
    if os.path.abspath(_p or ".") == _here:
        continue
    _cand = os.path.join(_p, "sitecustomize.py")
    if os.path.isfile(_cand):
        _spec = importlib.util.spec_from_file_location("_system_sitecustomize", _cand)
        _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
        break

import fp8mm_sglang_patch  # noqa: E402,F401
