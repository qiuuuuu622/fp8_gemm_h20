"""CPU-only profile loading and deterministic dispatch selection.

The configuration files are the runtime source of truth. Historical tables in
runs/ are immutable experiment evidence, not an additional runtime dependency.
"""
import copy
import json
import math
from functools import lru_cache
from pathlib import Path
from .engine import KERNELS

_CONFIG_DIR = Path(__file__).with_name("configs")
_PROFILES = ("35b_tp1", "122b_tp4")

def available_profiles():
    return _PROFILES

def token_tile(entry):
    c = entry["config"]
    return c.get("BLOCK_T", c.get("BLOCK_SIZE_M", c.get("BM", c.get("BT"))))

def validate_config(data):
    if data["schema_version"] != 1 or data["block_size"] != [128, 128]:
        raise ValueError("Unsupported schema or quantization block size")
    if data["m_range"] != [1, 16384]:
        raise ValueError("Profiles must declare the validated M=1..16384 range")
    expected = {f"{s['N']},{s['K']}" for s in data["shapes"]}
    if set(data["entries"]) != expected:
        raise ValueError("Shape inventory and dispatch entries disagree")
    policy = data["policy"]
    if policy["kind"] not in ("log_nearest", "nearby_equal_tiles"):
        raise ValueError("Unsupported dispatch policy")
    if policy["kind"] == "nearby_equal_tiles":
        if policy["max_distance"] < 0 or set(data["fallback_entries"]) != expected:
            raise ValueError("Incomplete bounded interpolation/fallback policy")
    for field in ("entries", "fallback_entries"):
        for nk, table in data.get(field, {}).items():
            n, k = map(int, nk.split(","))
            if n % 128 or k % 128 or not table:
                raise ValueError(f"Invalid shape/table: {nk}")
            for m, entry in table.items():
                if not 1 <= int(m) <= 16384 or entry["impl"] not in KERNELS:
                    raise ValueError(f"Invalid entry: {nk}/{m}")
                c = entry["config"]
                bt = token_tile(entry)
                bw = c.get("BLOCK_W", c.get("BLOCK_SIZE_N", c.get("BN")))
                if not bt or bt & (bt - 1) or not bw or 128 % bw:
                    raise ValueError(f"Invalid tile: {nk}/{m}")
                split = c.get("SPLIT", c.get("SPLIT_K", 1))
                if split < 1:
                    raise ValueError(f"Invalid split-K: {nk}/{m}")
                if entry["impl"] == "shortk" and k != 256:
                    raise ValueError("shortk requires K=256")
                if c.get("MODE") == 3 and (c["NUM_STAGES"] < 4 or (k // 128) % (4 * split)):
                    raise ValueError("MODE=3 needs complete groups of four K tiles")
                if field == "entries" and policy["kind"] == "nearby_equal_tiles":
                    if not math.isfinite(entry["anchor_us"]) or entry["anchor_us"] <= 0:
                        raise ValueError("Measured anchor latency must be positive")

@lru_cache(maxsize=None)
def _profile(name):
    if name not in _PROFILES:
        raise ValueError(f"Unknown profile {name!r}; choose one of {_PROFILES}")
    data = json.loads((_CONFIG_DIR / f"{name}.json").read_text())
    validate_config(data)
    return data

def get_profile(name):
    return copy.deepcopy(_profile(name))

def supports(profile, m, n, k):
    data = _profile(profile)
    return data["m_range"][0] <= m <= data["m_range"][1] and f"{n},{k}" in data["entries"]

@lru_cache(maxsize=4096)
def _select(profile, m, n, k):
    data = _profile(profile)
    if not supports(profile, m, n, k):
        raise ValueError(f"{profile} does not cover M,N,K={m},{n},{k}")
    entries = data["entries"][f"{n},{k}"]
    if str(m) in entries:
        return entries[str(m)], {"method": "exact", "anchor": m}
    policy = data["policy"]
    if policy["kind"] == "nearby_equal_tiles":
        matches = []
        for a, entry in entries.items():
            anchor, bt = int(a), token_tile(entry)
            if entry["impl"] in policy["exact_only"]:
                continue
            if abs(m - anchor) <= policy["max_distance"] and (m + bt - 1) // bt == (anchor + bt - 1) // bt:
                matches.append((entry["anchor_us"], abs(math.log(m / anchor)), anchor, entry))
        if matches:
            _, _, anchor, entry = min(matches, key=lambda x: x[:3])
            return entry, {"method": "equal_tile_count", "anchor": anchor}
        entries = data["fallback_entries"][f"{n},{k}"]
        method = "baseline_fallback"
    else:
        method = "log_nearest"
    anchor = min(entries, key=lambda a: abs(math.log(int(a) / m)))
    return entries[anchor], {"method": method, "anchor": int(anchor)}

def select(profile, m, n, k):
    """Return a copy of the selected entry and its routing explanation."""
    return copy.deepcopy(_select(profile, m, n, k))
