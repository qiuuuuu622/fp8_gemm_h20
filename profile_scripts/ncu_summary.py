# Summarize ncu reports side by side: python profile/ncu_summary.py <report1.ncu-rep> [...]
import csv, io, subprocess, sys

NCU = "/usr/local/cuda/bin/ncu"
METRICS = {
    "time_us": "gpu__time_duration.sum",
    "grid": "launch__grid_size",
    "block": "launch__block_size",
    "regs": "launch__registers_per_thread",
    "smem_KB": "launch__shared_mem_per_block_dynamic",
    "warps_act%": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm_thru%": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_thru%": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_hit%": "lts__t_sector_hit_rate.pct",
    "gmma_act%": "sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm_active%": "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l2_MB": "lts__t_bytes.sum",
    "dram_MB": "dram__bytes.sum",
    "inst": "smsp__inst_executed.sum",
}
STALLS = ["long_scoreboard", "barrier", "wait", "math_pipe_throttle", "mio_throttle", "short_scoreboard",
          "lg_throttle", "no_instruction", "dispatch_stall", "selected", "not_selected", "sleeping",
          "membar", "branch_resolving", "drain", "imc_miss", "misc", "tex_throttle"]


def raw(rep, names):
    out = subprocess.run([NCU, "-i", rep, "--page", "raw", "--csv", "--metrics", ",".join(names)],
                         capture_output=True, text=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    h, u, v = rows[0], rows[1], rows[-1]
    return {h[i]: (v[i], u[i]) for i in range(len(h))}


def num(x):
    try:
        return float(x.replace(",", ""))
    except ValueError:
        return None


for rep in sys.argv[1:]:
    d = raw(rep, list(METRICS.values()))
    parts = []
    for k, m in METRICS.items():
        val, unit = d.get(m, ("?", ""))
        f = num(val)
        if f is not None:
            if unit in ("usecond",) or k == "time_us" and unit == "nsecond":
                f = f / 1000 if unit == "nsecond" else f
            if k.endswith("_MB"):
                f = f / (1e6 if unit == "byte" else 1 if unit == "Mbyte" else 1e-3 if unit == "Kbyte" else 1e-3 * 1e-3 if unit == "Gbyte" else 1)
                f = f if unit != "Gbyte" else num(val) * 1e3
            if k == "smem_KB":
                f = f / 1024 if unit == "byte" else f
            val = f"{f:.1f}" if abs(f) < 1e6 else f"{f:.3g}"
        parts.append(f"{k}={val}")
    print(rep.split("/")[-1].replace(".ncu-rep", ""))
    print("   " + " ".join(parts))
    st = raw(rep, [f"smsp__average_warp_latency_issue_stalled_{s}.ratio" for s in STALLS] +
             ["smsp__average_warps_issue_stalled_" + s + "_per_issue_active.ratio" for s in STALLS])
    vals = []
    for s in STALLS:
        v = st.get(f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio", ("", ""))[0]
        f = num(v)
        if f:
            vals.append((f, s))
    vals.sort(reverse=True)
    print("   stalls(per issue): " + " ".join(f"{s}={f:.2f}" for f, s in vals[:7]))
