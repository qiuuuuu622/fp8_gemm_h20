# Summary tables for runs/bench_final.csv
import csv, math, sys

rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else "runs/bench_final.csv")))
gm = lambda v: math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")
bins = [("M<=128", lambda m: m <= 128), ("128<M<=512", lambda m: 128 < m <= 512),
        ("512<M<4096", lambda m: 512 < m < 4096), ("M>=4096", lambda m: m >= 4096), ("all", lambda m: True)]
print(f"{'M range':12}{'n':>4}{'dsp/DG':>9}{'min':>7}{'max':>7}{'DG wins':>9}{'c3/DG':>8}{'dsp/base':>10}{'c9 used':>9}")
for lab, f in bins:
    sel = [r for r in rows if f(int(r["M"]))]
    if not sel:
        continue
    d = [float(r["dsp_vs_dg"]) for r in sel]
    print(f"{lab:12}{len(sel):>4}{gm(d):>9.3f}{min(d):>7.3f}{max(d):>7.3f}{sum(x < 1 for x in d):>9}"
          f"{gm([float(r['c3_vs_dg']) for r in sel]):>8.3f}{gm([float(r['dsp_vs_base']) for r in sel]):>10.3f}"
          f"{sum(r['impl'] == 'c9' for r in sel):>9}")
print("\nper layer (dsp/DG geomean, worst):")
for lay in dict.fromkeys(r["layer"] for r in rows):
    d = [(float(r["dsp_vs_dg"]), r["M"]) for r in rows if r["layer"] == lay]
    w = min(d)
    print(f"  {lay:18} {gm([x for x, _ in d]):.3f}  worst {w[0]:.3f} @M={w[1]}")
print("\nDG still faster (dsp/DG < 0.98):")
for r in sorted(rows, key=lambda r: float(r["dsp_vs_dg"])):
    if float(r["dsp_vs_dg"]) < 0.98:
        print(f"  {r['layer']:18}N={r['N']:>5} K={r['K']:>4} M={r['M']:>5} {r['impl']} {r['dsp_vs_dg']}"
              f"  ({r['dsp_tf']} vs {r['dg_tf']} TF)")
print("max err dsp/dg:", max(float(r["err_dsp"]) for r in rows), max(float(r["err_dg"]) for r in rows))
