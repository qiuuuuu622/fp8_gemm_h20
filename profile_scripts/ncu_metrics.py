# python profile/ncu_metrics.py "<m1,m2,...>" rep1.ncu-rep [rep2 ...]  -> one line per report
import csv, io, subprocess, sys

NCU = "/usr/local/cuda/bin/ncu"
metrics = sys.argv[1].split(",")
for rep in sys.argv[2:]:
    out = subprocess.run([NCU, "-i", rep, "--page", "raw", "--csv", "--metrics", ",".join(metrics)],
                         capture_output=True, text=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    h, u, v = rows[0], rows[1], rows[-1]
    col = {name: i for i, name in enumerate(h)}
    vals = []
    for m in metrics:
        i = col.get(m)
        vals.append(f"{m.split('.')[0].replace('sm__', '').replace('smsp__', '')}={v[i]}{u[i] if u[i] not in ('', 'inst') else ''}"
                    if i is not None else f"{m}=NA")
    print(rep.split("/")[-1].replace(".ncu-rep", ""))
    print("   " + "  ".join(vals))
