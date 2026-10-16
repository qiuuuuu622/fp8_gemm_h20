# Mean per-token NLL of fixed texts under both servers (prefill only, teacher-forced). Equal NLL => equivalent numerics.
# python compare_nll.py <port_ref> <port_test>
import sys

from compare_logprob import score

TEXTS = [
    "The Industrial Revolution began in Great Britain in the late 18th century and spread to Western Europe and "
    "North America. It marked a shift from hand production methods to machines, new chemical manufacturing and iron "
    "production processes, improved efficiency of water power, the increasing use of steam power, and the development "
    "of machine tools. Textiles were the dominant industry in terms of employment, value of output and capital invested.",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n"
    "        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n"
    "        else:\n            hi = mid - 1\n    return -1\n\nprint(binary_search([1, 3, 5, 7, 9, 11], 7))\n",
    "杭州西湖位于浙江省杭州市西部，是中国著名的风景名胜区。西湖三面环山，面积约六点三九平方千米，湖中被孤山、白堤、苏堤、杨公堤分隔，"
    "形成外西湖、西里湖、北里湖、小南湖及岳湖等五片水面。自古以来，西湖就以其秀丽的湖光山色和众多的名胜古迹闻名中外。",
    "To compute the eigenvalues of a 2x2 matrix [[a, b], [c, d]], solve the characteristic equation "
    "lambda^2 - (a + d) lambda + (ad - bc) = 0. The discriminant (a - d)^2 + 4bc determines whether the eigenvalues "
    "are real and distinct, repeated, or a complex conjugate pair.",
]

ref_port, test_port = sys.argv[1], sys.argv[2]
tot_r = tot_t = n_all = 0
for t in TEXTS:
    lr, _ = score(ref_port, t)
    lt, _ = score(test_port, t)
    pr = [a[0] for a in lr[1:] if a[0] is not None]
    pt = [b[0] for b in lt[1:] if b[0] is not None]
    n = min(len(pr), len(pt))
    nr, nt = -sum(pr[:n]) / n, -sum(pt[:n]) / n
    tot_r += -sum(pr[:n]); tot_t += -sum(pt[:n]); n_all += n
    print(f"{t[:30]!r:34} tokens={n:4d} NLL ref={nr:.4f} test={nt:.4f} diff={nt - nr:+.4f}")
print(f"overall NLL ref={tot_r / n_all:.4f} test={tot_t / n_all:.4f} diff={(tot_t - tot_r) / n_all:+.4f} "
      f"(rel {(tot_t - tot_r) / tot_r * 100:+.2f}%)")
