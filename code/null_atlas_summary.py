"""Compare the atlas page's numbers on the real corpus with the same numbers after shuffling conditions among papers
(null_atlas.py). Statistic for the "any" null: share of predicted peaks that move against the measure's usual pattern;
for the "class" null: share that move against the condition's class. Cells are compared within bins of how many
contrasts they hold (the ridge shrinks small cells, so their nulls differ). Per real cell: an empirical p-value against
the null cells of its bin; Benjamini-Hochberg over cells. Per threshold: the expected share of false highlights
(mean null count above the threshold per fit divided by the real count).
usage: null_atlas_summary.py NULL_DIR OUT.json"""
import glob, json, sys, collections
import numpy as np
d = sys.argv[1]; real = json.load(open(f"{d}/real_0.json"))["cells"]
BINS = [(1, 1), (2, 4), (5, 14), (15, 49), (50, 10 ** 9)]; binof = lambda n: next(i for i, (a, b) in enumerate(BINS) if a <= n <= b)
out = {"real_cells": len(real)}
for mode, col, what in (("any", 4, "moves against the measure's usual pattern"), ("class", 5, "moves against the condition's class")):
    fits = [json.load(open(f))["cells"] for f in sorted(glob.glob(f"{d}/{mode}_*.json"))]
    if not fits: continue
    nf = len(fits); null = [x for f in fits for x in f]; r = {"fits": nf, "statistic": what, "by_size": [], "thresholds": []}
    nb = collections.defaultdict(list)
    for x in null: nb[binof(x[3])].append(x[col])
    for i, (a, b) in enumerate(BINS):
        rv = np.array([x[col] for x in real if a <= x[3] <= b]); nv = np.array(nb[i])
        if not len(rv) or not len(nv): continue
        q95 = float(np.quantile(nv, 0.95))
        r["by_size"].append({"contrasts": f"{a}-{b}" if b < 10 ** 9 else f"{a}+", "real_cells": int(len(rv)), "null_cells_per_fit": round(len(nv) / nf, 1),
                             "real_median": round(float(np.median(rv)), 3), "null_median": round(float(np.median(nv)), 3), "null_95th": round(q95, 3),
                             "real_above_null_95th": round(float((rv > q95).mean()), 3)})
    for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
        rc = sum(x[col] >= t for x in real); nc = sum(x[col] >= t for x in null) / nf
        r["thresholds"].append({"share_moved_at_least": t, "real_cells": int(rc), "null_cells_per_fit": round(nc, 1), "expected_false_share": round(min(1.0, nc / rc), 3) if rc else None})
    ps = []
    for x in real:
        nv = np.asarray(nb[binof(x[3])]); ps.append((1 + (nv >= x[col]).sum()) / (1 + len(nv)))
    ps = np.array(ps); o = np.argsort(ps); m = len(ps); bh = np.minimum.accumulate((ps[o] * m / np.arange(1, m + 1))[::-1])[::-1]; q = np.empty(m); q[o] = bh
    sig = [real[i] + [round(float(ps[i]), 4), round(float(q[i]), 4)] for i in np.where(q < 0.05)[0]]
    r["cells_q_below_0.05"] = len(sig); r["cells_p_below_0.05"] = int((ps < 0.05).sum()); r["min_possible_p"] = round(1 / (1 + min(len(v) for v in nb.values())), 5)
    r["by_size_significant"] = {f"{a}-{b}": int(sum(1 for i in np.where(q < 0.05)[0] if a <= real[i][3] <= b)) for a, b in BINS}
    r["top_significant"] = sorted(sig, key=lambda x: -x[col])[:25]
    out[mode] = r
json.dump(out, open(sys.argv[2], "w"), indent=1); print(json.dumps(out, indent=1)[:6000])
