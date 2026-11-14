"""Table of the K search (code/ksearch.sh). usage: ksearch_report.py CORPUS SEED [OUT.json]"""
import collections, glob, json, re, sys
import numpy as np
H = json.load(open(sys.argv[1])); seed = sys.argv[2]; tot = collections.Counter()
for d in H["docs"].values():
    for i in d["ids"]:
        if i.startswith("c:all:"): tot[i[6:].split(" | ")[0]] += 1
rows = []
for f in sorted(glob.glob(f"fits/ksearch/K*_s{seed}.json"), key=lambda x: int(re.search(r"K(\d+)_", x)[1])):
    K = int(re.search(r"K(\d+)_", f)[1]); p = f[:-5]; M = json.load(open(f)); n = np.load(p + "_n.npy").astype(float); g = collections.defaultdict(list)
    for i, x in enumerate(M["nodes"]):
        if x.startswith("c:all:") and not x[6:].startswith(("class ", "any condition")): g[x.split(" | ")[0][6:]].append(i)
    big = [c for c in g if tot[c] >= 100]; N = np.array([n[g[c]].sum(0) for c in big]); T = N[:, 1:] / N[:, 1:].sum(1, keepdims=True)
    ne = np.exp(-(T * np.log(np.clip(T, 1e-12, 1))).sum(1)); sh = n[[i for c in g for i in g[c]]].sum(0); sh = sh / sh.sum()
    r = {"K": K, "iterations": M["iterations"], "largest": float(np.median(T.max(1))), "one_network": float(np.mean(T.max(1) >= 0.9)), "effective": float(np.median(ne)),
         "unused": int((sh[1:] < 0.5 / K).sum()), "smallest_share": float(sh[1:].min())}
    try:
        d = json.load(open(f"fits/cv/calib_K{K + 1}_ks_s{seed}_cv.json")); r["cv"] = d["ll_per_peak"] - d["own_map"]; r["cv_iters"] = min(x["iterations"] for x in d["folds"])
        c = json.load(open(f"fits/cv/calib_K{K + 1}_ks_s{seed}_cell.json"))["cells"]; P = sum(x["peaks"] for x in c)
        r["cell"] = sum((x["model"] - x["window"]) * x["peaks"] for x in c) / P; r["cell_won"] = float(np.mean([x["model"] > x["window"] for x in c]))
    except Exception as e: r["error"] = repr(e)  # noqa: BLE001
    rows.append(r)
print("K | by paper gain | held-out measure gain (cells won) | median largest loading | >=0.9 on one | effective networks | unused (< half an even share) | fewest iterations in a fold")
for r in rows:
    print(f"{r['K']:2d} | {r.get('cv', float('nan')):+.4f} | {r.get('cell', float('nan')):+.4f} ({r.get('cell_won', float('nan')):.2f}) | {r['largest']:.2f} | {r['one_network']:.2f} | {r['effective']:.1f} | {r['unused']} | {r.get('cv_iters')}")
if len(sys.argv) > 3: json.dump(rows, open(sys.argv[3], "w"), indent=1)
