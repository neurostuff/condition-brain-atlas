"""K search over several seeds: held-out gains, unused networks, concentration, and how well networks reappear across seeds.
usage: ksearch_seeds.py CORPUS OUT.json SEED [SEED ...]   (fits from code/ksearch.sh)
Stability: for every pair of seeds, networks are matched one to one (Hungarian) on the correlation of their maps phi
(flat network left out); reported: mean matched r, and the share of networks matched at r >= 0.5."""
import collections, itertools, json, os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
H = json.load(open(sys.argv[1])); out = sys.argv[2]; seeds = sys.argv[3:]; tot = collections.Counter()
for d in H["docs"].values():
    for i in d["ids"]:
        if i.startswith("c:all:"): tot[i[6:].split(" | ")[0]] += 1
rows = []
for K in range(5, 61):
    S = [s for s in seeds if os.path.exists(f"fits/ksearch/K{K}_s{s}.json") and os.path.exists(f"fits/cv/calib_K{K + 1}_ks_s{s}_cell.json")]
    if len(S) < 2: continue
    cv, cell, unused, one, eff, phis = [], [], [], [], [], []
    for s in S:
        d = json.load(open(f"fits/cv/calib_K{K + 1}_ks_s{s}_cv.json")); cv.append(d["ll_per_peak"] - d["own_map"])
        c = json.load(open(f"fits/cv/calib_K{K + 1}_ks_s{s}_cell.json"))["cells"]; P = sum(x["peaks"] for x in c); cell.append(sum((x["model"] - x["window"]) * x["peaks"] for x in c) / P)
        p = f"fits/ksearch/K{K}_s{s}"; M = json.load(open(p + ".json")); n = np.load(p + "_n.npy").astype(float); g = collections.defaultdict(list)
        for i, x in enumerate(M["nodes"]):
            if x.startswith("c:all:") and not x[6:].startswith(("class ", "any condition")): g[x.split(" | ")[0][6:]].append(i)
        big = [c2 for c2 in g if tot[c2] >= 100]; N = np.array([n[g[c2]].sum(0) for c2 in big]); T = N[:, 1:] / N[:, 1:].sum(1, keepdims=True)
        sh = n[[i for c2 in g for i in g[c2]]].sum(0); sh = sh / sh.sum(); unused.append(int((sh[1:] < 0.5 / K).sum())); one.append(float(np.mean(T.max(1) >= 0.9)))
        eff.append(float(np.median(np.exp(-(T * np.log(np.clip(T, 1e-12, 1))).sum(1))))); ph = np.load(p + "_phi.npy").astype(float)[1:]; phis.append((ph - ph.mean(1, keepdims=True)) / ph.std(1, keepdims=True))
    mr, ok = [], []
    for a, b in itertools.combinations(range(len(S)), 2):
        R = phis[a] @ phis[b].T / phis[a].shape[1]; i, j = linear_sum_assignment(-R); mr.append(R[i, j].mean()); ok.append((R[i, j] >= 0.5).mean())
    rows.append({"K": K, "seeds": S, "cv": cv, "cell": cell, "unused": unused, "one_network": one, "effective": eff, "match_r": float(np.mean(mr)), "matched": float(np.mean(ok))})
json.dump(rows, open(out, "w"), indent=1)
print("K seeds | by paper: mean (min to max) | held-out measure: mean (min to max) | unused per seed | >=0.9 on one (mean) | effective (mean) | matched r | networks matched r>=0.5 | stable networks")
for r in rows:
    print(f"{r['K']:2d} {len(r['seeds'])} | {np.mean(r['cv']):+.4f} ({min(r['cv']):+.4f} to {max(r['cv']):+.4f}) | {np.mean(r['cell']):+.4f} ({min(r['cell']):+.4f} to {max(r['cell']):+.4f}) | {r['unused']} | "
          f"{np.mean(r['one_network']):.2f} | {np.mean(r['effective']):.1f} | {r['match_r']:.2f} | {r['matched']:.2f} | {r['matched'] * r['K']:.1f}")
