"""How concentrated conditions are in windowed fits. usage: concentration.py CORPUS FIT_PREFIX [FIT_PREFIX ...]
Among conditions with >= 100 contrasts: loadings over the learned networks (flat network left out, pooled over direction children)."""
import collections, json, sys
import numpy as np
H = json.load(open(sys.argv[1])); tot = collections.Counter()
for d in H["docs"].values():
    for i in d["ids"]:
        if i.startswith("c:all:"): tot[i[6:].split(" | ")[0]] += 1
for p in sys.argv[2:]:
    n = np.load(p + "_n.npy").astype(float); M = json.load(open(p + ".json")); g = collections.defaultdict(list)
    for i, x in enumerate(M["nodes"]):
        if x.startswith("c:all:") and not x[6:].startswith(("class ", "any condition")): g[x.split(" | ")[0][6:]].append(i)
    big = [c for c in g if tot[c] >= 100]; N = np.array([n[g[c]].sum(0) for c in big]); T = N[:, 1:] / N[:, 1:].sum(1, keepdims=True); flat = N[:, 0] / N.sum(1)
    ne = np.exp(-(T * np.log(np.clip(T, 1e-12, 1))).sum(1)); sh = n[[i for c in g for i in g[c]]].sum(0); sh = sh / sh.sum(); ad = T[big.index("alzheimer disease")]
    print(f"{p.split('/')[-1]:14s} iterations {M['iterations']:4d}: median largest loading {np.median(T.max(1)):.2f}; share >= 0.9 on one network {np.mean(T.max(1) >= 0.9):.2f}; "
          f"median effective networks {np.median(ne):.1f}; unused networks (< 2%) {int((sh[1:] < 0.02).sum())}; median flat share {np.median(flat):.2f}; Alzheimer top 3 {np.sort(ad)[::-1][:3].round(2).tolist()}")
