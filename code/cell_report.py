"""Summary of a pc_model.py cellcv result. usage: cell_report.py CELLS.json [BLEND]"""
import collections, json, sys
import numpy as np
d = json.load(open(sys.argv[1])); C = d["cells"]; lam = sys.argv[2] if len(sys.argv) > 2 else "0.35"
w = lambda cs: sum(c["peaks"] for c in cs)
g = lambda f, cs=C: sum(f(c) * c["peaks"] for c in cs) / w(cs)
dm = lambda c: c["model"] - c["window"]; db = lambda c: c["blend"][lam] - c["window"]
print(f"{len(C)} cells, {w(C)} peaks; model - window {g(dm):+.4f}; blend {lam} - window {g(db):+.4f}; class - window {g(lambda c: c['class'] - c['window']):+.4f}")
bm = collections.defaultdict(list)
for c in C: bm[c["measure"]].append(c)
print("by held-out measure: cells, peaks, model - window, blend - window, share of cells where the blend wins")
for m, cs in sorted(bm.items(), key=lambda x: -w(x[1])): print(f"  {m:20s} {len(cs):3d} {w(cs):6d}  {g(dm, cs):+.3f}  {g(db, cs):+.3f}  {np.mean([db(c) > 0 for c in cs]):.2f}")
f = lambda c: f"  {c['condition']} | {c['measure']} ({c['contrasts']} contrasts, {c['peaks']} peaks; {c['other_contrasts']} in other measures): model {dm(c):+.2f}, blend {db(c):+.3f}"
C2 = sorted(C, key=lambda c: -db(c))
print("largest gains:"); [print(f(c)) for c in C2[:12]]
print("largest losses:"); [print(f(c)) for c in C2[-8:]]
print("Alzheimer disease and mild cognitive impairment:"); [print(f(c)) for c in C if "cognitive impairment" in c["condition"] or c["condition"] == "alzheimer disease"]
for k in (100, 300):
    big = [c for c in C if c["peaks"] >= k]; print(f"cells with >= {k} peaks: {len(big)}; blend beats the window in {np.mean([db(c) > 0 for c in big]):.2f}; mean gain {g(db, big):+.4f}")
