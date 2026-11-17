"""ALE's known-location test on the plant_atlas.py draws (2026-11-16), the counterpart of the atlas model's "log ratio at
the blob centre". For each draw: the ALE map of the k planted studies (NiMARE ALE, approximate null, n = 30 each) and
its uncorrected voxel p-value; read at the blob centre (the 2 mm voxel nearest to it). Written per draw; the summary
compares each signal draw's ALE value at its centre with the null draws' values at theirs (same k), as for the atlas,
and also applies the textbook single-voxel rule p < 0.05.
usage: ale_roi_plant.py PLANT_DIR OUT.json"""
import glob, json, os, re, sys, collections
import numpy as np
from nimare.dataset import Dataset
from nimare.meta.cbma.ale import ALE
d = sys.argv[1]; rows = []
for fp in sorted(glob.glob(f"{d}/coords/*.json")):
    c = json.load(open(fp)); ex = c["experiments"]
    ds = Dataset({f"s{i}": {"contrasts": {"1": {"coords": {"space": "MNI", "x": [p[0] for p in v], "y": [p[1] for p in v], "z": [p[2] for p in v]}, "metadata": {"sample_sizes": [30]}}}} for i, v in enumerate(ex.values())})
    r = ALE(null_method="approximate").fit(ds); st = r.get_map("stat"); pm = r.get_map("p"); aff = st.affine
    ijk = np.rint(np.linalg.inv(aff) @ np.r_[c["centre"], 1])[:3].astype(int); sv = st.get_fdata(); pv = pm.get_fdata()
    m = re.match(r"k(\d+)_f([\d.]+)_d(\d+)(?:_sd([\d.]+)_h([\d.]+))?\.json", os.path.basename(fp))
    rows.append({"file": os.path.basename(fp), "k": int(m.group(1)), "f": float(m.group(2)), "sd": float(m.group(4) or 6), "h": float(m.group(5) or 1),
                 "ale_at_centre": float(sv[tuple(ijk)]), "p_at_centre": float(pv[tuple(ijk)])})
    print(rows[-1], flush=True)
res = {"draws": rows, "summary": {}}
by = collections.defaultdict(list)
for r in rows: by[r["k"], "null" if r["f"] == 0 else f"f{r['f']:g} sd{r['sd']:g} h{r['h']:g}"].append(r)
for k in sorted({r["k"] for r in rows}):
    nul = by[k, "null"]
    if not nul: continue
    v = np.array([r["ale_at_centre"] for r in nul]); thr = float(np.quantile(v, 0.95)); s = {"null_draws": len(nul), "null_p_below_0.05": round(float(np.mean([r["p_at_centre"] < 0.05 for r in nul])), 2), "threshold": thr}
    for key in sorted(x[1] for x in by if x[0] == k and x[1] != "null"):
        g = by[k, key]; s[key] = {"draws": len(g), "detected_vs_null_draws": round(float(np.mean([r["ale_at_centre"] > thr for r in g])), 2), "detected_p_below_0.05": round(float(np.mean([r["p_at_centre"] < 0.05 for r in g])), 2)}
    res["summary"][f"k={k}"] = s
json.dump(res, open(sys.argv[2], "w"), indent=1); print(json.dumps(res["summary"], indent=1))
