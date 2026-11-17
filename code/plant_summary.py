"""Summarise plant_atlas.py and ale_plant.py (2026-11-16).
Atlas model, per model and k: threshold = 95th percentile of the largest log ratio (planted cell against its class) over
the null draws (F = 0). False positive rate = share of null draws above the threshold of the OTHER null draws (leave one
out). Detection at F > 0 = largest log ratio above the threshold and within 12 mm of the blob centre. Also the known-location
test (log ratio at the blob centre against the null draws' values there), and the page's current statistic (share of
peaks moving against the class) with a fixed cut of 0.15. ALE: detection = significant voxel within 12 mm of the blob
(cluster-level FWE 0.05); at F = 0, how often a significant voxel lies within 12 mm of the (unplanted) centre, and how
often any cluster appears.
usage: plant_summary.py PLANT_DIR OUT.json"""
import glob, json, sys, collections, os
import numpy as np
d = sys.argv[1]; R = collections.defaultdict(list)
for f in glob.glob(f"{d}/*_k*.json"):
    x = json.load(open(f)); R[x["model"], x["k"], x["f"] if x["f"] == 0 else (x["f"], x.get("sd", 6.0), x.get("h", 1.0))].append(x)
ale = {json.load(open(f))["file"]: json.load(open(f)) for f in glob.glob(f"{d}/ale/*.json")}
out = {"atlas": {}, "ale": {}}
for m in sorted({k[0] for k in R}):
    for k in sorted({k[1] for k in R}):
        nul = R[m, k, 0.0]
        if len(nul) < 5: continue
        mx = np.array([x["max_log_ratio"] for x in nul]); ac = np.array([x["log_ratio_at_centre"] for x in nul]); thr = np.quantile(mx, 0.95); thc = np.quantile(ac, 0.95)
        fp = np.mean([mx[i] > np.quantile(np.delete(mx, i), 0.95) for i in range(len(mx))])
        row = {"null_draws": len(nul), "threshold_max_log_ratio": round(float(thr), 3), "false_positive_rate_loo": round(float(fp), 3),
               "page_rule_tv_ge_0.15_null_rate": round(float(np.mean([x["tv_vs_class"] >= 0.15 for x in nul])), 3), "null_tv_median": round(float(np.median([x["tv_vs_class"] for x in nul])), 3)}
        for f in sorted({kk[2] for kk in R if kk[0] == m and kk[1] == k and kk[2] != 0.0}):
            s = R[m, k, f]
            if not s: continue
            row["F%g sd%g h%g" % f] = {"draws": len(s), "detected_anywhere_and_at_blob": round(float(np.mean([x["max_log_ratio"] > thr and x["max_distance_mm"] <= 12 for x in s])), 2),
                            "detected_known_location": round(float(np.mean([x["log_ratio_at_centre"] > thc for x in s])), 2), "page_rule_tv_ge_0.15": round(float(np.mean([x["tv_vs_class"] >= 0.15 for x in s])), 2),
                            "median_max_log_ratio": round(float(np.median([x["max_log_ratio"] for x in s])), 3), "median_distance_mm": round(float(np.median([x["max_distance_mm"] for x in s])), 1)}
        out["atlas"][f"{m} k={k}"] = row
import re
groups = collections.defaultdict(list)
for n, v in ale.items():
    mm = re.match(r"k(\d+)_f([\d.]+)_d\d+(_sd[\d.]+_h[\d.]+)?\.json", n); groups[int(mm.group(1)), mm.group(2) + (mm.group(3) or "")].append(v)
for k in sorted({g[0] for g in groups}):
    row = {}
    for key in sorted(g[1] for g in groups if g[0] == k):
        a = groups[k, key]
        if a: row["F" + key] = {"draws": len(a), "significant_at_blob": round(float(np.mean([v["detected"] for v in a])), 2), "any_cluster": round(float(np.mean([v["n_clusters"] > 0 for v in a])), 2)}
    out["ale"][f"k={k}"] = row
json.dump(out, open(sys.argv[2], "w"), indent=1); print(json.dumps(out, indent=1))
