"""Table of the calibration runs (code/calib.sh): usage calib_report.py GLOB_PREFIX, e.g. fits/cv/calib_K11_"""
import json, glob, os, sys
pre = sys.argv[1]; rows = []
for f in sorted(glob.glob(pre + "*_cv.json")):
    name = f[len(pre):-8]; d = json.load(open(f)); cf = f.replace("_cv.json", "_cell.json")
    if not os.path.exists(cf): continue
    c = json.load(open(cf))["cells"]; P = sum(x["peaks"] for x in c); g = lambda fn: sum(fn(x) * x["peaks"] for x in c) / P
    bb = max(d["blend"].items(), key=lambda kv: kv[1]); cb = {l: g(lambda x: x["blend"][l]) for l in c[0]["blend"]}; cbb = max(cb.items(), key=lambda kv: kv[1])
    rows.append((name, d["ll_per_peak"], d["own_map"], bb, g(lambda x: x["model"]), g(lambda x: x["window"]), cbb, sum(x["model"] > x["window"] for x in c) / len(c), sum(fd["iterations"] for fd in d["folds"]) / 5))
print("name | by paper: as fitted, window, best blend | held-out measure: as fitted, window, best blend, cells won as fitted | iterations")
for r in sorted(rows, key=lambda r: -(r[1] + r[4])): print(f"{r[0]:20s} | {r[1]:.4f} {r[2]:.4f} {r[3][1]:.4f}@{r[3][0]} | {r[4]:.4f} {r[5]:.4f} {r[6][1]:.4f}@{r[6][0]} {r[7]:.2f} | {r[8]:.0f}")
