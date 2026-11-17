"""Add the null and planted-signal simulations to the atlas page's data (2026-11-16).

Per published cell (condition x measure x direction with >= 1 contrast): the empirical p-value of the page's two numbers
against the same numbers after shuffling diagnoses among papers (null_atlas.py), within bins of contrast count, exactly as
null_atlas_summary.py computes them; written as c.np[j][k] = [p against the measure's usual pattern, p against the class],
or null for an untested cell. meta.validation holds the summaries the Method page shows: the null simulation's per-size
rates and false-highlight shares, the cells above every shuffled cell of their size in both tests, and the planted-signal detection rates (plant_summary.py, ale_roi_plant.py outputs).
Checks that the real refit reproduces every page cell's numbers before writing.
usage: null_site.py NULL_DIR DATA_IN DATA_OUT NULL_SUMMARY PLANT_MID PLANT_MID_ALE_ROI PLANT_COMMON PLANT_COMMON_ALE_ROI"""
import ast, glob, json, sys
import numpy as np

nd, din, dout, nsum, pm, pmr, pc, pcr = sys.argv[1:9]
D = json.load(open(din)); real = json.load(open(f"{nd}/real_0.json"))["cells"]
BINS = [(1, 1), (2, 4), (5, 14), (15, 49), (50, 10 ** 9)]; binof = lambda n: next(i for i, (a, b) in enumerate(BINS) if a <= n <= b)
null = {}
for mode, col in (("any", 4), ("class", 5)):
    b = [[] for _ in BINS]
    for f in sorted(glob.glob(f"{nd}/{mode}_*.json")):
        for x in json.load(open(f))["cells"]: b[binof(x[3])].append(x[col])
    null[mode] = [np.sort(np.array(v)) for v in b]
pval = lambda mode, n, v: (1 + len(null[mode][binof(n)]) - np.searchsorted(null[mode][binof(n)], v, side="left")) / (1 + len(null[mode][binof(n)]))

M = D["meta"]["measures"]; DIRS = ["less", "more"]; idx = {c["name"]: i for i, c in enumerate(D["conditions"])}
for c in D["conditions"]: c["np"] = [[None, None] for _ in M]
seen = 0; beyond = []   # cells above every shuffled cell of their size bin in both tests
for name, m, d, n, tu, tc in real:
    c = D["conditions"][idx[name]]; j = M.index(m); k = DIRS.index(d); cell = c["cells"][j][k]
    assert cell[0] == n and abs(cell[1] - tu) < 1e-3 and abs(cell[2] - tc) < 1e-3, (name, m, d, cell[:3], n, tu, tc)
    c["np"][j][k] = [round(float(pval("any", n, tu)), 4), round(float(pval("class", n, tc)), 4)]; seen += 1
    if tu > null["any"][binof(n)][-1] and tc > null["class"][binof(n)][-1]: beyond.append([idx[name], j, k])
tested = sum(1 for c in D["conditions"] for p in c["cells"] for q in p if q[0] > 0)
assert seen == tested, (seen, tested)

S = json.load(open(nsum))
def plant(fa, fr, ks, mid):
    A = json.load(open(fa)); R = json.load(open(fr))["summary"]; R = ast.literal_eval(R) if isinstance(R, str) else R; rows = []
    for k in ks:
        for h in (0.5, 1):
            key = "F0.01 sd10 h%g" % h; a0 = A["atlas"][f"none k={k}"][key]; a6 = A["atlas"][f"w6 k={k}"][key]
            ale = A["ale"][f"k={k}"]["F0.01_sd10_h%g" % h]; roi = R[f"k={k}"]["f0.01 sd10 h%g" % h]
            rows.append({"k": k, "carry": "half" if h == 0.5 else "all", "whole": [a0["detected_anywhere_and_at_blob"], a6["detected_anywhere_and_at_blob"], ale["significant_at_blob"]],
                         "known": [a0["detected_known_location"], a6["detected_known_location"], roi["detected_vs_null_draws"], roi["detected_p_below_0.05"]]})
    return {"where": mid, "rows": rows, "ale_p05_null": {k: R[f"k={k}"]["null_p_below_0.05"] for k in ks},
            "ale_any_cluster_null": {k: A["ale"][f"k={k}"]["F0"]["any_cluster"] for k in ks}}
D["meta"]["validation"] = {
    "null": {"fits": S["any"]["fits"], "cells": S["real_cells"],
             "any": {"by_size": S["any"]["by_size"], "thresholds": S["any"]["thresholds"], "p05": S["any"]["cells_p_below_0.05"]},
             "class": {"by_size": S["class"]["by_size"], "thresholds": S["class"]["thresholds"], "p05": S["class"]["cells_p_below_0.05"]},
             "min_p": S["any"]["min_possible_p"], "beyond_all": beyond},
    "plant": [plant(pm, pmr, (3, 5, 10), "middling"), plant(pc, pcr, (5, 10, 20), "common")]}
json.dump(D, open(dout, "w"), separators=(",", ":"))
ps = np.array([q for c in D["conditions"] for p in c["np"] for q in p if q is not None])
print("cells", seen, "p<0.05 against usual", int((ps[:, 0] < 0.05).sum()), "against class", int((ps[:, 1] < 0.05).sum()), "both", int(((ps < 0.05).all(1)).sum()))
