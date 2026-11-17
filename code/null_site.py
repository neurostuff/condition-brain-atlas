"""Add the null and planted-signal simulations to the atlas page's data (2026-11-16).

Per published cell (condition x measure x direction with >= 1 contrast): the empirical p-value of the page's two numbers
against the same numbers after shuffling diagnoses among papers (null_atlas.py), within bins of contrast count, exactly as
null_atlas_summary.py computes them; written as c.np[j][k] = [p against the measure's usual pattern, p against the class],
or null for an untested cell. meta.validation holds the summaries the Method page shows: the null simulation's per-size
rates and false-highlight shares, the cells above every shuffled cell of their size in both tests, and the planted-signal detection rates (plant_summary.py, ale_roi_plant.py outputs).
With PEAK.npz (null_peak.py on refits that saved coefficients): the log-ratio tests, which the planted simulation found
sensitive. c.pk[j][k] = [p, p] for the largest log ratio over the brain against the measure's usual pattern (any-diagnosis
null) and against the class (same-class null), within size bins; c.rp[j][k] = [[region, p], ...] x 2 for each atlas region
whose log ratio of shares is positive and p < 0.1, against null cells of the same size bin and atlas (grey or white matter
measures pooled). Uncorrected across cells and regions; the chance rates of both (three shuffled fits per null against the
other nine) go in meta.validation.peak.
Checks that the real refit reproduces every page cell's numbers before writing.
usage: null_site.py NULL_DIR DATA_IN DATA_OUT NULL_SUMMARY PLANT_MID PLANT_MID_ALE_ROI PLANT_COMMON PLANT_COMMON_ALE_ROI [PEAK.npz]"""
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
if len(sys.argv) > 9:   # PEAK.npz from null_peak.py: the log-ratio tests (whole brain and per region)
    Z = np.load(sys.argv[9]); WMM = ["fa", "diffusivity", "white matter"]
    assert list(Z["GN"]) == D["meta"]["gm_names"] and (D["meta"].get("wm_names") is None or list(Z["WN"]) == D["meta"]["wm_names"])
    tags = sorted({k.split("/")[0] for k in Z.files if "/" in k}); fit = lambda mode: [t for t in tags if t.startswith(mode + "_")]
    real_t = fit("real")[0]; key = list(Z[f"{real_t}/key"]); n_r = Z[f"{real_t}/n"]
    RM = Z[f"{real_t}/ru"].shape[1]   # region columns are padded to the longer label list; padding is skipped
    skip = lambda wm: np.pad(np.array([x in ("Unlabeled", "White matter, no tract label") or (not wm and x == "White matter") for x in (Z["WN"] if wm else Z["GN"])]), (0, RM - len(Z["WN"] if wm else Z["GN"])), constant_values=True)
    NB = {}
    for mode, st in (("any", "u"), ("class", "c")):
        nn = np.concatenate([Z[f"{t}/n"] for t in fit(mode)]); nb = np.array([binof(int(x)) for x in nn])
        nwm = np.concatenate([np.array([k.split("|")[1] in WMM for k in Z[f"{t}/key"]]) for t in fit(mode)])
        mx = np.concatenate([Z[f"{t}/max_{st}"] for t in fit(mode)]); rr = np.concatenate([Z[f"{t}/r{st}"] for t in fit(mode)]).astype(np.float32)
        NB[st] = {"fits": len(fit(mode)), "max": [np.sort(mx[nb == i]) for i in range(len(BINS))], "reg": {(i, w): rr[(nb == i) & (nwm == w)] for i in range(len(BINS)) for w in (False, True)}}
    pv = lambda srt, v: (1 + len(srt) - np.searchsorted(srt, v, side="left")) / (1 + len(srt))
    for c in D["conditions"]: c["pk"] = [[None, None] for _ in M]; c["rp"] = [[None, None] for _ in M]
    whole = {"u": [], "c": []}; nreg = {"u": 0, "c": 0}; tested_reg = {"u": 0, "c": 0}
    for r, kk in enumerate(key):
        name, m, d = kk.split("|"); c = D["conditions"][idx[name]]; j = M.index(m); k = DIRS.index(d); n = int(n_r[r]); b = binof(n); wm = m in WMM
        assert c["cells"][j][k][0] == n, kk
        pk = []; rp = {}
        for st in ("u", "c"):
            p = float(pv(NB[st]["max"][b], Z[f"{real_t}/max_{st}"][r])); pk.append(round(p, 4)); whole[st].append(p)
            nul = NB[st]["reg"][(b, wm)]; rv = Z[f"{real_t}/r{st}"][r].astype(np.float32); ok = (rv > 0) & ~skip(wm)
            ps = (1 + (nul >= rv[None]).sum(0)) / (1 + len(nul)); tested_reg[st] += int(ok.sum())
            keep = np.where(ok & (ps < 0.1))[0]; nreg[st] += int((ok & (ps < 0.05)).sum())
            rp[st] = [[int(q), round(float(ps[q]), 4)] for q in keep]
        c["pk"][j][k] = pk; c["rp"][j][k] = [rp["u"], rp["c"]]
    cal = {}   # chance rates: three shuffled fits per null against the other nine (leave one fit out)
    for mode, st in (("any", "u"), ("class", "c")):
        F = {t: (np.array([binof(int(x)) for x in Z[f"{t}/n"]]), np.array([q.split("|")[1] in WMM for q in Z[f"{t}/key"]]), Z[f"{t}/max_{st}"], Z[f"{t}/r{st}"].astype(np.float32)) for t in fit(mode)}
        w05 = wn = pos = sig = 0
        for t in fit(mode)[:3]:
            b, w, mx, rr = F[t]; o = [F[u] for u in fit(mode) if u != t]; ob, ow, om, orr = (np.concatenate([x[i] for x in o]) for i in range(4))
            for i in range(len(b)):
                w05 += pv(np.sort(om[ob == b[i]]), mx[i]) < 0.05; wn += 1
                nul = orr[(ob == b[i]) & (ow == w[i])]; ok = (rr[i] > 0) & ~skip(w[i])
                pos += int(ok.sum()); sig += int(((1 + (nul[:, ok] >= rr[i][ok][None]).sum(0)) / (1 + len(nul)) < 0.05).sum())
        cal[st] = [round(w05 / wn, 4), round(sig / pos, 4)]
    D["meta"]["validation"]["peak"] = {"chance_whole_p05": [cal["u"][0], cal["c"][0]], "chance_region_p05": [cal["u"][1], cal["c"][1]],"fits": [NB["u"]["fits"], NB["c"]["fits"]], "cells": len(key),
        "whole_p05": [int((np.array(whole[s]) < 0.05).sum()) for s in ("u", "c")],
        "regions_tested": [tested_reg["u"], tested_reg["c"]], "regions_p05": [nreg["u"], nreg["c"]],
        "min_p_whole": round(1 / (1 + min(len(x) for x in NB["c"]["max"])), 5)}
    print("peak test: cells", len(key), "whole-brain p<0.05 against usual", D["meta"]["validation"]["peak"]["whole_p05"][0], "against class", D["meta"]["validation"]["peak"]["whole_p05"][1],
          "| regions with a positive log ratio", tested_reg, "p<0.05", nreg, "| chance (whole share, region share)", cal)
json.dump(D, open(dout, "w"), separators=(",", ":"))
ps = np.array([q for c in D["conditions"] for p in c["np"] for q in p if q is not None])
print("cells", seen, "p<0.05 against usual", int((ps[:, 0] < 0.05).sum()), "against class", int((ps[:, 1] < 0.05).sum()), "both", int(((ps < 0.05).all(1)).sum()))
