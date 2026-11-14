"""Where the model predicts an effect for a condition in a measure that condition has little or no data for.
usage: novel_predictions.py CORPUS FIT_PREFIX OUT.json      (windowed fit with direction children)

For every condition with >= MIN_TOTAL contrasts and every measure where it has <= MAX_CELL contrasts (direction
"decrease"): predicted map = the condition's "less" mixture (its pooled mixture if it has no "less" child) times the
networks' maps in that block, flat network left out and renormalized (what is specific to the condition). Per
Harvard-Oxford region: its share of the prediction and its lift over the measure's own map. A cell's headline is the
region with the largest lift among regions holding >= MIN_SHARE of the prediction. Also listed: which measures of the
condition carry the networks behind the prediction (where the evidence comes from).
"""
import collections, json, os, re, sys
import xml.etree.ElementTree as ET
import numpy as np, nibabel as nib

MIN_TOTAL, MAX_CELL, MIN_SHARE = 40, 2, 0.06
MEAS = ["grey matter", "cortical thickness", "fa", "reho", "alff", "glucose metabolism", "perfusion"]
corpus, fit, out = sys.argv[1:4]
H = json.load(open(corpus)); M = json.load(open(fit + ".json")); nodes, names = M["nodes"], M["blocks"]; B = len(names)
be, s, n, xyz = (np.load(f"{fit}_{x}.npy").astype(float) for x in ("beta", "s", "n", "xyz")); V = len(xyz); K = n.shape[1]; be = be.reshape(K, B, V); s = s.reshape(B, V)
ix = {x: i for i, x in enumerate(nodes)}; bix = {b: i for i, b in enumerate(names)}
A = os.path.expanduser("~/nilearn_data/fsl/data/atlases/")
def atlas(img, xml):
    a = nib.load(A + "HarvardOxford/" + img); nm = {int(l.get("index")) + 1: l.text for l in ET.parse(A + xml).getroot().iter("label")}
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(a.affine), xyz)).astype(int); inb = np.all((ijk >= 0) & (ijk < np.array(a.shape)), 1)
    lab = np.zeros(V, int); lab[inb] = np.asarray(a.dataobj)[tuple(ijk[inb].T)]
    return lab, nm
cl, cn = atlas("HarvardOxford-cort-maxprob-thr25-2mm.nii.gz", "HarvardOxford-Cortical.xml")
sl, sn = atlas("HarvardOxford-sub-maxprob-thr25-2mm.nii.gz", "HarvardOxford-Subcortical.xml")
skip = re.compile(r"Cerebral (White Matter|Cortex)|Ventricle", re.I)
region = np.array([cn[c] if c else (re.sub(r"^(Left|Right) ", "", sn[q]) if q and not skip.search(sn[q]) else "white matter" if q else "cerebellum / outside atlas") for c, q in zip(cl, sl)])
rn, rc = np.unique(region, return_inverse=True); rsum = lambda v: np.bincount(rc, v, len(rn))
cm = collections.defaultdict(collections.Counter)
for d in H["docs"].values():
    for i in d["ids"]:
        if i.startswith("c:all:"): cm[i[6:].split(" | ")[0]][d["measure"]] += 1
flat = 0 if M.get("flat_background") else None; res = []
for c, v in cm.items():
    if sum(v.values()) < MIN_TOTAL or c.startswith(("class ", "any condition")) or c == "unspecified condition": continue
    i = ix.get(f"c:all:{c} | less", ix.get(f"c:all:{c}")); t = n[i].copy()
    if flat is not None: t[flat] = 0
    if t.sum() == 0: continue
    t /= t.sum()
    for m in MEAS:
        if v[m] > MAX_CELL: continue
        b = bix[f"{m} | decrease"]; pred = t @ be[:, b]; pr = rsum(pred) / pred.sum(); br = rsum(s[b]) / s[b].sum(); lift = pr / np.maximum(br, 1e-6)
        okr = [j for j in np.argsort(-lift) if pr[j] >= MIN_SHARE and rn[j] not in ("white matter", "cerebellum / outside atlas")][:3]
        if not okr: continue
        top = np.argsort(-t)[:2]
        res.append({"condition": c, "measure": m, "contrasts_in_measure": int(v[m]), "contrasts_total": int(sum(v.values())),
                    "regions": [[rn[j], round(float(pr[j]), 3), round(float(lift[j]), 1)] for j in okr], "headline_lift": float(lift[okr[0]]),
                    "networks": [[int(k), round(float(t[k]), 2)] for k in top], "evidence_measures": [[mm, int(k)] for mm, k in v.most_common(4)]})
res.sort(key=lambda r: -r["headline_lift"])
json.dump(res, open(out, "w"), indent=1)
print(len(res), "cells with a prediction;", len({r["condition"] for r in res}), "conditions")
for r in res[:60]:
    print(f"{r['headline_lift']:5.1f}x  {r['condition']} | {r['measure']} ({r['contrasts_in_measure']} of {r['contrasts_total']}): " +
          "; ".join(f"{a} {p:.0%} ({l}x)" for a, p, l in r["regions"]) + "  <- " + ", ".join(f"{m} {k}" for m, k in r["evidence_measures"]))
