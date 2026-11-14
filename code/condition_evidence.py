"""What the corpus holds for one condition in given measures, and what a fit predicts there.
usage: condition_evidence.py CORPUS FIT_PREFIX "CONDITION" MEASURE [MEASURE ...]   (read-only catalog lookups for DOI and year)"""
import collections, json, os, re, sqlite3, sys
import xml.etree.ElementTree as ET
import numpy as np, nibabel as nib
corpus, fit, cname = sys.argv[1:4]; want = sys.argv[4:]
H = json.load(open(corpus)); M = json.load(open(fit + ".json")); nodes, names = M["nodes"], M["blocks"]; B = len(names)
be, s, n, xyz = (np.load(f"{fit}_{x}.npy").astype(float) for x in ("beta", "s", "n", "xyz")); V = len(xyz); K = n.shape[1]; be = be.reshape(K, B, V); s = s.reshape(B, V)
A = os.path.expanduser("~/nilearn_data/fsl/data/atlases/")
def load(img, xml):
    a = nib.load(A + "HarvardOxford/" + img); return a, {int(l.get("index")) + 1: l.text for l in ET.parse(A + xml).getroot().iter("label")}
ca, cn = load("HarvardOxford-cort-maxprob-thr25-2mm.nii.gz", "HarvardOxford-Cortical.xml"); sa, sn = load("HarvardOxford-sub-maxprob-thr25-2mm.nii.gz", "HarvardOxford-Subcortical.xml")
skip = re.compile(r"Cerebral (White Matter|Cortex)|Ventricle", re.I)
def lab(p):
    out = []
    for q in np.atleast_2d(p):
        i = np.rint(nib.affines.apply_affine(np.linalg.inv(ca.affine), q[:3])).astype(int)
        if np.any(i < 0) or np.any(i >= np.array(ca.shape)): out.append("outside"); continue
        c = int(np.asarray(ca.dataobj)[tuple(i)]); q2 = int(np.asarray(sa.dataobj)[tuple(i)])
        side = "L " if q[0] < -2 else "R " if q[0] > 2 else ""
        out.append(side + cn[c].split(",")[0] if c else (sn[q2] if q2 and not skip.search(sn[q2]) else "white matter" if q2 else "cerebellum / brainstem / outside atlas"))
    return out
region = np.array([x.replace("L ", "").replace("R ", "").replace("Left ", "").replace("Right ", "") for x in lab(xyz)]); rn, rc = np.unique(region, return_inverse=True)
Y = json.load(open("data_all/paper_years.json")); art = {}
for line in open("data_all/contrasts.jsonl"):
    r = json.loads(line); art[r["paper"]] = r["article"]
cat = sqlite3.connect("file:/data/alejandro/projects/ns-pond/catalog/catalog.sqlite?mode=ro", uri=True)
def doi(p):
    b = cat.execute("select value from aliases where article_id=? and kind='doi' limit 1", (art.get(p),)).fetchone(); return b[0] if b else None
docs = [(k, d) for k, d in H["docs"].items() if any(i.split(" | ")[0] == "c:all:" + cname for i in d["ids"])]
print(cname, ":", len(docs), "contrasts;", dict(collections.Counter(d["measure"] for _, d in docs).most_common()))
ix = {x: i for i, x in enumerate(nodes)}; bix = {b: i for i, b in enumerate(names)}; per = n.sum() / len(H["docs"])
ok = s > np.median(s, 1, keepdims=True)
for m in want:
    print(f"\n=== {m}")
    for k, d in docs:
        if d["measure"] != m: continue
        others = sorted({i.split(" | ")[0][6:] for i in d["ids"] if i.startswith("c:all:")} - {cname})
        print(f"  {d['direction']:8s} {len(d['foci'])} peaks, paper {d['group']} ({Y.get(d['group'])}, doi {doi(d['group'])}), scope {d.get('scope')}" + (f", also {others}" if others else ""))
        print("      " + "; ".join(sorted(set(lab(np.array([f[:3] for f in d["foci"]], float))))) if d["foci"] else "      no peaks")
    for dr, child in (("decrease", " | less"), ("increase", " | more")):
        i = ix.get(f"c:all:{cname}{child}"); src = "direction-specific loadings"
        if i is None or n[i].sum() / per < 3: i = ix[f"c:all:{cname}"]; src = "pooled loadings (too few contrasts in this direction)"
        ids = [j for x, j in ix.items() if x.split(" | ")[0] == "c:all:" + cname] if "pooled" in src else [i]
        t = n[ids].sum(0).copy(); flat = t[0] / t.sum(); t[0] = 0; t /= t.sum(); b = bix[f"{m} | {dr}"]
        r = np.array([(be[k, b][ok[b]] / s[b][ok[b]]).min() for k in range(K)]); own = np.clip(be[:, b] - r[:, None] * s[b][None], 0, None); own[0] = 0; own /= np.maximum(own.sum(1, keepdims=True), 1e-30)
        pred = t @ own; pr = np.bincount(rc, pred, len(rn)); pr /= pr.sum(); br = np.bincount(rc, s[b], len(rn)); br /= br.sum(); lift = pr / np.maximum(br, 1e-6)
        o = [j for j in np.argsort(-lift) if pr[j] >= 0.03 and rn[j] != "white matter"][:6]
        print(f"  model, {dr} ({src}; flat share {flat:.2f}; top networks {[(int(k), round(float(t[k]), 2)) for k in np.argsort(-t)[:3]]}):")
        print("      " + "; ".join(f"{rn[j]} {pr[j]:.0%} ({lift[j]:.1f}x)" for j in o))
