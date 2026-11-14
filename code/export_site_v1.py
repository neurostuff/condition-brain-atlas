"""Data for the condition atlas page, from one windowed fit with a flat network (pc_model.py fit --model window --flat-bg).
usage: export_site.py CORPUS FIT_PREFIX VOCABULARY.json CELL_TEST.json FUTURE_TEST.json OUT.json   (PYTHONPATH needs task_networks)

Networks: the learned ones (the flat network 0 is reported as "no specific pattern"). A network's picture in a measure is
its map there minus the measure's own map (beta[k, b] - r s[b], r = min beta / s), "less in patients" direction.
Conditions: every condition with >= MIN_C contrasts; a condition = its author plus its "less" / "more" children, loadings
from pooled counts. Its picture in a measure: its "less" loadings (pooled if it has no "less" child) over the learned
networks' pictures. Images are transparent PNGs (sagittal and axial maximum projections), base64.
No text from papers: condition names are the normalized vocabulary; counts are counts.
"""
import base64, collections, io, json, os, re, sys
import xml.etree.ElementTree as ET
import numpy as np, nibabel as nib
from PIL import Image

MIN_C, NEAR = 15, 6
NET_MEAS = ["grey matter", "cortical thickness", "fa", "reho", "glucose metabolism", "perfusion"]
CON_MEAS = ["grey matter", "cortical thickness", "fa", "reho", "glucose metabolism"]
LABEL = {"grey matter": "Grey matter volume", "white matter": "White matter volume", "cortical thickness": "Cortical thickness", "fa": "Fractional anisotropy",
         "diffusivity": "Diffusivity", "alff": "ALFF", "falff": "Fractional ALFF", "reho": "ReHo", "degree centrality": "Degree centrality",
         "glucose metabolism": "Glucose metabolism", "perfusion": "Perfusion", "molecular binding": "Tracer binding", "task activation": "Task activation", "eeg meg": "EEG / MEG sources"}
corpus, fit, vocab, cellf, futf, out = sys.argv[1:7]
H = json.load(open(corpus)); M = json.load(open(fit + ".json")); nodes, names = M["nodes"], M["blocks"]; B = len(names)
be, s, n, xyz = (np.load(f"{fit}_{x}.npy").astype(float) for x in ("beta", "s", "n", "xyz")); V = len(xyz); K = n.shape[1]; be = be.reshape(K, B, V); s = s.reshape(B, V)
ix = {x: i for i, x in enumerate(nodes)}; bix = {b: i for i, b in enumerate(names)}; per = float(n.sum()) / len(H["docs"])
ok = s > np.median(s, 1, keepdims=True)
r = np.array([[(be[k, b][ok[b]] / s[b][ok[b]]).min() for b in range(B)] for k in range(K)]); own = np.clip(be - r[:, :, None] * s[None], 0, None)
own = own / np.maximum(own.sum(2, keepdims=True), 1e-30); own[0] = 0.0          # the flat network has no picture of its own
A = os.path.expanduser("~/nilearn_data/fsl/data/atlases/")
def atlas(img, xml):
    a = nib.load(A + "HarvardOxford/" + img); nm = {int(l.get("index")) + 1: l.text for l in ET.parse(A + xml).getroot().iter("label")}
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(a.affine), xyz)).astype(int); inb = np.all((ijk >= 0) & (ijk < np.array(a.shape)), 1)
    lab = np.zeros(V, int); lab[inb] = np.asarray(a.dataobj)[tuple(ijk[inb].T)]
    return lab, nm
cl, cn = atlas("HarvardOxford-cort-maxprob-thr25-2mm.nii.gz", "HarvardOxford-Cortical.xml")
sl, sn = atlas("HarvardOxford-sub-maxprob-thr25-2mm.nii.gz", "HarvardOxford-Subcortical.xml")
skip = re.compile(r"Cerebral (White Matter|Cortex)|Ventricle", re.I)
short = lambda x: re.sub(r" \(.*\)", "", x.split(",")[0]).replace(" Cortex", "").replace(" Gyrus", "").replace("Juxtapositional Lobule", "Supplementary motor").replace("Precuneous", "Precuneus")
region = np.array([short(cn[c]) if c else (re.sub(r"^(Left|Right) ", "", sn[q]) if q and not skip.search(sn[q]) else "White matter" if q else "Cerebellum") for c, q in zip(cl, sl)])
rn, rc = np.unique(region, return_inverse=True)
def regions(v, k=3, base=None):
    """regions holding the most of a picture; with base (the measure's own map), ranked by how much more of the picture
    than of the base they hold, among regions with >= 4% of the picture"""
    w = np.bincount(rc, v, len(rn)); w = w / max(w.sum(), 1e-30)
    if base is None:
        o = [j for j in np.argsort(-w) if rn[j] != "White matter"][:k]
    else:
        b0 = np.bincount(rc, base, len(rn)); b0 = b0 / b0.sum(); lift = w / np.maximum(b0, 1e-6)
        o = [j for j in np.argsort(-lift) if w[j] >= 0.04 and rn[j] != "White matter"][:k]
    return [[rn[j], round(float(w[j]), 2)] + ([] if base is None else [round(float(lift[j]), 1)]) for j in o]
# pictures
lo = xyz.min(0); ijk = np.rint((xyz - lo) / 4).astype(int); shp = ijk.max(0) + 1; mask = np.zeros(shp, bool); mask[tuple(ijk.T)] = True
ramp = np.array([[254, 196, 79], [236, 112, 20], [204, 76, 2], [140, 45, 4], [80, 20, 10]], float)
def panel(v, Z=3):
    vol = np.zeros(shp); vol[tuple(ijk.T)] = v / max(v.max(), 1e-30)
    def view(img, m):
        img, m = img[::-1], m[::-1]                                  # up is up
        rgba = np.zeros(img.shape + (4,), np.uint8); rgba[m] = (128, 128, 128, 46)
        on = img >= 0.3; t = np.clip((img[on] - 0.3) / 0.7, 0, 1) * (len(ramp) - 1); i0 = np.minimum(t.astype(int), len(ramp) - 2); f = (t - i0)[:, None]
        rgba[on, :3] = (ramp[i0] * (1 - f) + ramp[i0 + 1] * f).astype(np.uint8); rgba[on, 3] = 255
        return np.kron(rgba, np.ones((Z, Z, 1), np.uint8))
    a, b = view(vol.max(0).T, mask.max(0).T), view(vol.max(2).T, mask.max(2).T); h = max(a.shape[0], b.shape[0])
    pad = lambda x: np.pad(x, ((h - x.shape[0], 0), (0, 2 * Z), (0, 0)))
    return np.concatenate([pad(a), pad(b)], 1)
def strip(vs):
    buf = io.BytesIO(); Image.fromarray(np.concatenate([np.pad(panel(v), ((0, 0), (0, 12), (0, 0))) for v in vs], 1), "RGBA").save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()
voc = {v["condition"]: v for v in json.load(open(vocab))}
grp = collections.defaultdict(list)
for x in nodes:
    if x.startswith("c:all:") and not x[6:].startswith(("class ", "any condition")): grp[x.split(" | ")[0][6:]].append(ix[x])
cm = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0])); papers = collections.defaultdict(set)
for d in H["docs"].values():
    for i in d["ids"]:
        if i.startswith("c:all:") and not i[6:].startswith(("class ", "any condition")):
            c = i[6:].split(" | ")[0]; cm[c][d["measure"]][d["direction"] == "increase"] += 1; papers[c].add(d["group"])
tot = {c: sum(a + b for a, b in v.values()) for c, v in cm.items()}
keep = sorted((c for c in grp if tot.get(c, 0) >= MIN_C and c != "unspecified condition"), key=lambda c: -tot[c]); cid = {c: i for i, c in enumerate(keep)}
nz = lambda v: v / max(v.sum(), 1e-12)
T = np.array([nz(n[grp[c]].sum(0)) for c in keep]); L = T[:, 1:] / np.maximum(T[:, 1:].sum(1, keepdims=True), 1e-12)      # learned networks only, for similarity
Z = (L - L.mean(1, keepdims=True)) / np.maximum(L.std(1, keepdims=True), 1e-12); R = Z @ Z.T / L.shape[1]; np.fill_diagonal(R, -9)
g = n[[i for c in keep for i in grp[c]]].sum(0); g = g / g.sum()
nets = []
for k in range(1, K):
    cs = sorted(((T[i, k] * tot[c], i) for i, c in enumerate(keep)), reverse=True)[:8]
    reg = regions(own[k, bix["grey matter | decrease"]], base=s[bix["grey matter | decrease"]])
    nets.append({"k": k, "name": " / ".join(x[0] for i_, x in enumerate(reg[:2]) if i_ == 0 or x[2] >= 2.0) or "Diffuse", "share": round(float(g[k]), 3),
                 "regions": {m: regions(own[k, bix[m + " | decrease"]], base=s[bix[m + " | decrease"]]) for m in NET_MEAS}, "conditions": [[i, round(float(T[i, k]), 2)] for _, i in cs],
                 "img": strip([own[k, bix[m + " | decrease"]] for m in NET_MEAS])})
cons = []
for i, c in enumerate(keep):
    less, more = ix.get(f"c:all:{c} | less"), ix.get(f"c:all:{c} | more")
    tl = nz(n[less]) if less is not None and n[less].sum() / per >= 3 else None; tm = nz(n[more]) if more is not None and n[more].sum() / per >= 3 else None
    t = (tl if tl is not None else T[i]).copy(); t[0] = 0; t = nz(t)
    pics = [t @ own[:, bix[m + " | decrease"]] for m in CON_MEAS]
    top = np.sort(pics[0])[::-1]; focal = float(top[: max(1, V // 20)].sum() / max(top.sum(), 1e-30))     # share of the grey matter picture in its top 5% of voxels
    info = voc.get(c) or {}
    cons.append({"name": c, "class": (info.get("class") or "other").replace("_", " "), "parent": cid.get(info.get("parent")), "contrasts": tot[c], "papers": len(papers[c]),
                 "measures": {m: v for m, v in sorted(cm[c].items(), key=lambda kv: -sum(kv[1]))}, "load": [round(float(x), 3) for x in T[i]],
                 "less": None if tl is None else [round(float(x), 3) for x in tl], "more": None if tm is None else [round(float(x), 3) for x in tm],
                 "near": [[int(j), round(float(R[i, j]), 2)] for j in np.argsort(-R[i])[:NEAR]], "focal": round(focal, 2),
                 "kids": [cid[x] for x in keep if (voc.get(x) or {}).get("parent") == c],
                 "regions": {m: regions(p, base=s[bix[m + " | decrease"]]) for m, p in zip(CON_MEAS, pics)}, "img": strip(pics)})
cells = json.load(open(cellf))["cells"]; bm = collections.defaultdict(list)
for x in cells: bm[x["measure"]].append(x)
rel = {m: {"cells": len(v), "gain": round(sum((x["model"] - x["window"]) * x["peaks"] for x in v) / sum(x["peaks"] for x in v), 3),
           "won": round(sum(x["model"] > x["window"] for x in v) / len(v), 2)} for m, v in bm.items()}
F = json.load(open(futf)); fk = collections.defaultdict(list)
for x in F["cells"]: fk[x["kind"]].append(x)
fut = {k: {"cells": len(v), "peaks": sum(x["peaks"] for x in v), "model": round(sum((x["model"] - x["window"]) * x["peaks"] for x in v) / sum(x["peaks"] for x in v), 4),
           "class": round(sum((x["class"] - x["window"]) * x["peaks"] for x in v) / sum(x["peaks"] for x in v), 4),
           "blend": round(sum((x["blend"]["0.35"] - x["window"]) * x["peaks"] for x in v) / sum(x["peaks"] for x in v), 4)} for k, v in fk.items()}
D = {"meta": {"contrasts": len(H["docs"]), "papers": len({d["group"] for d in H["docs"].values()}), "peaks": sum(len(d["foci"]) for d in H["docs"].values()),
              "conditions_all": len(grp), "conditions_shown": len(keep), "min_contrasts": MIN_C, "learned_networks": K - 1, "flat_share": round(float(g[0]), 3),
              "settings": {k: M.get(k) for k in ("seed", "restarts", "iterations", "window")}, "net_measures": NET_MEAS, "con_measures": CON_MEAS, "labels": LABEL,
              "future_cut": F["cut"], "future_train": F["train_contrasts"], "future_test": F["test_contrasts"]},
     "networks": nets, "conditions": cons, "reliability": rel, "future": fut}
json.dump(D, open(out, "w"), separators=(",", ":"))
print(len(keep), "conditions,", len(nets), "networks;", round(os.path.getsize(out) / 1e6, 2), "MB")
print([(x["k"], x["name"], x["share"]) for x in nets])
