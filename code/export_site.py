"""Data for the condition atlas page, from one windowed fit with a flat network (pc_model.py fit --model window --flat-bg).
usage: export_site.py CORPUS FIT_PREFIX VOCABULARY.json CELL_TEST.json FUTURE_TEST.json OUT_DIR [JHU_LABELS.nii.gz JHU.xml]
writes OUT_DIR/data.json and OUT_DIR/maps.bin                         (PYTHONPATH needs task_networks)

maps.bin holds, on the fit's 4 mm grid, everything the page needs to draw any condition or network in any measure:
ijk (uint8, V x 3), grey matter labels (uint8, V), white matter labels (uint8, V), each measure's own map (the window;
uint8, M x V, float32 scales) and each learned network's picture in each measure (uint8, (K-1) x M x V, float32 scales).
A network's picture in a measure is its map there minus the measure's own map (beta - r s, r = min beta / s over voxels
where s is above its median), normalized to sum 1; the flat network has none. A condition's picture is its loadings over
the learned networks (flat left out, renormalized) times those pictures, computed in the page. meta.layout gives offsets.
Labels: Harvard-Oxford (cortical and subcortical maxprob thr 25, 2 mm): the most common grey matter label in the 3 x 3 x 3
neighbourhood (2 mm voxels) of each grid point, sided by x. Points with none: cerebral white matter if the subcortical atlas
says so, else cerebellum below and behind the tentorium (y < -30, z < -8; Harvard-Oxford has no cerebellum), else
unlabeled. White matter measures (FA, diffusivity, white matter volume) use JHU ICBM-DTI-81 labels the same way when
given, else the grey matter labels. No text from papers: condition names are the normalized vocabulary.
"""
import collections, json, os, re, sys
import xml.etree.ElementTree as ET
import numpy as np, nibabel as nib

MIN_C, NEAR = 15, 6
WM_MEAS = ("fa", "diffusivity", "white matter")
LABEL = {"grey matter": "Grey matter volume", "cortical thickness": "Cortical thickness", "white matter": "White matter volume", "fa": "Fractional anisotropy",
         "diffusivity": "Diffusivity", "alff": "ALFF", "falff": "Fractional ALFF", "reho": "ReHo", "degree centrality": "Degree centrality",
         "glucose metabolism": "Glucose metabolism", "perfusion": "Perfusion", "molecular binding": "Tracer binding", "task activation": "Task activation", "eeg meg": "EEG / MEG sources"}
FAMILY = {"grey matter": "structure", "cortical thickness": "structure", "white matter": "structure", "fa": "diffusion", "diffusivity": "diffusion",
          "alff": "resting function", "falff": "resting function", "reho": "resting function", "degree centrality": "resting function",
          "glucose metabolism": "metabolism and perfusion", "perfusion": "metabolism and perfusion", "molecular binding": "molecular",
          "task activation": "task", "eeg meg": "electrophysiology"}     # as build_corpus.py
corpus, fit, vocab, cellf, futf, outd = sys.argv[1:7]; jhu = sys.argv[7:9] if len(sys.argv) > 8 else None
H = json.load(open(corpus)); M = json.load(open(fit + ".json")); nodes, names = M["nodes"], M["blocks"]; B = len(names)
be, s, n, xyz = (np.load(f"{fit}_{x}.npy").astype(float) for x in ("beta", "s", "n", "xyz")); V = len(xyz); K = n.shape[1]; be = be.reshape(K, B, V); s = s.reshape(B, V)
ix = {x: i for i, x in enumerate(nodes)}; bix = {b: i for i, b in enumerate(names)}; per = float(n.sum()) / len(H["docs"])
meas = sorted({b.split(" | ")[0] for b in names}, key=lambda m: list(LABEL).index(m)); Mn = len(meas)
ok = s > np.median(s, 1, keepdims=True); own = np.zeros((K, Mn, V)); win = np.zeros((Mn, V))
for j, m in enumerate(meas):
    b = bix[m + " | decrease"]; win[j] = s[b] / s[b].sum()      # the window is pooled over a measure's directions: both blocks agree
    for k in range(1, K):
        r = (be[k, b][ok[b]] / s[b][ok[b]]).min(); o = np.clip(be[k, b] - r * s[b], 0, None); own[k, j] = o / max(o.sum(), 1e-30)
A = os.path.expanduser("~/nilearn_data/fsl/data/atlases/")
def nice(x):
    x = re.sub(r"^(Left|Right) ", "", x).replace("Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)", "Supplementary motor cortex")
    x = x.replace("Heschl's Gyrus (includes H1 and H2)", "Heschl's gyrus").replace("Precuneous Cortex", "Precuneus").replace("Brain-Stem", "Brainstem")
    x = re.sub(r", (superior|inferior|anterior|posterior) division", r" (\1)", x).replace(", temporooccipital part", " (temporo-occipital)")
    x = x.replace(", pars triangularis", " (triangularis)").replace(", pars opercularis", " (opercularis)")
    return (x[0] + x[1:].lower()).replace("heschl", "Heschl")
def neighbourhood_mode(img, keep_labels, pts, r=1):
    dat = np.asarray(img.dataobj).astype(int); c = np.rint(nib.affines.apply_affine(np.linalg.inv(img.affine), pts)).astype(int); labs = []
    for d in np.stack(np.meshgrid(*[np.arange(-r, r + 1)] * 3, indexing="ij"), -1).reshape(-1, 3):
        q = c + d; inb = np.all((q >= 0) & (q < np.array(dat.shape)), 1); l = np.zeros(len(pts), int); l[inb] = dat[tuple(q[inb].T)]
        l[~np.isin(l, keep_labels)] = 0; labs.append(l)
    labs = np.stack(labs, 1); out = np.zeros(len(pts), int)
    for i in range(len(pts)):
        nzl = labs[i][labs[i] > 0]
        if len(nzl): out[i] = np.bincount(nzl).argmax()
    return out
ci = nib.load(A + "HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"); cn = {int(l.get("index")) + 1: l.text for l in ET.parse(A + "HarvardOxford-Cortical.xml").getroot().iter("label")}
si = nib.load(A + "HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz"); sn = {int(l.get("index")) + 1: l.text for l in ET.parse(A + "HarvardOxford-Subcortical.xml").getroot().iter("label")}
skip = re.compile(r"Cerebral (White Matter|Cortex)|Ventricle", re.I)
cort = neighbourhood_mode(ci, list(cn), xyz); sub = neighbourhood_mode(si, [k for k, v in sn.items() if not skip.search(v)], xyz)
wmv = neighbourhood_mode(si, [k for k, v in sn.items() if "White Matter" in v], xyz); side = np.where(xyz[:, 0] < 0, "Left ", "Right ")
gname = []
for i in range(V):
    if sub[i] and (not cort[i] or sn[sub[i]].split()[-1] in ("Hippocampus", "Amygdala", "Thalamus", "Caudate", "Putamen", "Pallidum", "Accumbens", "Brain-Stem")):
        nm = nice(sn[sub[i]]); gname.append(nm if nm == "Brainstem" else side[i] + nm.lower())
    elif cort[i]: gname.append(side[i] + nice(cn[cort[i]]).lower().replace("heschl", "Heschl"))
    elif wmv[i]: gname.append("White matter")
    elif xyz[i, 1] < -30 and xyz[i, 2] < -8: gname.append(side[i] + "cerebellum")
    else: gname.append("Unlabeled")
GN = ["Unlabeled"] + sorted(set(gname) - {"Unlabeled"}); glab = np.array([GN.index(x) for x in gname], np.uint8)
if jhu:
    # JHU's XML index is the image value (index 0 = Unclassified), unlike Harvard-Oxford's (value = index + 1)
    raw = {int(l.get("index")): l.text for l in ET.parse(jhu[1]).getroot().iter("label")}
    def jname(t):
        t = t.strip(); m_ = re.match(r"(.*) (L|R)$", t); sd = {"L": "Left ", "R": "Right "}[m_.group(2)] if m_ else ""; t = m_.group(1) if m_ else t
        t = re.sub(r" \(.*?\)", "", t); return sd + (t[0].lower() + t[1:] if sd else t)
    jn = {k: jname(v) for k, v in raw.items() if k > 0 and v and "Unclassified" not in v}
    jl = neighbourhood_mode(nib.load(jhu[0]), list(jn), xyz, r=2)   # 5 x 5 x 5 at 2 mm: JHU labels only the deep white matter, so look one grid step around
    # points with no tract label keep their grey matter label, so a white matter measure's pattern in grey matter is named too
    wname = [jn[x] if x else (gname[i] if gname[i] != "White matter" else "White matter, no tract label") for i, x in enumerate(jl)]
    WN = ["Unlabeled"] + sorted(set(wname) - {"Unlabeled"}); wlab = np.array([WN.index(x) for x in wname], np.uint8)
    wm_atlas = "JHU ICBM-DTI-81 white matter atlas (tracts), with Harvard-Oxford labels where no tract is labelled"
    print("tract-labelled grid points:", int((jl > 0).sum()), "of", V)
else:
    WN, wlab, wm_atlas = None, glab, None
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
T = np.array([nz(n[grp[c]].sum(0)) for c in keep]); L = T[:, 1:] / np.maximum(T[:, 1:].sum(1, keepdims=True), 1e-12)
Z = (L - L.mean(1, keepdims=True)) / np.maximum(L.std(1, keepdims=True), 1e-12); R = Z @ Z.T / L.shape[1]; np.fill_diagonal(R, -9)
g = n[[i for c in keep for i in grp[c]]].sum(0); g = g / g.sum()
gm = meas.index("grey matter"); bil = np.array([re.sub(r"^(Left|Right) ", "", x) for x in GN]); BN, bcode = np.unique(bil, return_inverse=True)
# hand-made names (from each network's top regions, checked by eye) win over the automatic ones when OUT_DIR/network_names.json exists
NAMES = json.load(open(os.path.join(outd, "network_names.json"))) if os.path.exists(os.path.join(outd, "network_names.json")) else {}
nets = []
for k in range(1, K):
    w = np.bincount(bcode[glab], own[k, gm], len(BN)); w /= w.sum(); b0 = np.bincount(bcode[glab], win[gm], len(BN)); b0 /= b0.sum(); lift = w / np.maximum(b0, 1e-6)
    cand = [j for j in np.argsort(-lift) if w[j] >= 0.05 and BN[j] not in ("Unlabeled", "White matter")]
    nm = BN[cand[0]] if cand else "Diffuse"
    if len(cand) > 1 and lift[cand[1]] >= 2.0: nm += " and " + BN[cand[1]].lower()
    cs = sorted(((T[i, k] * tot[c], i) for i, c in enumerate(keep)), reverse=True)[:8]
    top = [(BN[j], round(float(w[j]), 3), round(float(lift[j]), 1)) for j in np.argsort(-w) if BN[j] not in ("Unlabeled", "White matter")][:5]
    if lift[cand[0]] < 2.0 if cand else True: nm = "diffuse, no focal pattern"
    nets.append({"k": k, "name": NAMES.get(str(k), nm[0].upper() + nm[1:]), "auto_name": nm, "share": round(float(g[k]), 3), "conditions": [[i, round(float(T[i, k]), 2)] for _, i in cs], "_top": top})
cons = []
for i, c in enumerate(keep):
    # loadings per direction, and per direction and measure family where the corpus has such authors (multimodal_fam): "_" pools
    # the direction's authors; a family entry exists when that author has >= 3 contrasts' worth of counts
    dirs = {}
    for dname in ("less", "more"):
        ids = [j for x, j in ix.items() if x == f"c:all:{c} | {dname}" or x.startswith(f"c:all:{c} | {dname} | ")]
        tot_d = n[ids].sum(0) if ids else np.zeros(K); e = {"_": [round(float(x), 4) for x in nz(tot_d)] if tot_d.sum() / per >= 3 else None}
        for x, j in ix.items():
            if x.startswith(f"c:all:{c} | {dname} | ") and n[j].sum() / per >= 3: e[x.rsplit(" | ", 1)[1]] = [round(float(v), 4) for v in nz(n[j])]
        dirs[dname] = e
    tl = dirs["less"].get("structure") or dirs["less"]["_"]; tm = dirs["more"]["_"]
    tl = None if tl is None else np.array(tl); tm = None if tm is None else np.array(tm)
    t = (tl if tl is not None else T[i]).copy(); t[0] = 0; t = nz(t); pic = t @ own[:, gm]
    top = np.sort(pic)[::-1]; focal = float(top[: max(1, V // 20)].sum() / max(top.sum(), 1e-30)); info = voc.get(c) or {}
    cons.append({"name": c, "class": (info.get("class") or "other").replace("_", " "), "parent": cid.get(info.get("parent")), "contrasts": tot[c], "papers": len(papers[c]),
                 "measures": {m: v for m, v in sorted(cm[c].items(), key=lambda kv: -sum(kv[1]))}, "load": [round(float(x), 4) for x in T[i]],
                 "less": dirs["less"]["_"], "more": dirs["more"]["_"], "dir": dirs,
                 "near": [[int(j), round(float(R[i, j]), 2)] for j in np.argsort(-R[i])[:NEAR]], "focal": round(focal, 2),
                 "kids": [cid[x] for x in keep if (voc.get(x) or {}).get("parent") == c]})
cells = json.load(open(cellf))["cells"]; bm = collections.defaultdict(list)
for x in cells: bm[x["measure"]].append(x)
rel = {m: {"cells": len(v), "gain": round(sum((x["model"] - x["window"]) * x["peaks"] for x in v) / sum(x["peaks"] for x in v), 3),
           "won": round(sum(x["model"] > x["window"] for x in v) / len(v), 2)} for m, v in bm.items()}
F = json.load(open(futf)); fk = collections.defaultdict(list)
for x in F["cells"]: fk[x["kind"]].append(x)
gain = lambda v, f: round(sum((f(x) - x["window"]) * x["peaks"] for x in v) / sum(x["peaks"] for x in v), 4)
fut = {k: {"cells": len(v), "peaks": sum(x["peaks"] for x in v), "model": gain(v, lambda x: x["model"]), "class": gain(v, lambda x: x["class"]),
           "blend": gain(v, lambda x: x["blend"]["0.35"])} for k, v in fk.items()}
os.makedirs(outd, exist_ok=True); ijk = np.rint((xyz - xyz.min(0)) / 4).astype(np.uint8); shape = [int(x) + 1 for x in ijk.max(0)]
parts, layout = [], {}; off = [0]
def add(name, arr):
    pad = (-off[0]) % 4
    if pad: parts.append(b"\0" * pad); off[0] += pad
    b = np.ascontiguousarray(arr).tobytes(); layout[name] = [off[0], str(arr.dtype), list(arr.shape)]; parts.append(b); off[0] += len(b)
def quant(X):
    mx = X.max(-1, keepdims=True); return np.rint(255 * X / np.maximum(mx, 1e-30)).astype(np.uint8), (mx[..., 0] / 255).astype(np.float32)
qw, sw = quant(win); qo, so = quant(own[1:])
for nm_, arr in (("win_scale", sw), ("own_scale", so), ("ijk", ijk), ("gm", glab), ("wm", wlab), ("win", qw), ("own", qo)): add(nm_, arr)
open(os.path.join(outd, "maps.bin"), "wb").write(b"".join(parts))
D = {"meta": {"contrasts": len(H["docs"]), "papers": len({d["group"] for d in H["docs"].values()}), "peaks": sum(len(d["foci"]) for d in H["docs"].values()),
              "conditions_all": len(grp), "conditions_shown": len(keep), "min_contrasts": MIN_C, "learned_networks": K - 1, "flat_share": round(float(g[0]), 3),
              "settings": {k: M.get(k) for k in ("seed", "restarts", "iterations", "window")}, "measures": meas, "labels": LABEL, "wm_measures": list(WM_MEAS), "family": FAMILY,
              "gm_names": GN, "wm_names": WN, "wm_atlas": wm_atlas, "grid": {"shape": shape, "mm": 4, "origin": [float(x) for x in xyz.min(0)]}, "layout": layout,
              "future_cut": F["cut"], "future_train": F["train_contrasts"], "future_test": F["test_contrasts"]},
     "networks": [{k: v for k, v in x.items() if k != "_top"} for x in nets], "conditions": cons, "reliability": rel, "future": fut}
json.dump(D, open(os.path.join(outd, "data.json"), "w"), separators=(",", ":"))
print(len(keep), "conditions,", len(nets), "networks; data.json", round(os.path.getsize(os.path.join(outd, "data.json")) / 1e6, 2), "MB; maps.bin", round(off[0] / 1e6, 2), "MB")
print("label counts:", collections.Counter(re.sub(r"^(Left|Right) ", "", x) for x in gname).most_common(10))
for x in nets: print(x["k"], x["name"], x["share"], "|", "; ".join(f"{a} {p:.0%} ({l}x)" for a, p, l in x["_top"]), "|", [(keep[i], l) for i, l in x["conditions"][:3]])
