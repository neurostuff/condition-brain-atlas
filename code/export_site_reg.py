"""Data for the condition atlas page from the additive regression (checks.py): fits it on all contrasts and writes
OUT_DIR/data.json and OUT_DIR/maps.bin.     usage: export_site_reg.py CORPUS VOCABULARY.json OUT_DIR JHU.nii.gz JHU.xml [LAM]

A cell's map is computed in the page: P(v) = softmax_v(log window[measure, v] + Phi[v] . (A[block] + sum of A over the
cell's label chain)); the chain of (condition, direction, family) is the deepest existing of "c | dir | family", "c | dir",
then "c" and all its ancestors. maps.bin: Phi as a sparse matrix (ptr uint32, idx uint16, val uint8 / 255), A for the labels
the page needs (int8 with a float32 scale per label), log windows per measure (float32), grid ijk and atlas labels (as
export_site.py, whose label code this reuses by import).
Per condition: label ids of its chain; "strength": the standard deviation over voxels of its condition-specific log-intensity
(its own labels and its parent conditions', structure / less; class and "any condition" left out); "near": conditions whose
condition-specific maps correlate most.
"""
import collections, json, os, re, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks as K, pc_model as P
from task_networks.corpus import load_corpus
from build_corpus import FAMILY

corpus, vocab, outd, jhu_img, jhu_xml = sys.argv[1:6]; lam = float(sys.argv[6]) if len(sys.argv) > 6 else 20.0
LABEL = {"grey matter": "Grey matter volume", "cortical thickness": "Cortical thickness", "white matter": "White matter volume", "fa": "Fractional anisotropy",
         "diffusivity": "Diffusivity", "alff": "ALFF", "falff": "Fractional ALFF", "reho": "ReHo", "degree centrality": "Degree centrality",
         "glucose metabolism": "Glucose metabolism", "perfusion": "Perfusion", "molecular binding": "Tracer binding", "task activation": "Task activation", "eeg meg": "EEG / MEG sources"}
MIN_C, NEAR = 15, 6
H = json.load(open(corpus)); C = load_corpus(corpus); blk, names = P.blocks_of(H, C, "measure_dir"); B = len(names); K.FIT["names"] = names
nodes = C["nodes"]; par = C["parents"]; ix = {x: i for i, x in enumerate(nodes)}; D = len(C["ids"]); xyz = C["xyz"]; V = len(xyz)
Phi = K.basis(xyz); rows, cols, vals, F = K.features(C, H, blk, B, "full")
X = torch.zeros(D, F); X[torch.tensor(rows), torch.tensor(cols)] = torch.tensor(vals, dtype=torch.float32)
K.fit_additive(C, blk, B, Phi, X, list(range(D)), lam, torch.device("cuda:0")); A = K.FIT["A"]; logw = K.FIT["logw"]
print("fitted: train ll per peak", round(K.FIT["train_ll"], 4), flush=True)
meas = sorted({b.split(" | ")[0] for b in names}, key=lambda m: list(LABEL).index(m)); bix = {b: i for i, b in enumerate(names)}
# conditions
voc = {v["condition"]: v for v in json.load(open(vocab))}; is_cls = lambda x: x[6:].startswith(("class ", "any condition"))
cm = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0])); papers = collections.defaultdict(set)
for d in H["docs"].values():
    for i in d["ids"]:
        if i.startswith("c:all:") and not is_cls(i):
            c = i[6:].split(" | ")[0]; cm[c][d["measure"]][d["direction"] == "increase"] += 1; papers[c].add(d["group"])
tot = {c: sum(a + b for a, b in v.values()) for c, v in cm.items()}
keep = sorted((c for c in tot if tot[c] >= MIN_C and c != "unspecified condition" and "c:all:" + c in ix), key=lambda c: -tot[c]); cid = {c: i for i, c in enumerate(keep)}
def ancestors(i):
    out, stack = [], list(par[i])
    while stack:
        j = stack.pop()
        if j not in out: out.append(j); stack += par[j]
    return out
need = set(); cons = []; spec = np.zeros((len(keep), Phi.shape[1]))
nc = H["n_contrasts"]
for i, c in enumerate(keep):
    b0 = ix["c:all:" + c]; anc = ancestors(b0); base = [b0] + [j for j in anc if not is_cls(nodes[j])]; cls = [j for j in anc if is_cls(nodes[j])]
    dirs = {}; fams = {}; cnt = {}
    for dn in ("less", "more"):
        j = ix.get(f"c:all:{c} | {dn}"); dirs[dn] = j; fams[dn] = {}
        for x, k in ix.items():
            if x.startswith(f"c:all:{c} | {dn} | "): fams[dn][x.rsplit(" | ", 1)[1]] = k; cnt[k] = nc.get(x, 0)
    need.update(base + cls + [j for j in dirs.values() if j is not None] + [k for f in fams.values() for k in f.values()])
    s_ids = base + [j for j in (dirs["less"], fams["less"].get("structure")) if j is not None]; spec[i] = A[s_ids].sum(0)
    info = voc.get(c) or {}
    cons.append({"name": c, "class": (info.get("class") or "other").replace("_", " "), "parent": cid.get(info.get("parent")), "contrasts": tot[c], "papers": len(papers[c]),
                 "measures": {m: v for m, v in sorted(cm[c].items(), key=lambda kv: -sum(kv[1]))}, "base": base, "cls": cls, "dir": dirs, "fam": fams,
                 "kids": [cid[x] for x in keep if (voc.get(x) or {}).get("parent") == c]})
Dm = spec @ Phi.T; st = Dm.std(1); Z = (Dm - Dm.mean(1, keepdims=True)) / np.maximum(Dm.std(1, keepdims=True), 1e-9); R = Z @ Z.T / V; np.fill_diagonal(R, -9)
for i, c in enumerate(cons): c["strength"] = round(float(st[i]), 3); c["near"] = [[int(j), round(float(R[i, j]), 2)] for j in np.argsort(-R[i])[:NEAR]]
# classes
classes = []
for x, j in ix.items():
    if x.startswith("c:all:class "):
        nm = x[len("c:all:class "):]; mem = [i for i, c in enumerate(cons) if c["class"] == nm]
        classes.append({"name": nm, "ids": [j] + [k for k in ancestors(j)], "conditions": len(mem), "contrasts": int(sum(cons[i]["contrasts"] for i in mem)), "top": sorted(mem, key=lambda i: -cons[i]["contrasts"])[:8]}); need.update(classes[-1]["ids"])
classes.sort(key=lambda z: -z["contrasts"])
blocks = {f"{m}|{dn}": len(nodes) + bix[f"{m} | {dr}"] for m in meas for dn, dr in (("less", "decrease"), ("more", "increase"))}; need.update(blocks.values())
order = sorted(need); remap = {j: k for k, j in enumerate(order)}
for c in cons:
    c["base"] = [remap[j] for j in c["base"]]; c["cls"] = [remap[j] for j in c["cls"]]; c["dir"] = {k: (None if v is None else remap[v]) for k, v in c["dir"].items()}
    c["fam"] = {k: {f: remap[j] for f, j in v.items()} for k, v in c["fam"].items()}
for z in classes: z["ids"] = [remap[j] for j in z["ids"]]
blocks = {k: remap[j] for k, j in blocks.items()}
# labels per grid point (the code of export_site.py, run as a function of xyz)
import xml.etree.ElementTree as ET, nibabel as nib
src = open(Path(__file__).resolve().parent / "export_site.py").read(); a0 = src.index('A = os.path.expanduser("~/nilearn_data'); a1 = src.index("voc = {v[\"condition\"]")
ns = {"os": os, "re": re, "np": np, "nib": nib, "ET": ET, "xyz": xyz, "V": V, "jhu": [jhu_img, jhu_xml], "collections": collections}
exec(src[a0:a1], ns); GN, glab, WN, wlab, wm_atlas = ns["GN"], ns["glab"], ns["WN"], ns["wlab"], ns["wm_atlas"]
# binary
os.makedirs(outd, exist_ok=True); ijk = np.rint((xyz - xyz.min(0)) / 4).astype(np.uint8); shape = [int(x) + 1 for x in ijk.max(0)]
nzr, nzc = np.nonzero(Phi > 1 / 255); ptr = np.r_[0, np.cumsum(np.bincount(nzr, minlength=V))].astype(np.uint32)
As = A[order]; sc = np.maximum(np.abs(As).max(1), 1e-12) / 127; Aq = np.rint(As / sc[:, None]).astype(np.int8)
mw = np.stack([logw[bix[m + " | decrease"]] for m in meas]).astype(np.float32)
parts, layout = [], {}; off = [0]
def add(name, arr):
    pad = (-off[0]) % 4
    if pad: parts.append(b"\0" * pad); off[0] += pad
    b = np.ascontiguousarray(arr).tobytes(); layout[name] = [off[0], str(arr.dtype), list(arr.shape)]; parts.append(b); off[0] += len(b)
for nm_, arr in (("phi_ptr", ptr), ("a_scale", sc.astype(np.float32)), ("logw", mw), ("phi_idx", nzc.astype(np.uint16)), ("phi_val", np.rint(Phi[nzr, nzc] * 255).astype(np.uint8)),
                 ("ijk", ijk), ("gm", glab), ("wm", wlab), ("a", Aq)): add(nm_, arr)
open(os.path.join(outd, "maps.bin"), "wb").write(b"".join(parts))
tests = json.load(open("fits/checks/full_lam20.json")) if os.path.exists("fits/checks/full_lam20.json") else {}
Dj = {"meta": {"model": "regression", "lam": lam, "contrasts": D, "papers": len({d["group"] for d in H["docs"].values()}), "peaks": int(sum(len(f) for f in C["foci"])),
               "conditions_all": len(tot), "conditions_shown": len(keep), "min_contrasts": MIN_C, "measures": meas, "labels": LABEL, "family": FAMILY,
               "wm_measures": ["fa", "diffusivity", "white matter"], "gm_names": GN, "wm_names": WN, "wm_atlas": wm_atlas, "grid": {"shape": shape, "mm": 4},
               "layout": layout, "bumps": int(Phi.shape[1]), "blocks": blocks, "tests": tests, "strength_q": [float(np.quantile(st, q)) for q in (0.33, 0.67)]},
      "conditions": cons, "classes": classes}
json.dump(Dj, open(os.path.join(outd, "data.json"), "w"), separators=(",", ":"))
print(len(keep), "conditions,", len(classes), "classes,", len(order), "labels; phi nonzeros", len(nzc), "; data.json", round(os.path.getsize(os.path.join(outd, "data.json")) / 1e6, 2), "MB; maps.bin", round(off[0] / 1e6, 2), "MB")
for nm in ("alzheimer disease", "parkinson disease", "schizophrenia", "autism spectrum disorder", "traumatic brain injury", "huntington disease", "substance use disorder"):
    c = cons[cid[nm]]; print(f"{nm}: strength {c['strength']}; near {[(cons[j]['name'], r) for j, r in c['near'][:5]]}")
print("strength quantiles", np.quantile(st, [0.1, 0.33, 0.5, 0.67, 0.9]).round(3))
