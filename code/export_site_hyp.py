"""Data for the condition atlas page (hypothesis-finding version) from the additive regression kept after MODEL_PLAN.md:
labels split by direction, direction by meaning, family-level penalty 5, window from peaks counted once.
usage: export_site_hyp.py CORPUS VOCABULARY.json OUT_DIR JHU.nii.gz JHU.xml TESTS.json CELL_SCORES.npy [BOOT_coef.npz RHO.json COVERAGE.json]

Fits checks.py's regression on all contrasts and writes OUT_DIR/data.json and OUT_DIR/maps.bin. A cell's map is computed in
the page: P(v) = softmax_v(log window[measure, v] + Phi[v] . (A[block] + sum of A over the cell's labels)); the labels of
(condition, measure, direction) are, with dn the direction by meaning (flipped for the measures in checks.FLIP):
"c | dn | family" if it exists, "x | dn" for the condition and each ancestor x (parent conditions, class, any condition).
maps.bin as export_site_reg.py: Phi sparse, A int8 with a scale per label, log windows per measure, grid ijk, atlas labels.
Per condition and cell (measure x reported direction): contrasts; share of predicted peaks that move relative to the
measure's usual pattern and relative to the class's (total variation distance); the three regions gaining most.
"transfer": for cells held out in the held-out-measure test, gain per peak when the measure was predicted from the rest.
"pairs": conditions of different classes whose condition-specific maps (structure, less) correlate most.
Intervals (with BOOT_coef.npz from uncertainty.py fit --test all --save-coef, RHO.json from uncertainty.py score on that fit,
COVERAGE.json from unc_report.py): for each cell and up to 16 regions (the 8 gaining most against the measure's usual pattern
and the 8 gaining most against the class), from the fits on resampled papers: the 10th and 90th percentile of the region's
share of predicted peaks, of that share relative to the measure's usual share and relative to the class's (each as a
percentage of the estimate), and the 80% predictive interval for how many of one new paper's 10 peaks fall in the region
(beta-binomial with the within-paper correlation of RHO.json, mixed over the fits). maps.bin: "iv_idx" (cells x 16, region
index, 255 = none; cell = (condition * measures + measure) * 2 + direction), "iv" (cells x 16 x 7: share low, high; relative
to measure low, high; relative to class low, high; predictive low * 16 + high).
"""
import collections, json, os, re, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks as K, pc_model as P
from task_networks.corpus import load_corpus
from build_corpus import FAMILY

corpus, vocab, outd, jhu_img, jhu_xml, tests_json, cell_npy = sys.argv[1:8]; lam = 20.0; VARIANT = "full+dirsplit+meaning"; PEN = {"fam": 5.0}
LABEL = {"grey matter": "Grey matter volume", "cortical thickness": "Cortical thickness", "white matter": "White matter volume", "fa": "Fractional anisotropy",
         "diffusivity": "Diffusivity", "alff": "ALFF", "falff": "Fractional ALFF", "reho": "ReHo", "degree centrality": "Degree centrality",
         "glucose metabolism": "Glucose metabolism", "perfusion": "Perfusion", "molecular binding": "Tracer binding", "task activation": "Task activation", "eeg meg": "EEG / MEG sources"}
MIN_C, NEAR, WM = 15, 6, ["fa", "diffusivity", "white matter"]
H = json.load(open(corpus)); C = load_corpus(corpus); blk, names = P.blocks_of(H, C, "measure_dir"); B = len(names)
K.FIT.update(names=names, wfrom="foci", opt="lbfgs", outer=15, pen=PEN)
nodes = C["nodes"]; par = C["parents"]; nix = {x: i for i, x in enumerate(nodes)}; D = len(C["ids"]); xyz = C["xyz"]; V = len(xyz)
Phi = K.basis(xyz); rows, cols, vals, F = K.features(C, H, blk, B, VARIANT, names); ix = {k: i for i, k in enumerate(K.FIT["keys"])}
X = torch.zeros(D, F); X[torch.tensor(rows), torch.tensor(cols)] = torch.tensor(vals, dtype=torch.float32)
rk = collections.defaultdict(list)
for r_, c_, v_ in zip(rows, cols, vals): rk[r_].append((c_, round(v_, 6)))
gk = {}; K.FIT["gid"] = np.array([gk.setdefault(tuple(sorted(rk[d])), len(gk)) for d in range(D)])
K.fit_additive(C, blk, B, Phi, X, list(range(D)), lam, torch.device("cuda:0")); A = K.FIT["A"]; logw = K.FIT["logw"]
print("fitted: train ll per peak", round(K.FIT["train_ll"], 4), flush=True)
meas = sorted({b.split(" | ")[0] for b in names}, key=lambda m: list(LABEL).index(m)); bix = {b: i for i, b in enumerate(names)}
mdir = lambda m, dn: ({"less": "more", "more": "less"}[dn] if m in K.FLIP else dn)      # direction by meaning
# conditions
voc = {v["condition"]: v for v in json.load(open(vocab))}; is_cls = lambda x: x[6:].startswith(("class ", "any condition"))
cm = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0])); papers = collections.defaultdict(set)
for d in H["docs"].values():
    for i in d["ids"]:
        if i.startswith("c:all:") and not is_cls(i):
            c = i[6:].split(" | ")[0]; cm[c][d["measure"]][d["direction"] == "increase"] += 1; papers[c].add(d["group"])
tot = {c: sum(a + b for a, b in v.values()) for c, v in cm.items()}
keep = sorted((c for c in tot if tot[c] >= MIN_C and c != "unspecified condition" and "c:all:" + c in nix), key=lambda c: -tot[c]); cid = {c: i for i, c in enumerate(keep)}
def ancestors(i):
    out, stack = [], list(par[i])
    while stack:
        j = stack.pop()
        if j not in out: out.append(j); stack += par[j]
    return out
def split(js, dn): return [ix[nodes[j] + " | " + dn] for j in js if nodes[j] + " | " + dn in ix]
need = set(); cons = []; spec = np.zeros((len(keep), Phi.shape[1]))
for i, c in enumerate(keep):
    b0 = nix["c:all:" + c]; anc = ancestors(b0); bn = [b0] + [j for j in anc if not is_cls(nodes[j])]; cn = [j for j in anc if is_cls(nodes[j])]
    base = {dn: split(bn, dn) for dn in ("less", "more")}; cls = {dn: split(cn, dn) for dn in ("less", "more")}
    fams = {dn: {x.rsplit(" | ", 1)[1]: k for x, k in ix.items() if x.startswith(f"c:all:{c} | {dn} | ")} for dn in ("less", "more")}
    own = {dn: f"c:all:{c} | {dn}" in ix for dn in ("less", "more")}
    for dn in ("less", "more"): need.update(base[dn] + cls[dn] + list(fams[dn].values()))
    spec[i] = A[base["less"] + ([fams["less"]["structure"]] if "structure" in fams["less"] else [])].sum(0)
    info = voc.get(c) or {}
    cons.append({"name": c, "class": (info.get("class") or "other").replace("_", " "), "parent": cid.get(info.get("parent")), "contrasts": tot[c], "papers": len(papers[c]),
                 "measures": {m: v for m, v in sorted(cm[c].items(), key=lambda kv: -sum(kv[1]))}, "base": base, "cls": cls, "fam": fams, "own": own,
                 "kids": [cid[x] for x in keep if (voc.get(x) or {}).get("parent") == c]})
Dm = spec @ Phi.T; st = Dm.std(1); Z = (Dm - Dm.mean(1, keepdims=True)) / np.maximum(Dm.std(1, keepdims=True), 1e-9); R = Z @ Z.T / V; np.fill_diagonal(R, -9)
for i, c in enumerate(cons): c["strength"] = round(float(st[i]), 3); c["near"] = [[int(j), round(float(R[i, j]), 2)] for j in np.argsort(-R[i])[:NEAR]]
# pairs of conditions from different classes with similar condition-specific maps (neither weak, neither the other's relative)
def related(i, j):
    a, b = cons[i], cons[j]; pa, pb = {a["parent"], i}, {b["parent"], j}
    return bool((pa & pb) - {None})
q33 = float(np.quantile(st, 0.33)); pairs = sorted(((float(R[i, j]), i, j) for i in range(len(cons)) for j in range(i + 1, len(cons))
               if cons[i]["class"] != cons[j]["class"] and min(st[i], st[j]) >= q33 and not related(i, j)), reverse=True)[:80]
pairs = [[i, j, round(r, 2)] for r, i, j in pairs]
# classes
classes = []
for x, j in nix.items():
    if x.startswith("c:all:class "):
        nm = x[len("c:all:class "):]; mem = [i for i, c in enumerate(cons) if c["class"] == nm]
        ids = {dn: split([j] + ancestors(j), dn) for dn in ("less", "more")}; need.update(ids["less"] + ids["more"])
        classes.append({"name": nm, "ids": ids, "conditions": len(mem), "contrasts": int(sum(cons[i]["contrasts"] for i in mem)), "top": sorted(mem, key=lambda i: -cons[i]["contrasts"])[:8]})
classes.sort(key=lambda z: -z["contrasts"])
blocks = {f"{m}|{dn}": len(nodes) + bix[f"{m} | {dr}"] for m in meas for dn, dr in (("less", "decrease"), ("more", "increase"))}; need.update(blocks.values())
# labels per grid point (the code of export_site.py, run as a function of xyz)
import xml.etree.ElementTree as ET, nibabel as nib
src = open(Path(__file__).resolve().parent / "export_site.py").read(); a0 = src.index('A = os.path.expanduser("~/nilearn_data'); a1 = src.index("voc = {v[\"condition\"]")
ns = {"os": os, "re": re, "np": np, "nib": nib, "ET": ET, "xyz": xyz, "V": V, "jhu": [jhu_img, jhu_xml], "collections": collections}
exec(src[a0:a1], ns); GN, glab, WN, wlab, wm_atlas = ns["GN"], ns["glab"], ns["WN"], ns["wlab"], ns["wm_atlas"]
# per cell: how far the prediction moves from the measure's usual pattern and from the class's; top regions
def soft(lg): lg = lg - lg.max(-1, keepdims=True); p = np.exp(lg); return p / p.sum(-1, keepdims=True)
SKIP = {"Unlabeled", "White matter, no tract label"}
def regions(p, pb, m):
    wm = m in WM and WN is not None; lb, nm = (wlab, WN) if wm else (glab, GN); ex = np.bincount(lb, p, len(nm)) - np.bincount(lb, pb, len(nm))
    for k, x in enumerate(nm):
        if x in SKIP or (not wm and x == "White matter"): ex[k] = -1
    return [int(k) for k in np.argsort(-ex)[:3] if ex[k] > 0]
PhiT = Phi.T.astype(np.float64); pbase = {}
for m in meas:
    for dn in ("less", "more"): b = bix[f"{m} | {'decrease' if dn == 'less' else 'increase'}"]; pbase[m, dn] = soft(logw[b] + A[len(nodes) + b] @ PhiT)
for c in cons:
    c["cells"] = []
    for m in meas:
        pair = []
        for k, dn in enumerate(("less", "more")):
            b = bix[f"{m} | {'decrease' if dn == 'less' else 'increase'}"]; md = mdir(m, dn); f = c["fam"][md].get(FAMILY[m])
            ids = c["base"][md] + c["cls"][md] + ([f] if f is not None else []); blkc = A[len(nodes) + b]
            p = soft(logw[b] + (blkc + A[ids].sum(0)) @ PhiT); pc = soft(logw[b] + (blkc + A[c["cls"][md]].sum(0)) @ PhiT); pb = pbase[m, dn]
            pair.append([c["measures"].get(m, [0, 0])[k], round(float(0.5 * np.abs(p - pb).sum()), 3), round(float(0.5 * np.abs(p - pc).sum()), 3)] + regions(p, pb, m))
        c["cells"].append(pair)
# intervals: every cell's share per region in every fit on resampled papers
IV = None
if len(sys.argv) > 10:
    Zc = np.load(sys.argv[8]); Ad, Ld = Zc["A"], Zc["logw"]; nd = len(Ad) - 1; rho = float(json.load(open(sys.argv[9]))["rho"]); dev = torch.device("cuda:0"); NP, KR = 10, 16
    print("intervals:", nd, "draws; largest difference between draw 0 and this fit", float(np.abs(Ad[0].astype(np.float32) - A).max()), "rho", rho, flush=True)
    RM = max(len(GN), len(WN)); oh = lambda lab: torch.zeros(V, RM, device=dev).scatter_(1, torch.tensor(lab.astype(np.int64), device=dev)[:, None], 1.0)
    Rg_, Rw_ = oh(glab), oh(wlab); Phit = torch.tensor(Phi, dtype=torch.float32, device=dev)
    full, clso, cb, cwm = [], [], [], []
    for c in cons:
        for m in meas:
            for dn in ("less", "more"):
                b = bix[f"{m} | {'decrease' if dn == 'less' else 'increase'}"]; md = mdir(m, dn); f = c["fam"][md].get(FAMILY[m])
                full.append([len(nodes) + b] + c["base"][md] + c["cls"][md] + ([f] if f is not None else [])); clso.append([len(nodes) + b] + c["cls"][md]); cb.append(b); cwm.append(m in WM)
    NC = len(full); cb = np.array(cb); cwm = np.array(cwm)
    def sp(lists): r = np.repeat(np.arange(len(lists)), [len(x) for x in lists]); k = np.concatenate(lists); return torch.sparse_coo_tensor(np.stack([r, k]), np.ones(len(r), np.float32), (len(lists), A.shape[0])).coalesce().to(dev)
    Mf, Mc, Mb = sp(full), sp(clso), sp([[len(nodes) + b] for b in range(B)]); wmt = torch.tensor(cwm, device=dev)[:, None]; bwm = torch.tensor([n_.split(" | ")[0] in WM for n_ in names], device=dev)[:, None]
    SF = np.zeros((nd + 1, NC, RM), np.float32); SC = np.zeros_like(SF); SB = np.zeros((nd + 1, B, RM), np.float32)
    def shares(M, bb, wm_, At, Lw):
        co = torch.sparse.mm(M, At); out = []
        for k in range(0, len(bb), 1500):
            p = torch.softmax(Lw[bb[k:k + 1500]] + co[k:k + 1500] @ Phit.T, 1); out.append(torch.where(wm_[k:k + 1500], p @ Rw_, p @ Rg_))
        return torch.cat(out).cpu().numpy()
    with torch.no_grad():
        for d in range(nd + 1):
            At = torch.tensor(Ad[d].astype(np.float32), device=dev); Lw = torch.tensor(Ld[d].astype(np.float32), device=dev); bt = torch.tensor(cb, device=dev)
            SF[d] = shares(Mf, bt, wmt, At, Lw); SC[d] = shares(Mc, bt, wmt, At, Lw); SB[d] = shares(Mb, torch.arange(B, device=dev), bwm, At, Lw)
    okg = np.array([x not in SKIP and x != "White matter" for x in GN] + [False] * (RM - len(GN))); okw = np.array([x not in SKIP for x in WN] + [False] * (RM - len(WN)))
    ok = np.where(cwm[:, None], okw[None], okg[None]); est, pc0, pb0 = SF[0], SC[0], SB[0][cb]; eps = 1e-9
    def top(extra): extra = np.where(ok & (extra > 0), extra, -1.0); o = np.argsort(-extra, 1)[:, :KR // 2]; return np.where(np.take_along_axis(extra, o, 1) > 0, o, 255)
    idx = np.full((NC, KR), 255, np.int64)
    for n_, (a_, b_) in enumerate(zip(top(est - pb0), top(est - pc0))): u = [int(x) for x in dict.fromkeys(list(a_) + list(b_)) if x != 255]; idx[n_, :len(u)] = u
    has = idx != 255; ii = np.where(has, idx, 0); take = lambda X: np.take_along_axis(X, ii[None].repeat(len(X), 0) if X.ndim == 3 else ii, -1)
    f_, c_, b_ = take(SF), take(SC), take(SB[:, cb])                                  # draws x cells x KR
    def pct(x, x0): q = np.quantile(x[1:], [0.1, 0.9], axis=0) / np.maximum(x0, eps); return np.clip(np.rint(100 * q), 0, 255).astype(np.uint8)
    q_share = pct(f_, f_[0]); q_m = pct(f_ / np.maximum(b_, eps), f_[0] / np.maximum(b_[0], eps)); q_c = pct(f_ / np.maximum(c_, eps), f_[0] / np.maximum(c_[0], eps))
    # predictive: of NP peaks from one new paper, the count in the region; beta-binomial mixed over the draws
    y = torch.arange(NP + 1, dtype=torch.float64, device=dev); kk = (1 - rho) / rho; lb = lambda a_, b_: torch.lgamma(a_) + torch.lgamma(b_) - torch.lgamma(a_ + b_); plo, phi_ = [], []
    for k in range(0, NC, 1000):
        pi = torch.tensor(f_[1:, k:k + 1000], dtype=torch.float64, device=dev).clamp(1e-7, 1 - 1e-7)[..., None]
        lpm = torch.lgamma(torch.tensor(NP + 1.0, device=dev)) - torch.lgamma(y + 1) - torch.lgamma(NP - y + 1) + lb(y + pi * kk, NP - y + (1 - pi) * kk) - lb(pi * kk, (1 - pi) * kk)
        cdf = torch.exp(lpm).mean(0).cumsum(-1); plo.append((cdf < 0.1).sum(-1).clamp(max=NP).cpu().numpy()); phi_.append((cdf < 0.9).sum(-1).clamp(max=NP).cpu().numpy())
    plo, phi_ = np.concatenate(plo), np.concatenate(phi_)
    iv = np.stack([q_share[0], q_share[1], q_m[0], q_m[1], q_c[0], q_c[1], (plo * 16 + phi_).astype(np.uint8)], -1); iv[~has] = 0
    IV = (np.where(has, idx, 255).astype(np.uint8), iv)
    t1 = has[:, 0]; rel = (q_share[1][:, 0][t1].astype(float) - q_share[0][:, 0][t1]) / 200
    print(f"intervals: {NC} cells, {int(has.sum())} cell-regions; first region: half-width of the share interval median {np.median(rel):.3f} of the estimate; "
          f"estimate outside its interval in {float(((q_share[0] > 100) | (q_share[1] < 100))[has].mean()):.3f}; high bound clipped at 2.55x in {float((q_share[1] == 255)[has].mean()):.4f}; "
          f"relative to the measure: interval above 1 in {float((q_m[0][:, 0][t1] * (f_[0] / np.maximum(b_[0], eps))[:, 0][t1] > 100).mean()):.3f} of first regions", flush=True)
    R_ = json.load(open(sys.argv[10])); cvr = lambda k: {"independent80": R_[k]["binomial"]["expecting >= 1"]["80"], **{q: R_[k]["with paper correlation"]["expecting >= 1"][q] for q in ("50", "80", "95", "n")}}
    IVMETA = {"draws": nd, "level": 80, "rho": rho, "peaks": NP, "k": KR, "coverage": {"paper": cvr("paper boot"), "cell": cvr("cell boot")}}
# held-out-measure test, by cell (the same cells as checks.py --tests cell --seed 100)
base_ = lambda i: nodes[i].split(" | ")[0]; isc = lambda i: nodes[i].startswith("c:all:") and not is_cls(nodes[i])
dc = [sorted({base_(i) for i in C["auth"][d] if isc(i)}) for d in range(D)]; dmeas = [H["docs"][x]["measure"] for x in C["ids"]]
cm2 = collections.defaultdict(collections.Counter)
for d, cs in enumerate(dc):
    for c_ in cs: cm2[c_][dmeas[d]] += 1
elig = {c_: sorted(m for m, k in v.items() if k >= 8) for c_, v in cm2.items()}; elig = {c_: v for c_, v in elig.items() if len(v) >= 3}
rng = np.random.default_rng(100); order_ = {c_: list(rng.permutation(v)) for c_, v in sorted(elig.items())}; S = np.load(cell_npy); pos = 0; acc = collections.defaultdict(lambda: np.zeros(2))
for rnd in range(3):
    pick = {c_: v[rnd % len(v)] for c_, v in order_.items()}; te = [d for d in range(D) if any(pick.get(c_) == dmeas[d] for c_ in dc[d]) and len(C["foci"][d])]
    rows_ = S[pos:pos + len(te)]; pos += len(te); assert np.array_equal(rows_[:, 0].astype(int), te), "cell scores do not match the held-out cells"
    for d, m_, w_, n_ in rows_[:, :4]:
        for c_ in dc[int(d)]:
            if pick.get(c_) == dmeas[int(d)]: acc[c_[6:], dmeas[int(d)]] += (m_ - w_, n_)
assert pos == len(S)
for c in cons: c["transfer"] = {m: [round(float(v[0] / v[1]), 3), int(v[1])] for (c_, m), v in acc.items() if c_ == c["name"] and v[1] >= 15}
order = sorted(need); remap = {j: k for k, j in enumerate(order)}
for c in cons:
    for key in ("base", "cls"): c[key] = {dn: [remap[j] for j in v] for dn, v in c[key].items()}
    c["fam"] = {k: {f: remap[j] for f, j in v.items()} for k, v in c["fam"].items()}
for z in classes: z["ids"] = {dn: [remap[j] for j in v] for dn, v in z["ids"].items()}
blocks = {k: remap[j] for k, j in blocks.items()}
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
if IV is not None: add("iv_idx", IV[0]); add("iv", IV[1])
open(os.path.join(outd, "maps.bin"), "wb").write(b"".join(parts))
T = json.load(open(tests_json)); tests = {k: T[k] for k in ("paper", "paper_ci", "paper_per_contrast", "cell", "cell_won", "future", "cell_by_measure") if k in T}
Dj = {"meta": {"model": "regression", "variant": VARIANT, "pen": PEN, "lam": lam, "contrasts": D, "papers": len({d["group"] for d in H["docs"].values()}), "peaks": int(sum(len(f) for f in C["foci"])),
               "conditions_all": len(tot), "conditions_shown": len(keep), "min_contrasts": MIN_C, "measures": meas, "labels": LABEL, "family": FAMILY, "flip": sorted(K.FLIP),
               "wm_measures": WM, "gm_names": GN, "wm_names": WN, "wm_atlas": wm_atlas, "grid": {"shape": shape, "mm": 4},
               "layout": layout, "bumps": int(Phi.shape[1]), "blocks": blocks, "tests": tests, "strength_q": [float(np.quantile(st, q)) for q in (0.33, 0.67)]},
      "conditions": cons, "classes": classes, "pairs": pairs}
if IV is not None: Dj["meta"]["intervals"] = IVMETA; Dj["meta"]["layout"] = layout
json.dump(Dj, open(os.path.join(outd, "data.json"), "w"), separators=(",", ":"))
print(len(keep), "conditions,", len(classes), "classes,", len(order), "labels; data.json", round(os.path.getsize(os.path.join(outd, "data.json")) / 1e6, 2), "MB; maps.bin", round(off[0] / 1e6, 2), "MB")
nm_of = lambda m, k: (WN if m in WM and WN is not None else GN)[k]
for nm, m in (("alzheimer disease", "grey matter"), ("alzheimer disease", "glucose metabolism"), ("parkinson disease", "grey matter"), ("schizophrenia", "grey matter"), ("amyotrophic lateral sclerosis", "fa")):
    if nm in cid:
        c = cons[cid[nm]]; cl = c["cells"][meas.index(m)]; print(f"{nm} | {m}: less n={cl[0][0]} moved {cl[0][1]} {[nm_of(m, k) for k in cl[0][3:]]}; more n={cl[1][0]} moved {cl[1][1]} {[nm_of(m, k) for k in cl[1][3:]]}; transfer {c['transfer']}")
print("pairs:", [(cons[i]["name"], cons[j]["name"], r) for i, j, r in pairs[:12]])
gaps = sorted(((cl[k][2], c["name"], m, ("less", "more")[k]) for c in cons for m, cl in zip(meas, c["cells"]) for k in (0, 1) if cl[k][0] == 0 and c["own"][mdir(m, ("less", "more")[k])]), reverse=True)
print(len(gaps), "untested cells with the condition's own estimate in that direction; top:", gaps[:10])
