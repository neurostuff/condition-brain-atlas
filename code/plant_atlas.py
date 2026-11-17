"""Planted-signal check of the condition atlas model (2026-11-16): null data in, null result out? Planted signal found,
and with how few studies?

Data: grey matter contrasts only (4,430; both directions), the kept model otherwise (full+dirsplit+meaning, ridge 20,
family level 5, window from peaks counted once), optionally with distance-weighted peaks (env PW_WIDTH, checks.py
--peak-width). Each draw: k papers are picked from the single-diagnosis mood-class grey-matter-decrease contrasts (360
contrasts, 232 papers), one contrast each, and relabelled as a new condition "planted condition" in the mood class. A
blob centre is drawn among voxels of middling grey-matter peak density (33rd-66th percentile; env BLOB_Q=0.95: among
the most reported voxels, at or above that quantile), and in each picked
contrast a share F of its peaks (at least one when F > 0) is replaced by points scattered around the centre (Gaussian,
sd 6 mm, snapped to the grid). F = 0 is the null: the same relabelling, peaks untouched. The model is refitted on all grey
matter contrasts; the planted cell's map (grey matter, less) is compared with its class's map: the largest log ratio
over voxels and where it is, the log ratio at the blob centre, and the page's "share of peaks moving against the class"
(total variation). Draws are seeded by (k, F, draw), so every model sees the same planted data.
Optional per draw: K:F:DRAW:SD:H, SD the blob's scatter in mm (default 6) and H the share of the k studies that carry
the effect (default 1); F = 0.01 means "one peak per carrying study" for typical peak counts.
usage: plant_atlas.py CORPUS OUT_DIR K:F:DRAW[:SD:H] [...]   (writes OUT_DIR/<tag>_k<K>_f<F>_d<DRAW>.json; coordinates for
ALE to OUT_DIR/coords/k<K>_f<F>_d<DRAW>.json)"""
import collections, json, os, sys, zlib
from pathlib import Path
import numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks as K, pc_model as P
from task_networks.corpus import load_corpus
corpus, outd = sys.argv[1:3]; specs = [tuple(x.split(":")) for x in sys.argv[3:]]; PW = float(os.environ.get("PW_WIDTH", "0")); tag = f"w{PW:g}" if PW else "none"
Path(outd, "coords").mkdir(parents=True, exist_ok=True)
H0 = json.load(open(corpus)); C0 = load_corpus(corpus); blk0, names = P.blocks_of(H0, C0, "measure_dir"); B = len(names); bix = {b: i for i, b in enumerate(names)}
keep = [d for d, x in enumerate(C0["ids"]) if H0["docs"][x]["measure"] == "grey matter"]; ids = [C0["ids"][d] for d in keep]; blk = np.array([blk0[d] for d in keep]); D = len(keep)
xyz = C0["xyz"]; tree = cKDTree(xyz); CLASS = "c:all:class mood"; FAKE = "c:all:planted condition"
def anc(n, P_):
    out, st = [], list(P_.get(n, []))
    while st:
        x = st.pop()
        if x not in out: out.append(x); st += P_.get(x, [])
    return out
isc = lambda i: i.startswith("c:all:") and not i[6:].startswith(("class ", "any condition"))
src = [j for j, x in enumerate(ids) if H0["docs"][x]["direction"] == "decrease" and len([i for i in H0["docs"][x]["ids"] if isc(i)]) == 1
       and any(CLASS == a for i in H0["docs"][x]["ids"] if isc(i) for a in anc(i, H0["parents"]))]
srcp = collections.defaultdict(list)
for j in src: srcp[H0["docs"][ids[j]]["group"]].append(j)
papers = sorted(srcp)
gm_dec = [j for j, x in enumerate(ids) if H0["docs"][x]["direction"] == "decrease"]; dens = np.zeros(len(xyz))
for j in gm_dec: np.add.at(dens, C0["toks"][keep[j]], 1.0)
nz = np.where(dens > 0)[0]; BLOB_Q = float(os.environ.get("BLOB_Q", "0"))      # BLOB_Q > 0: centres among the most reported voxels (density at or above that quantile)
lo, hi = np.quantile(dens[nz], [BLOB_Q, 1.0] if BLOB_Q else [1 / 3, 2 / 3]); cand = nz[(dens[nz] >= lo) & (dens[nz] <= hi)]
print(f"{D} grey matter contrasts; planted from {len(src)} contrasts in {len(papers)} papers; {len(cand)} candidate blob voxels; model {tag}", flush=True)
def soft(lg): lg = lg - lg.max(-1, keepdims=True); p = np.exp(lg); return p / p.sum(-1, keepdims=True)
Phi = K.basis(xyz); PhiT = Phi.T.astype(np.float64)
for sp in specs:
    k, f, dr = int(sp[0]), float(sp[1]), int(sp[2]); sd = float(sp[3]) if len(sp) > 3 else 6.0; hh = float(sp[4]) if len(sp) > 4 else 1.0
    extra = "" if len(sp) <= 3 else f"_sd{sd:g}_h{hh:g}"; rng = np.random.default_rng(zlib.crc32(f"{k}:{f}:{dr}{extra}".encode()))
    pick = [int(rng.choice(srcp[p])) for p in rng.choice(papers, k, replace=False)]; centre = int(rng.choice(cand))
    H = {"docs": {}, "parents": dict(H0["parents"])}
    H["parents"][FAKE] = [CLASS]; H["parents"][FAKE + " | less"] = [FAKE]; H["parents"][FAKE + " | less | structure"] = [FAKE + " | less"]
    foci = [np.asarray(C0["foci"][d]) for d in keep]
    for j, x in enumerate(ids): H["docs"][x] = H0["docs"][x]
    carry = set(pick[:max(1, int(round(hh * k)))]) if f > 0 else set()
    for j in pick:
        x = ids[j]; H["docs"][x] = dict(H0["docs"][x], ids=[FAKE + " | less | structure"] + [i for i in H0["docs"][x]["ids"] if not isc(i)])
        if j in carry:
            fo = foci[j]; m = max(1, int(round(f * len(fo)))); stay = rng.permutation(len(fo))[m:]
            new = tree.query(xyz[centre] + rng.normal(0, sd, (m, 3)))[1]; foci[j] = np.unique(np.concatenate([fo[stay], new]))
    nodes = sorted(H["parents"]); nidx = {n: i for i, n in enumerate(nodes)}
    C = {"ids": ids, "xyz": xyz, "foci": foci, "toks": foci, "nodes": nodes, "parents": [[nidx[q] for q in H["parents"][n]] for n in nodes],
         "auth": [np.array([nidx[n] for n in H["docs"][x]["ids"]]) for x in ids]}
    K.FIT.clear(); K.FIT.update(names=names, wfrom="foci", opt="lbfgs", outer=15, pen={"fam": 5.0}, pw=K.peak_weights(C, PW) if PW else None)
    rows, cols, vals, F = K.features(C, H, blk, B, "full+dirsplit+meaning", names); ix = {kk: i for i, kk in enumerate(K.FIT["keys"])}
    X = torch.zeros(D, F); X[torch.tensor(rows), torch.tensor(cols)] = torch.tensor(vals, dtype=torch.float32)
    rk = collections.defaultdict(list)
    for r_, c_, v_ in zip(rows, cols, vals): rk[r_].append((c_, round(v_, 6)))
    gk = {}; K.FIT["gid"] = np.array([gk.setdefault(tuple(sorted(rk[d])), len(gk)) for d in range(D)])
    K.fit_additive(C, blk, B, Phi, X, list(range(D)), 20.0, torch.device("cuda:0")); A = K.FIT["A"]; logw = K.FIT["logw"]
    b = bix["grey matter | decrease"]; blkc = A[len(nodes) + b]
    lab = lambda names_: [ix[n] for n in names_ if n in ix]
    cls = lab([CLASS + " | less", "c:all:any condition | less"]); own = lab([FAKE + " | less", FAKE + " | less | structure"])
    p = soft(logw[b] + (blkc + A[cls + own].sum(0)) @ PhiT); pc = soft(logw[b] + (blkc + A[cls].sum(0)) @ PhiT); pb = soft(logw[b] + blkc @ PhiT)
    lr = np.log(p) - np.log(pc); v = int(np.argmax(lr))
    out = {"model": tag, "k": k, "f": f, "draw": dr, "sd": sd, "h": hh, "carrying": len(carry), "centre": xyz[centre].round().tolist(), "labels_found": len(own), "max_log_ratio": float(lr[v]), "max_at": xyz[v].round().tolist(),
           "max_distance_mm": float(np.linalg.norm(xyz[v] - xyz[centre])), "log_ratio_at_centre": float(lr[centre]), "tv_vs_class": float(0.5 * np.abs(p - pc).sum()), "tv_vs_usual": float(0.5 * np.abs(p - pb).sum()),
           "planted_peaks": int(sum(len(foci[j]) for j in pick)), "contrasts": pick}
    json.dump(out, open(f"{outd}/{tag}_k{k}_f{f:g}_d{dr}{extra}.json", "w"))
    cp = f"{outd}/coords/k{k}_f{f:g}_d{dr}{extra}.json"
    if not os.path.exists(cp): json.dump({"centre": out["centre"], "experiments": {ids[j]: xyz[foci[j]].tolist() for j in pick}}, open(cp, "w"))
    print(tag, k, f, dr, "max log ratio %.3f at %.0f mm from the blob; at blob %.3f; tv vs class %.3f" % (out["max_log_ratio"], out["max_distance_mm"], out["log_ratio_at_centre"], out["tv_vs_class"]), flush=True)
