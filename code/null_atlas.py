"""How often would the condition atlas page show a condition-specific pattern by chance? (2026-11-16)

Each fit refits the kept atlas model (export_site_hyp.py: full+dirsplit+meaning, ridge 20, family level 5, window from
peaks counted once) on all contrasts, then computes for every (condition, measure, reported direction) cell with at least
one contrast the two numbers the page shows: the share of predicted peaks that move against the measure's usual pattern
(total variation distance between the cell's map and window + block) and against the condition's class (cell map
against window + block + class labels).
modes:
  real   the corpus as is;
  any    each unit (paper x its diagnosis set) gets the diagnosis set of another unit, drawn at random over all units;
  class  the same, but only among units whose diagnoses are in the same classes (the null for "beyond its class").
A contrast keeps its measure, direction, peaks and measure family; only which condition it belongs to changes. Label
nodes that the new combination needs (condition | direction | family) are added to the hierarchy.
env PW_WIDTH=<mm> [PW_MIRROR=1]: fit with distance-weighted peaks (checks.py --peak-width).
usage: null_atlas.py CORPUS VOCABULARY.json OUT_DIR MODE SEED [SEED ...]   (writes OUT_DIR/<mode>_<seed>.json)"""
import collections, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks as K, pc_model as P
from task_networks.corpus import load_corpus
from build_corpus import FAMILY
import os
PW = float(os.environ.get("PW_WIDTH", "0")); PW_MIRROR = os.environ.get("PW_MIRROR", "0") == "1"     # optional: fit with distance-weighted peaks
corpus, vocab, outd, mode = sys.argv[1:5]; seeds = [int(x) for x in sys.argv[5:]]; Path(outd).mkdir(parents=True, exist_ok=True)
H0 = json.load(open(corpus)); C0 = load_corpus(corpus); blk, names = P.blocks_of(H0, C0, "measure_dir"); B = len(names); bix = {b: i for i, b in enumerate(names)}
voc = {v["condition"]: v for v in json.load(open(vocab))}; ids = C0["ids"]; D = len(ids)
is_cond = lambda i: i.startswith("c:all:") and not i[6:].startswith(("class ", "any condition"))
def parse(d):
    cs = [i for i in H0["docs"][d]["ids"] if is_cond(i)]; rest = [i for i in H0["docs"][d]["ids"] if not is_cond(i)]
    bases = tuple(sorted({i[6:].split(" | ")[0] for i in cs})); suf = cs[0][6:].split(" | ", 1)[1] if cs and " | " in cs[0][6:] else None
    return bases, suf, rest
info = {d: parse(d) for d in ids}
def relabel(rng):
    units = sorted({(H0["docs"][d]["group"], info[d][0]) for d in ids}); grpkey = lambda u: tuple(sorted({(voc.get(b) or {}).get("class") or "other" for b in u[1]})) if mode == "class" else 0
    by = collections.defaultdict(list)
    for u in units: by[grpkey(u)].append(u)
    new = {}
    for g, us in by.items():
        perm = rng.permutation(len(us)); new.update({u: us[p][1] for u, p in zip(us, perm)})
    H = {"docs": {}, "parents": dict(H0["parents"])}
    for d in ids:
        bases, suf, rest = info[d]; nb = new[(H0["docs"][d]["group"], bases)] if bases else bases; cs = []
        for b in nb:
            node = f"c:all:{b} | {suf}" if suf else f"c:all:{b}"
            if node not in H["parents"]:            # add the missing condition | direction (| family) nodes under the condition
                parts = node.split(" | ")
                for k in range(2, len(parts) + 1):
                    x = " | ".join(parts[:k])
                    if x not in H["parents"]: H["parents"][x] = [" | ".join(parts[:k - 1])]
            cs.append(node)
        H["docs"][d] = dict(H0["docs"][d], ids=cs + rest)
    return H
def corpus_of(H):
    C = dict(C0); nodes = sorted(H["parents"]); nidx = {n: i for i, n in enumerate(nodes)}
    C["nodes"] = nodes; C["parents"] = [[nidx[p] for p in H["parents"][n]] for n in nodes]; C["auth"] = [np.array([nidx[n] for n in H["docs"][d]["ids"]]) for d in ids]
    return C
def soft(lg): lg = lg - lg.max(-1, keepdims=True); p = np.exp(lg); return p / p.sum(-1, keepdims=True)
mdir = lambda m, dn: ({"less": "more", "more": "less"}[dn] if m in K.FLIP else dn)
def one(H, tag):
    C = corpus_of(H); K.FIT.clear(); K.FIT.update(names=names, wfrom="foci", opt="lbfgs", outer=15, pen={"fam": 5.0})
    if PW > 0: K.FIT["pw"] = K.peak_weights(C, PW, PW_MIRROR)      # distance-weighted peaks (checks.py --peak-width)
    Phi = K.basis(C["xyz"]); rows, cols, vals, F = K.features(C, H, blk, B, "full+dirsplit+meaning", names); ix = {k: i for i, k in enumerate(K.FIT["keys"])}
    X = torch.zeros(D, F); X[torch.tensor(rows), torch.tensor(cols)] = torch.tensor(vals, dtype=torch.float32)
    rk = collections.defaultdict(list)
    for r_, c_, v_ in zip(rows, cols, vals): rk[r_].append((c_, round(v_, 6)))
    gk = {}; K.FIT["gid"] = np.array([gk.setdefault(tuple(sorted(rk[d])), len(gk)) for d in range(D)])
    K.fit_additive(C, blk, B, Phi, X, list(range(D)), 20.0, torch.device("cuda:0")); A = K.FIT["A"]; logw = K.FIT["logw"]; nodes = C["nodes"]; par = C["parents"]; nix = {x: i for i, x in enumerate(nodes)}
    cm = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for d in ids:
        doc = H["docs"][d]
        for c in {i[6:].split(" | ")[0] for i in doc["ids"] if is_cond(i)}: cm[c][doc["measure"]][doc["direction"] == "increase"] += 1
    tot = {c: sum(a + b for a, b in v.values()) for c, v in cm.items()}
    keep = [c for c in tot if tot[c] >= 15 and c != "unspecified condition" and "c:all:" + c in nix]
    def anc(i):
        out, st = [], list(par[i])
        while st:
            j = st.pop()
            if j not in out: out.append(j); st += par[j]
        return out
    isc = lambda x: x[6:].startswith(("class ", "any condition")); split = lambda js, dn: [ix[nodes[j] + " | " + dn] for j in js if nodes[j] + " | " + dn in ix]
    PhiT = Phi.T.astype(np.float64); meas = sorted({b.split(" | ")[0] for b in names}); cells = []
    for c in keep:
        b0 = nix["c:all:" + c]; a = anc(b0); bn = [b0] + [j for j in a if not isc(nodes[j])]; cn = [j for j in a if isc(nodes[j])]
        for m in meas:
            for k, (dn, dr) in enumerate((("less", "decrease"), ("more", "increase"))):
                n = cm[c].get(m, [0, 0])[k]
                if n < 1: continue
                b = bix[f"{m} | {dr}"]; md = mdir(m, dn); fk = ix.get(f"c:all:{c} | {md} | {FAMILY[m]}"); base = split(bn, md); cls = split(cn, md); blkc = A[len(nodes) + b]
                p = soft(logw[b] + (blkc + A[base + cls + ([fk] if fk is not None else [])].sum(0)) @ PhiT); pc = soft(logw[b] + (blkc + A[cls].sum(0)) @ PhiT); pb = soft(logw[b] + blkc @ PhiT)
                cells.append([c, m, dn, int(n), round(float(0.5 * np.abs(p - pb).sum()), 4), round(float(0.5 * np.abs(p - pc).sum()), 4)])
    json.dump({"mode": mode, "tag": tag, "conditions": len(keep), "cells": cells, "train_ll": K.FIT["train_ll"]}, open(f"{outd}/{tag}.json", "w"))
    print(tag, len(keep), "conditions", len(cells), "cells; median TV vs usual", np.median([x[4] for x in cells]), "vs class", np.median([x[5] for x in cells]), flush=True)
for s in seeds:
    one(H0 if mode == "real" else relabel(np.random.default_rng(s)), f"{mode}_{s}")
