"""Two cheap checks before building a new model (2026-11-14; results/model_search_summary.md).

additive: regularised additive regression on labels with a point-process likelihood (each peak once).
    P(peak at v | contrast j) = softmax_v( log window[block_j, v] + sum_f x[j, f] (A[f] . Phi[v]) )
  window: the measure's own map from the training contrasts (pc_model.windows, as the topic model's baseline), so A = 0 is
  exactly "knowing the measure alone". Phi: Gaussian bumps (sigma SIG mm) on a grid every STEP voxels. Features of a
  contrast: each of its condition authors in the multimodal_fam corpus and all their ancestors (condition | direction |
  family -> condition | direction -> condition -> broader condition -> class -> any condition), weight 1 / (number of
  condition authors); the region-restricted author; its block (measure x direction). Ridge penalty LAM on A.
  Variants drop levels: full | nofam (no family level: sharing across measures assumed) | nodir (condition and above) | class.
  Tests as pc_model.py: paper (5 folds), cell (held-out measure for a condition, 3 rounds, same cells as seed SEED there),
  future (train before CUT). Reports gain per held-out peak over the window.
  usage: checks.py additive CORPUS OUT.json [--variant full] [--lam 3] [--tests paper,cell,future] [--seed 100] [--cut 2020]
  Additions under test (MODEL_PLAN.md), each off by default:
    --pen any=..,class=..,cond=..,dir=..,fam=..,block=..,scope=..,factor=..   a ridge penalty per level (default LAM for all)
    --variant full+dirsplit   condition, broader condition, class and "any condition" labels are split by direction
    --variant full+meaning    direction recoded by meaning: more diffusivity counts with less of everything else
    --variant full+couple --rank R   the condition | direction and condition | direction | family labels are replaced by
        R maps per condition, mixed into each of the 28 blocks by a learned loading matrix (columns scaled to mean square
        one half; the first two start as the less and more indicators, so R = 2 with --freeze-l is the nofam variant)
    --tests inner             3-fold paper test within the papers before CUT (for choosing penalties without the test data)
    --tests ic                fit on everything; AIC and BIC with the effective number of parameters of the penalised fit,
        edf = trace((H + P)^-1 H), H the Hessian of the negative log-likelihood and P of the penalty, estimated with
        Rademacher probes and conjugate gradients (Hutchinson). n for BIC = peaks; also given with n = papers.

coupling: is a condition's pattern in one measure informative about its pattern in another, pair by pair?
  For every cell (condition, measure, direction) with >= MINC single- or multi-condition contrasts: its peak density on Phi
  minus the block's own density on Phi. For blocks a, b: mean over conditions with both cells of corr(dev[c, a], dev[c, b])
  (matched) minus the same with conditions mismatched (all other pairs): the coupling. Permutation p from shuffling
  conditions within block b.
  usage: checks.py coupling CORPUS OUT.json [--minc 8]
"""
import argparse, collections, json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pc_model as P
from task_networks.corpus import load_corpus

torch.set_default_dtype(torch.float32)
STEP, SIG = 3, 8.0


def basis(xyz):
    """Gaussian bumps of width SIG mm centred every STEP grid voxels (4 mm each); both set from the command line"""
    ijk = np.rint((xyz - xyz.min(0)) / 4).astype(int); keep = np.all(ijk % STEP == 1, 1); cen = xyz[keep]
    d2 = ((xyz[:, None, :] - cen[None]) ** 2).sum(-1); Phi = np.exp(-d2 / (2 * SIG ** 2)); Phi[d2 > (3 * SIG) ** 2] = 0
    return Phi.astype(np.float32)


FLIP = {"diffusivity"}     # measures whose "more" means the damage that "less" means elsewhere (--variant ...+meaning)


def features(C, H, blk, B, variant, names=None):
    """rows, cols, vals of the contrast x label matrix, the number of labels, each label's level, and (with +couple) the
    contrast x condition weights of the labels that the learned factors replace"""
    nodes = C["nodes"]; par = C["parents"]; keys = {x: i for i, x in enumerate(nodes)}; rows, cols, vals = [], [], []; grp = {}
    for b in range(B): keys["block:%d" % b] = len(nodes) + b; grp[len(nodes) + b] = "block"
    v0, *o = variant.split("+"); opts = set(o); assert opts <= {"dirsplit", "meaning", "couple"}, opts
    conds = {}; crow, ccol, cval = [], [], []
    def kid(k, g):
        if k not in keys: keys[k] = len(keys)
        grp[keys[k]] = g; return keys[k]
    for d in range(len(C["ids"])):
        ca = [a for a in C["auth"][d] if nodes[a].startswith("c:")]; w = 1.0 / max(len(ca), 1); acc = collections.Counter(); cacc = collections.Counter()
        if names is not None:
            m, dr = names[blk[d]].split(" | "); dr = {"decrease": "less", "increase": "more"}[dr]
            if "meaning" in opts and m in FLIP: dr = {"less": "more", "more": "less"}[dr]
        for a in ca:
            stack = [a]; seen = set(); ks = {}; cs = set()
            while stack:
                i = stack.pop()
                if i in seen: continue
                seen.add(i); stack += par[i]; x = nodes[i]; parts = x.split(" | "); dep = len(parts) - 1; base = parts[0][6:]
                if v0 == "nofam" and dep == 2: continue
                if v0 == "nodir" and dep >= 1: continue
                if v0 == "class" and not base.startswith(("class ", "any condition")): continue
                if "couple" in opts and dep >= 1: cs.add(parts[0]); continue
                g = "any" if base.startswith("any condition") else "class" if base.startswith("class ") else ("cond", "dir", "fam")[dep]; k = x
                if "meaning" in opts and dep >= 1: parts[1] = dr; k = " | ".join(parts)
                if "dirsplit" in opts and dep == 0: k = x + " | " + dr; g = "dir" if k in ks else g
                ks.setdefault(k, g)
            for k, g in ks.items(): acc[kid(k, g)] += w
            for c in cs: cacc[conds.setdefault(c, len(conds))] += w
        for a in C["auth"][d]:
            if nodes[a] == "m:scope:restricted": acc[a] += 1.0; grp[a] = "scope"
        acc[len(nodes) + int(blk[d])] += 1.0
        for i, v in acc.items(): rows.append(d); cols.append(i); vals.append(v)
        for i, v in cacc.items(): crow.append(d); ccol.append(i); cval.append(v)
    FIT["groups"] = [grp.get(i, "unused") for i in range(len(keys))]; FIT["keys"] = list(keys); FIT["couple"] = (crow, ccol, cval, len(conds)) if "couple" in opts else None
    return rows, cols, vals, len(keys)


def fit_additive(C, blk, B, Phi, X, tr, lam, dev, steps=300, lr=0.03, chunk=int(os.environ.get("PC_CHUNK", 1500))):   # rows per batch: lower it when GPU memory is short
    # FIT["wfrom"]: "toks" (default; the window from each peak's 10 mm sphere, as the topic model) or "foci" (each peak once)
    # FIT["opt"]: "lbfgs" (default; FIT["outer"] rounds of up to 20 iterations) or "adam" (steps passes)
    # Contrasts with the same labels and block have the same predicted map, so the map is computed once per distinct row
    # (FIT["gid"], set in main) and every contrast's peaks are read from it: exact, and about 2.3 times less work.
    """returns a function scoring documents: (sum log p at peaks under the model, under the window, number of peaks)"""
    V = Phi.shape[0]; hist = np.zeros((B, V))
    for d in tr:
        src = C["foci"][d] if FIT.get("wfrom") == "foci" else C["toks"][d]
        if len(src): np.add.at(hist[blk[d]], np.asarray(src), 1.0)
    names = FIT["names"]; ms = [x.split(" | ")[0] for x in names]; ijk = np.rint((C["xyz"] - C["xyz"].min(0)) / 4).astype(int)
    w = P.windows(hist, [ms.index(x) for x in ms], (ijk, tuple(ijk.max(0) + 1)), 1.5, 0.02); logw = torch.tensor(np.log(w), dtype=torch.float32, device=dev)
    Phit = torch.tensor(Phi, device=dev); Xd = X.to(dev); A = torch.zeros(Xd.shape[1], Phi.shape[1], device=dev, requires_grad=True)
    pv_ = FIT.get("pen") or {}; penv = torch.tensor([pv_.get(g, lam) for g in FIT["groups"]], dtype=torch.float32, device=dev)[:, None] if "groups" in FIT else lam
    params = [A]; Xc = FIT.get("Xc"); R = FIT.get("rank", 0) if Xc is not None else 0
    if R:
        # learned coupling: contrast j in block b gets sum_r Leff[b, r] * (sum_c Xc[j, c] Z[r, c]) on top of the fixed labels
        Xc = Xc.to(dev); Z = torch.zeros(R, Xc.shape[1], Phi.shape[1], device=dev, requires_grad=True); g_ = torch.Generator().manual_seed(FIT.get("lseed", 0))
        L0 = 0.5 * torch.randn(B, R, generator=g_); dirs = torch.tensor([x.endswith("increase") for x in FIT["names"]])
        L0[:, 0] = (~dirs).float()
        if R > 1: L0[:, 1] = dirs.float()
        L = L0.to(dev).requires_grad_(not FIT.get("freeze_l")); params += [Z] + ([L] if not FIT.get("freeze_l") else []); lamz = pv_.get("factor", lam)
        Leff = lambda: L * (0.5 / (L ** 2).mean(0, keepdim=True)).sqrt()
    def linear(ds, bb):
        XA = Xd[ds] @ A
        if R:
            T = Xc[ds]; Le = Leff()[bb]
            for r in range(R): XA = XA + Le[:, r:r + 1] * (T @ Z[r])
        return XA
    gid = FIT["gid"] if FIT.get("gid") is not None else np.arange(len(C["ids"])); blk_ = np.asarray(blk)
    def pack(ds):
        """batches of at most chunk distinct rows: (a contrast per row, row of each peak, voxel of each peak, block per row, contrast position of each peak)"""
        ds = np.asarray([d for d in ds if len(C["foci"][d])], dtype=int); out = []
        if not len(ds): return out, ds
        u, inv = np.unique(gid[ds], return_inverse=True); pos = np.arange(len(ds))
        for k in range(0, len(u), chunk):
            sel = (inv >= k) & (inv < k + chunk); rep_ = np.zeros(min(chunk, len(u) - k), dtype=int); rep_[inv[sel] - k] = ds[sel]; pr, pv, pp = [], [], []
            for j, d, q in zip(inv[sel] - k, ds[sel], pos[sel]):
                f = C["foci"][d]; pr += [j] * len(f); pv += list(f); pp += [q] * len(f)
            out.append(tuple(torch.tensor(x, device=dev) for x in (rep_, pr, pv, blk_[rep_], pp)))
        return out, ds
    batches, trd = pack(tr)
    npk = sum(len(b[1]) for b in batches); state = {"tot": 0.0}
    def closure():
        opt.zero_grad(); tot = 0.0
        for ds, pd_, pv, bb, _ in batches:
            lp = torch.log_softmax(logw[bb] + linear(ds, bb) @ Phit.T, 1)[pd_, pv].sum(); (-lp / npk).backward(); tot += float(lp.detach())
        pen = (penv * A ** 2).sum() / npk + (lamz * (Z ** 2).sum() / npk if R else 0.0); pen.backward(); state["tot"] = tot
        return torch.tensor(-tot / npk) + pen.detach()
    if FIT.get("opt", "lbfgs") == "lbfgs":
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=20, history_size=20, line_search_fn="strong_wolfe")
        for it in range(FIT.get("outer", 15)): opt.step(closure)
    else:
        opt = torch.optim.Adam(params, lr=lr)
        for it in range(steps): closure(); opt.step()
    tot = state["tot"]
    FIT["train_ll"] = tot / npk
    FIT["A"] = A.detach().cpu().numpy(); FIT["logw"] = np.log(w)      # label coefficients (features x bumps) and log windows (blocks x voxels)
    FIT["size"] = {g: float((A.detach()[[i for i, x in enumerate(FIT["groups"]) if x == g]] ** 2).mean()) for g in sorted(set(FIT["groups"])) if g != "unused"} if "groups" in FIT else {}
    if R: FIT["L"] = Leff().detach().cpu().numpy(); FIT["size"]["factor"] = float((Z.detach() ** 2).mean())
    if FIT.get("want_ic"):
        pens = [2 * penv * torch.ones_like(A)] + ([2 * lamz * torch.ones_like(Z)] if R else []) + ([1e-3 * torch.ones_like(L)] if R and not FIT.get("freeze_l") else [])
        dot = lambda x, y: sum(float((a_ * b_).sum()) for a_, b_ in zip(x, y))
        def hvp(vs):
            o = [torch.zeros_like(p_) for p_ in params]
            for ds, pd_, pv, bb, _ in batches:
                nll = -torch.log_softmax(logw[bb] + linear(ds, bb) @ Phit.T, 1)[pd_, pv].sum(); g = torch.autograd.grad(nll, params, create_graph=True)
                for o_, h_ in zip(o, torch.autograd.grad(sum((a_ * b_).sum() for a_, b_ in zip(g, vs)), params)): o_ += h_
            return o
        gen = torch.Generator(device=dev).manual_seed(0); est = []
        for probe in range(FIT.get("probes", 6)):
            v = [(torch.randint(0, 2, p_.shape, generator=gen, device=dev) * 2 - 1).float() for p_ in params]; b = hvp(v)
            x = [torch.zeros_like(p_) for p_ in params]; r = [t.clone() for t in b]; z = [r_ / q_ for r_, q_ in zip(r, pens)]; pdir = [t.clone() for t in z]; rz = dot(r, z); r0 = dot(r, r) ** 0.5
            for it in range(150):          # conjugate gradients on (H + P) x = H v, preconditioned by the penalty
                Ap = [h_ + q_ * p_ for h_, q_, p_ in zip(hvp(pdir), pens, pdir)]; al = rz / dot(pdir, Ap)
                x = [x_ + al * p_ for x_, p_ in zip(x, pdir)]; r = [r_ - al * a_ for r_, a_ in zip(r, Ap)]
                if dot(r, r) ** 0.5 < 1e-3 * r0: break
                z = [r_ / q_ for r_, q_ in zip(r, pens)]; rz2 = dot(r, z); pdir = [z_ + (rz2 / rz) * p_ for z_, p_ in zip(z, pdir)]; rz = rz2
            est.append(dot(v, x)); print(f"  edf probe {probe}: {est[-1]:.0f} ({it + 1} CG steps)", flush=True)
        edf = float(np.mean(est)); wll = float(sum(logw[bb][pd_, pv].sum() for ds, pd_, pv, bb, _ in batches))
        FIT["ic"] = {"peaks": npk, "loglik": tot, "window_loglik": wll, "edf": edf, "edf_se": float(np.std(est) / len(est) ** 0.5), "parameters": int(sum(p_.numel() for p_ in params)),
                     "AIC": -2 * tot + 2 * edf, "BIC_peaks": -2 * tot + np.log(npk) * edf, "window_AIC": -2 * wll}
    def score(ds):
        bt, dd = pack(ds); m = torch.zeros(len(dd), device=dev); wv = torch.zeros(len(dd), device=dev)
        with torch.no_grad():
            for rows_, pd_, pv, bb, pp in bt:
                m.index_add_(0, pp, torch.log_softmax(logw[bb] + linear(rows_, bb) @ Phit.T, 1)[pd_, pv]); wv.index_add_(0, pp, logw[bb][pd_, pv])
        return [(int(d), float(x), float(y), len(C["foci"][d])) for d, x, y in zip(dd, m.cpu(), wv.cpu())]
    return score


FIT = {}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("corpus"); ap.add_argument("out")
    ap.add_argument("--variant", default="full"); ap.add_argument("--lam", type=float, default=3.0); ap.add_argument("--tests", default="paper,cell,future")
    ap.add_argument("--seed", type=int, default=100); ap.add_argument("--cut", type=int, default=2020); ap.add_argument("--minc", type=int, default=8); ap.add_argument("--split", action="store_true")
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--sig", type=float, default=8.0); ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--fold-seed", type=int, default=None, help="another random split of papers into 5 folds (default: the corpus's own)")
    ap.add_argument("--pen", default=""); ap.add_argument("--rank", type=int, default=0); ap.add_argument("--freeze-l", action="store_true"); ap.add_argument("--l-seed", type=int, default=0)
    ap.add_argument("--window-from", default="toks"); ap.add_argument("--opt", default="lbfgs"); ap.add_argument("--outer", type=int, default=15); ap.add_argument("--save-scores", action="store_true"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--years", default="data_all/paper_years.json")
    a = ap.parse_args(); global STEP, SIG; STEP, SIG = a.step, a.sig; FIT["wfrom"] = a.window_from; FIT["opt"] = a.opt; FIT["outer"] = a.outer; H = json.load(open(a.corpus)); C = load_corpus(a.corpus); blk, names = P.blocks_of(H, C, "measure_dir"); B = len(names); FIT["names"] = names
    if a.fold_seed is not None:
        pp = sorted({H["docs"][x]["group"] for x in C["ids"]}); pf = dict(zip(pp, np.random.default_rng(a.fold_seed).integers(0, 5, len(pp)))); C["fold"] = np.array([pf[H["docs"][x]["group"]] for x in C["ids"]])
    Phi = basis(C["xyz"]); nodes = C["nodes"]; D = len(C["ids"]); meas = [H["docs"][x]["measure"] for x in C["ids"]]
    base = lambda i: nodes[i].split(" | ")[0]; is_c = lambda i: nodes[i].startswith("c:all:") and not nodes[i][6:].startswith(("class ", "any condition"))
    dc = [sorted({base(i) for i in C["auth"][d] if is_c(i)}) for d in range(D)]
    print(f"{D} documents, basis {Phi.shape}", flush=True)
    if a.cmd == "coupling":
        cell = collections.defaultdict(lambda: np.zeros(Phi.shape[0])); cn = collections.Counter(); bl = np.zeros((B, Phi.shape[0]))
        # --split: papers are split in two halves; a cell is built separately in each half and block a is taken from half
        # 0, block b from half 1, so a matched pair never shares a paper (one paper's two contrasts often sit in the same place)
        papers = sorted({H["docs"][x]["group"] for x in C["ids"]}); half = dict(zip(papers, np.random.default_rng(1).integers(0, 2, len(papers))))
        for d in range(D):
            f = C["foci"][d]
            if not len(f): continue
            np.add.at(bl[blk[d]], f, 1.0); h = int(half[H["docs"][C["ids"][d]]["group"]]) if a.split else 0
            for c in dc[d]: np.add.at(cell[c, int(blk[d]), h], f, 1.0); cn[c, int(blk[d]), h] += 1
        blp = (bl / bl.sum(1, keepdims=True)) @ Phi; dev = {k: (v / v.sum()) @ Phi - blp[k[1]] for k, v in cell.items() if cn[k] >= a.minc}
        z = {k: (v - v.mean()) / v.std() for k, v in dev.items()}; byb = collections.defaultdict(dict); byb2 = collections.defaultdict(dict)
        for (c, b, h), v in z.items(): (byb if h == 0 else byb2)[b][c] = v
        if not a.split: byb2 = byb
        rng = np.random.default_rng(0); res = []
        for b1 in range(B):
            for b2 in range(B) if a.split else range(b1 + 1, B):
                if b1 == b2 and not a.split: continue
                cs = sorted(set(byb[b1]) & set(byb2[b2]))
                if len(cs) < 8: continue
                Z1 = np.array([byb[b1][c] for c in cs]); Z2 = np.array([byb2[b2][c] for c in cs]); R = Z1 @ Z2.T / Z1.shape[1]; n = len(cs)
                matched = float(np.trace(R) / n); mism = float((R.sum() - np.trace(R)) / (n * n - n))
                perm = np.array([np.mean(R[np.arange(n), rng.permutation(n)]) for _ in range(2000)]); p = float((np.sum(perm >= matched) + 1) / 2001)
                res.append({"a": names[b1], "b": names[b2], "conditions": n, "matched": matched, "mismatched": mism, "coupling": matched - mism, "p": p})
        res.sort(key=lambda r: -r["coupling"]); json.dump({"minc": a.minc, "split_papers": a.split, "cells": len(dev), "pairs": res}, open(a.out, "w"), indent=1)
        print(len(dev), "cells;", len(res), "pairs of blocks with >= 8 shared conditions")
        for r in res[:25] + res[-8:]: print(f"  {r['coupling']:+.3f} (matched {r['matched']:+.3f}, mismatched {r['mismatched']:+.3f}; {r['conditions']} conditions; p {r['p']:.4f})  {r['a']}  x  {r['b']}")
        sig = [r for r in res if r["p"] < 0.05 / len(res)]; print(len(sig), "pairs significant after Bonferroni; same measure, opposite direction:",
              [(r["a"].split(" | ")[0], round(r["coupling"], 3)) for r in res if r["a"].split(" | ")[0] == r["b"].split(" | ")[0] and r["a"] != r["b"]], "; same block, two halves of the papers:", [(r["a"], round(r["coupling"], 3), r["conditions"]) for r in res if r["a"] == r["b"]])
        return
    rows, cols, vals, F = features(C, H, blk, B, a.variant, names)
    X = torch.zeros(D, F); X[torch.tensor(rows), torch.tensor(cols)] = torch.tensor(vals, dtype=torch.float32)
    rk = collections.defaultdict(list)
    for r_, c_, v_ in zip(rows, cols, vals): rk[r_].append((0, c_, round(v_, 6)))
    if FIT["couple"]:
        for r_, c_, v_ in zip(*FIT["couple"][:3]): rk[r_].append((1, c_, round(v_, 6)))
    gk = {}; FIT["gid"] = np.array([gk.setdefault(tuple(sorted(rk[d])), len(gk)) for d in range(D)]); print(len(gk), "distinct label rows", flush=True)
    FIT["pen"] = {k: float(v) for k, v in (x.split("=") for x in a.pen.split(",") if x)}; FIT["rank"] = a.rank; FIT["freeze_l"] = a.freeze_l; FIT["lseed"] = a.l_seed
    assert set(FIT["pen"]) <= {"any", "class", "cond", "dir", "fam", "block", "scope", "factor"}, FIT["pen"]
    if FIT["couple"]:
        cr, cc, cv, nc = FIT["couple"]; FIT["Xc"] = torch.zeros(D, nc); FIT["Xc"][torch.tensor(cr), torch.tensor(cc)] = torch.tensor(cv, dtype=torch.float32); assert a.rank > 0
    pidx = {g: i for i, g in enumerate(sorted({H["docs"][x]["group"] for x in C["ids"]}))}; pap = np.array([pidx[H["docs"][x]["group"]] for x in C["ids"]])
    def by(sc, key):
        acc = collections.defaultdict(lambda: np.zeros(2))
        for d_, m_, w_, n_ in sc: acc[key(d_)] += (m_ - w_, n_)
        return {k: [float(v[0] / v[1]), int(v[1])] for k, v in sorted(acc.items())}
    def keep(sc, test):
        out[test + "_by_direction"] = by(sc, lambda d_: names[blk[d_]].split(" | ")[1]); out[test + "_by_measure"] = by(sc, lambda d_: meas[d_]); out["size"] = FIT["size"]
        if FIT.get("L") is not None: out["L"] = np.round(FIT["L"], 3).tolist()
        if a.save_scores: np.save(a.out.replace(".json", f"_{test}.npy"), np.array([(d_, m_, w_, n_, blk[d_], pap[d_]) for d_, m_, w_, n_ in sc]))
        return "; ".join(f"{k} {v[0]:+.4f}" for k, v in out[test + "_by_direction"].items())
    dev = torch.device(a.device); out = {"variant": a.variant, "lam": a.lam, "features": F, "basis": Phi.shape[1], "args": vars(a)}
    def gain(sc): n = sum(x[3] for x in sc); return (sum(x[1] for x in sc) - sum(x[2] for x in sc)) / n, n
    for test in a.tests.split(","):
        t0 = time.time()
        if test == "paper":
            sc = []
            for k in range(5):
                score = fit_additive(C, blk, B, Phi, X, [d for d in range(D) if C["fold"][d] != k], a.lam, dev, a.steps); sc += score([d for d in range(D) if C["fold"][d] == k])
            g, n = gain(sc); out["paper"] = g; out["paper_window_ll"] = sum(x[2] for x in sc) / n; out["paper_model_ll"] = sum(x[1] for x in sc) / n
            # bootstrap over papers: the interval of the gain per peak, and the mean gain per contrast
            grp = collections.defaultdict(lambda: np.zeros(2))
            for d_, m_, w_, n_ in sc: grp[H["docs"][C["ids"][d_]]["group"]] += (m_ - w_, n_)
            G_ = np.array(list(grp.values())); rb = np.random.default_rng(0); bs = [(lambda i: G_[i, 0].sum() / G_[i, 1].sum())(rb.integers(0, len(G_), len(G_))) for _ in range(2000)]
            out["paper_ci"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]; out["paper_per_contrast"] = float(np.mean([(m_ - w_) / n_ for _, m_, w_, n_ in sc]))
            bd = keep(sc, "paper")
            print(f"{a.variant} lam {a.lam} paper: gain {g:+.4f} ({bd}) [{out['paper_ci'][0]:+.4f}, {out['paper_ci'][1]:+.4f}] over {n} peaks; per contrast {out['paper_per_contrast']:+.4f}; window ll {out['paper_window_ll']:.4f}, model ll {out['paper_model_ll']:.4f} (train ll {FIT['train_ll']:.4f}) {time.time() - t0:.0f}s", flush=True)
        elif test == "cell":
            cm = collections.defaultdict(collections.Counter)
            for d, cs in enumerate(dc):
                for c in cs: cm[c][meas[d]] += 1
            elig = {c: sorted(m for m, k in v.items() if k >= 8) for c, v in cm.items()}; elig = {c: v for c, v in elig.items() if len(v) >= 3}
            rng = np.random.default_rng(a.seed); order = {c: list(rng.permutation(v)) for c, v in sorted(elig.items())}; sc = []; won = []
            for rnd in range(3):
                pick = {c: v[rnd % len(v)] for c, v in order.items()}; te = [d for d in range(D) if any(pick.get(c) == meas[d] for c in dc[d])]; tes = set(te)
                score = fit_additive(C, blk, B, Phi, X, [d for d in range(D) if d not in tes], a.lam, dev, a.steps); s_ = score(te); sc += s_
                acc = collections.defaultdict(lambda: np.zeros(2))
                for d, m_, w_, n_ in s_:
                    for c in dc[d]:
                        if pick.get(c) == meas[d]: acc[c] += (m_, w_)
                won += [v[0] > v[1] for v in acc.values()]
            g, n = gain(sc); out["cell"] = g; out["cell_won"] = float(np.mean(won)); bd = keep(sc, "cell"); print(f"{a.variant} lam {a.lam} cell: gain {g:+.4f} ({bd}) over {n} peaks; cells won {np.mean(won):.2f} {time.time() - t0:.0f}s", flush=True)
        elif test == "future":
            Y = json.load(open(a.years)); yr = np.array([Y[H["docs"][x]["group"]] for x in C["ids"]]); tr = [d for d in range(D) if yr[d] < a.cut]; te = [d for d in range(D) if yr[d] >= a.cut]
            cm = collections.defaultdict(collections.Counter)
            for d in tr:
                if len(C["toks"][d]):
                    for c in dc[d]: cm[c][meas[d]] += 1
            score = fit_additive(C, blk, B, Phi, X, tr, a.lam, dev, a.steps); kinds = collections.defaultdict(lambda: np.zeros(3))
            def kind(c, m):
                tot = sum(cm[c].values()); k = cm[c][m]
                return "condition unseen" if tot == 0 else "condition seen, measure new" if k == 0 and tot >= 8 else "condition barely seen, measure new" if k == 0 else "cell seen 1-4 times" if k < 5 else "cell seen 5+ times"
            sc = score(te); bd = keep(sc, "future")
            for d, m_, w_, n_ in sc:
                for c in dc[d]: kinds[kind(c, meas[d])] += (m_, w_, n_); kinds["all"] += (m_, w_, n_)
            out["future"] = {k: float((v[0] - v[1]) / v[2]) for k, v in kinds.items()}
            print(f"{a.variant} lam {a.lam} future {a.cut} ({bd}): " + "; ".join(f"{k} {v:+.4f}" for k, v in sorted(out["future"].items())) + f" {time.time() - t0:.0f}s", flush=True)
        elif test == "ic":
            FIT["want_ic"] = True; fit_additive(C, blk, B, Phi, X, list(range(D)), a.lam, dev, a.steps); FIT["want_ic"] = False; ic = FIT["ic"]; npap = int(pap.max() + 1)
            ic["BIC_papers"] = -2 * ic["loglik"] + np.log(npap) * ic["edf"]; out["ic"] = ic
            print(f"{a.variant} lam {a.lam} pen {a.pen} rank {a.rank} ic: log-lik {ic['loglik']:.0f} (window {ic['window_loglik']:.0f}); edf {ic['edf']:.0f} +- {ic['edf_se']:.0f} of {ic['parameters']} parameters; "
                  f"AIC {ic['AIC']:.0f} (window {ic['window_AIC']:.0f}; difference {ic['AIC'] - ic['window_AIC']:+.0f}); BIC n=peaks {ic['BIC_peaks'] - ic['window_AIC']:+.0f}, n=papers {ic['BIC_papers'] - ic['window_AIC']:+.0f} relative to the window {time.time() - t0:.0f}s", flush=True)
        elif test == "inner":
            Y = json.load(open(a.years)); yr = np.array([Y[H["docs"][x]["group"]] for x in C["ids"]]); f3 = np.random.default_rng(7).integers(0, 3, pap.max() + 1)[pap]; sc = []
            for k in range(3):
                score = fit_additive(C, blk, B, Phi, X, [d for d in range(D) if yr[d] < a.cut and f3[d] != k], a.lam, dev, a.steps); sc += score([d for d in range(D) if yr[d] < a.cut and f3[d] == k])
            g, n = gain(sc); out["inner"] = g; bd = keep(sc, "inner"); print(f"{a.variant} lam {a.lam} pen {a.pen} rank {a.rank} inner: gain {g:+.5f} ({bd}) over {n} peaks {time.time() - t0:.0f}s", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
