"""Author-topic model with a map per network and block (multimodal networks), for the patient-versus-control corpus.

The task model (task_networks.model) gives every network one map over voxels. Here a document belongs to a block
(its measure, or its measure and direction: BLOCKS = "measure" | "measure_dir"; "none" is the task model), and a
network k has one map per block, beta[k, b, .], each summing to 1 over voxels:

    P(v | document in block b) = mean_{a in authors} sum_k theta[a, k] beta[k, b, v]

So an author's mixture theta[a] is shared across blocks: a condition that loads on network k is predicted to show
that network's white-matter map in its FA contrasts and its cortical map in its VBM or ReHo contrasts. What a
measure can show at all is learned: s[b, .] is block b's own map (all its tokens, normalized), and

    beta[k, b, .] ∝ counts[k, b, .] + LAM * s[b, .] + ETA

so a network with little data in a block falls back on what that measure shows anywhere, never on voxels the
measure does not report (ReHo in white matter, cortical thickness in the hippocampus). theta has the task model's
hierarchical prior (strength S toward the mean of the parents' pooled mixtures).
The E-step is the task model's (its _Split, over voxel indices offset by block); only the M-step for beta differs.

usage:
  pc_model.py cv   CORPUS K OUT.json [--blocks measure_dir] [--lam 200] [--restarts 2] [--iters 1000] [--device cuda:0]
       5-fold cross-validation by paper (load_corpus's folds): mean log-likelihood per held-out peak.
  pc_model.py fit  CORPUS K OUT_PREFIX [--blocks ...] [--lam ...] [--seed 100] [--restarts 2]
       fit on all documents; writes OUT_PREFIX_{theta,beta,s,n}.npy and OUT_PREFIX.json.
  pc_model.py cellcv CORPUS K OUT.json [--model window --flat-bg] [--rounds 3]
       hold out condition x measure cells (see main); ll per held-out peak under the model, the class, the window.
  --model window: one map per network seen through each block's window; --flat-bg: network 0 flat, background pinned.
  pc_model.py future CORPUS K OUT.json --cut YEAR [--years data_all/paper_years.json]
       train on papers before YEAR, score the contrasts published from YEAR on, by what the training years held.
  pc_model.py sim                                    recovery check on simulated data (CPU, a few seconds).
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from task_networks import model as M
from task_networks.corpus import load_corpus

S, ALPHA, ETA = 100.0, M.ALPHA, M.ETA
# Calibration (2026-11-14: the fitted maps are overconfident on held-out peaks). CS scales the expected counts before the
# M-step: a peak is ~54 tokens (every grid voxel within 10 mm), so CS = 1/54 puts the counts in peaks, which makes S,
# ALPHA and ETA 54 times stronger relative to the data. RHO (windowed model with a flat network): every author's mixture
# is (1 - RHO) of its own plus RHO on the flat network, during the fit. Set from the command line.
CS, RHO = 1.0, 0.0
# WARM: that many first iterations of every fit use the uncalibrated setting (counts in tokens, the task model's S and ETA).
# With counts in peaks the smoothing is strong against a network's data at larger K, and the near-flat random start is
# then a fixed point (every K 30 fit stopped after 6 iterations); the warm-up gives the networks a shape first.
WARM = 0
MIX = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75)     # shares of the model's prediction when blended with the block's own map on held-out peaks
if os.environ.get("PC_THREADS"): torch.set_num_threads(int(os.environ["PC_THREADS"]))   # beast: every job otherwise takes all cores


def order_of(parents):
    n = len(parents); children = [[] for _ in range(n)]
    for c, ps in enumerate(parents):
        for p in ps: children[p].append(c)
    order, seen = [], set()
    def visit(i):
        if i in seen: return
        seen.add(i)
        for c in children[i]: visit(c)
        order.append(i)
    sys.setrecursionlimit(100000)
    for i in range(n): visit(i)
    return order


def theta_from(n, parents, order, s=None):
    """the task model's M-step for theta (model._fit_once)"""
    N = n.copy()
    for i in order:
        for p in parents[i]: N[p] += N[i] / len(parents[i])
    glob = n.sum(0) + ALPHA; glob /= glob.sum()
    bar = N + 1.0; bar /= bar.sum(1, keepdims=True)
    prior = np.array([bar[parents[i]].mean(0) if parents[i] else glob for i in range(len(parents))])
    th = n + (S if s is None else s) * prior + ALPHA
    return th / th.sum(1, keepdims=True)


def theta_mats(parents, order):
    """theta_from as matrices, for the update on the GPU: N = A n (counts pooled up the hierarchy), prior = Pm bar for
    nodes with parents (root: the rows of `root`, which take the global mixture)."""
    n = len(parents); A = np.eye(n); Pm = np.zeros((n, n))
    for i in order:
        for p in parents[i]: A[p] += A[i] / len(parents[i])
    for i, ps in enumerate(parents):
        for p in ps: Pm[i, p] += 1.0 / len(ps)
    return A, Pm, np.array([not ps for ps in parents])


def theta_from_t(n, A, Pm, root, s=None):
    """theta_from on tensors (any device)"""
    glob = n.sum(0) + ALPHA; glob = glob / glob.sum()
    bar = A @ n + 1.0; bar = bar / bar.sum(1, keepdim=True)
    prior = Pm @ bar; prior[root] = glob
    th = n + (S if s is None else s) * prior + ALPHA
    return th / th.sum(1, keepdim=True)


def estep_dev(split, theta, beta):
    """task_networks.model._Split.estep for one device, with the counts left on that device"""
    d = split.devices[0]; th, be = theta.to(d), beta.to(d)
    n = torch.zeros(split.n_nodes, split.K, device=d); b = torch.zeros(split.V, split.K, device=d); ll = torch.zeros((), device=d)
    for _, T, Au, Vx, w, nt, _dw in split.chunks:
        q = th[Au] * be[:, Vx].T * w[:, None]
        tok_p = torch.zeros(nt, device=d).index_add_(0, T, q.sum(1))
        r = q / tok_p[T][:, None]; del q
        n.index_add_(0, Au, r); b.index_add_(0, Vx, r); ll += torch.log(tok_p).sum(); del r, tok_p
    return n, b, float(ll)


def beta_from(b, B, V, lam):
    """block-normalized maps from voxel-network counts b (B*V x K): (K, B*V) tensor and the block maps s (B, V)"""
    c = b.T.reshape(-1, B, V)                                      # K, B, V
    s = c.sum(0) + 1e-9; s = s / s.sum(1, keepdim=True)            # each block's own map
    be = c + lam * s[None] + ETA
    be = be / be.sum(2, keepdim=True)
    return be.reshape(-1, B * V), s


def beta_window(b, B, V, phi, w, inner=5, flat=False):
    """windowed maps: beta[k, b, v] = phi[k, v] w[b, v] / Z[k, b], one map phi per network seen through each block's
    window w (the block's own map, fixed). MM update of phi from voxel-network counts b (B*V x K):
    phi[k, v] <- C[k, v] / sum_b N[k, b] w[b, v] / Z[k, b], with C the counts summed over blocks and N over voxels."""
    c = b.T.reshape(-1, B, V); C = c.sum(1) + ETA; N = c.sum(2)
    for _ in range(inner):
        Z = phi @ w.T
        phi = C / ((N / Z) @ w); phi = phi / phi.sum(1, keepdim=True)
        if flat: phi[0] = 1.0 / V            # network 0 is every block's own map (the window itself), never learned
    be = phi[:, None, :] * w[None]; be = be / be.sum(2, keepdim=True)
    return be.reshape(-1, B * V), phi


def windows(hist, wgroup=None, grid=None, sigma=1.5, floor=0.02):
    """each block's window from the training tokens per block (B x V). wgroup (length B): blocks with the same value
    share one window, pooled over them (a measure's two directions: where a measure can report is a property of the
    measure). grid = (ijk, shape): the pooled map is smoothed within the brain (Gaussian, sigma in voxels), so a voxel
    next to where the measure reports is not impossible. floor: that share of every window is spread evenly over
    voxels. Nothing is borrowed from other measures. With wgroup and grid None: the block's own map (+1e-9)."""
    h = hist.copy()
    if wgroup is not None:
        g = np.asarray(wgroup); h = np.stack([hist[g == g[b]].sum(0) for b in range(len(g))])
    if grid is not None:
        from scipy.ndimage import gaussian_filter
        ijk, shp = grid; m = np.zeros(shp); m[tuple(ijk.T)] = 1.0; den = gaussian_filter(m, sigma)[tuple(ijk.T)]
        for b in range(len(h)):
            v = np.zeros(shp); v[tuple(ijk.T)] = h[b]; h[b] = gaussian_filter(v, sigma)[tuple(ijk.T)] / den
    h = h + 1e-9; h = h / h.sum(1, keepdims=True)
    if wgroup is not None or grid is not None: h = (1 - floor) * h + floor / h.shape[1]
    return h


def fit(docs_tok, doc_auth, doc_block, n_nodes, parents, K, B, V, lam, iters=1000, seed=0, tol=1e-6, restarts=1, device="cpu", verbose=False, model="blocks", flat=None, wgroup=None, grid=None, wsigma=1.5, wfloor=0.02, gpu_update=False):
    """returns theta (n_nodes x K), beta (K x B x V), s (B x V), n (n_nodes x K), log-likelihood, iterations, phi (K x V; None unless model="window").
    flat (window model): index of the background author. Network 0 then has a flat phi (it predicts each block's own map)
    and the background author is pinned to it; other authors may load on it too ("nothing beyond the base rate").
    gpu_update: the counts stay on the device and theta and beta are updated there (same arithmetic as the CPU path)."""
    global S, ETA
    order = order_of(parents); best = None; dev = torch.device(device); t_fit = time.time(); cal = (CS, S, ETA)
    toks = [np.asarray(t, int) + int(bk) * V for t, bk in zip(docs_tok, doc_block)]
    ntok = np.array([len(t) for t in toks]); tok_doc = np.repeat(np.arange(len(toks)), ntok)
    tok_vox = np.concatenate(toks) if len(toks) else np.array([], int)
    cnt = np.array([len(A) for A in doc_auth])
    pair_tok = np.repeat(np.arange(len(tok_doc)), cnt[tok_doc])
    pair_auth = np.concatenate([np.tile(np.asarray(A, int), n) for A, n in zip(doc_auth, ntok)])
    hold = torch.empty(4 * K * B * V + 3 * n_nodes * n_nodes, device=dev) if gpu_update and dev.type == "cuda" else None   # room for the update's arrays
    split = M._Split([dev], pair_tok, pair_auth, tok_vox, (1.0 / cnt[tok_doc])[pair_tok], K, n_nodes, B * V, None, verbose)
    del hold
    if gpu_update:
        if dev.type == "cuda": torch.cuda.empty_cache()
        A, Pm, root = (torch.tensor(x, device=dev) for x in theta_mats(parents, order))
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed + r)
        theta = torch.rand(n_nodes, K, generator=g) + 0.5; theta /= theta.sum(1, keepdim=True)
        beta = torch.rand(K, B, V, generator=g) + 0.5; beta = (beta / beta.sum(2, keepdim=True)).reshape(K, B * V)
        phi = None
        if model == "window":
            w = torch.tensor(windows(np.bincount(tok_vox, minlength=B * V).reshape(B, V).astype(float), wgroup, grid, wsigma, wfloor))
            phi = torch.rand(K, V, generator=g).double() + 0.5; phi = phi / phi.sum(1, keepdim=True)
            if flat is not None: phi[0] = 1.0 / V; theta[flat] = 0.0; theta[flat, 0] = 1.0
            beta = phi[:, None, :] * w[None]; beta = (beta / beta.sum(2, keepdim=True)).reshape(K, B * V).float()
            if gpu_update: phi, w = phi.to(dev), w.to(dev)
        prev = -np.inf; t_it = time.time()
        for it in range(iters):
            cs, S, ETA = (1.0, 100.0, M.ETA) if it < WARM else cal
            if gpu_update:
                n, bsum, ll = estep_dev(split, theta, beta); theta = theta_from_t(n * cs, A, Pm, root)
            else:
                n, bsum, ll = split.estep(theta, beta); theta = torch.tensor(theta_from(n * cs, parents, order))
            if flat is not None:
                if RHO: theta = (1 - RHO) * theta; theta[:, 0] += RHO
                theta[flat] = 0.0; theta[flat, 0] = 1.0
            if cs != 1.0: bsum = bsum * cs
            if model == "window":
                beta, phi = beta_window(bsum.double() if hasattr(bsum, "double") else torch.tensor(bsum), B, V, phi, w, flat=flat is not None); s = w; beta = beta.float()
            else:
                beta, s = beta_from(bsum, B, V, lam)
            if it >= WARM and abs(ll - prev) < tol * abs(ll): break
            prev = ll
        _, S, ETA = cal
        if verbose: print(f"  restart {r}: ll {ll:.1f}, {it + 1} iterations, {(time.time() - t_it) / (it + 1):.3f}s per iteration, setup {t_it - t_fit:.0f}s", flush=True)
        if best is None or ll > best[4]:
            c = lambda x: x.cpu().numpy() if torch.is_tensor(x) else x
            best = (c(theta), c(beta.reshape(K, B, V)), c(s), c(n), ll, it + 1, None if phi is None else c(phi))
    return best


def blocks_of(H, C, mode):
    if mode == "none":
        return np.zeros(len(C["ids"]), int), ["all"]
    key = (lambda d: d["measure"]) if mode == "measure" else (lambda d: f"{d['measure']} | {d['direction']}")
    names = sorted({key(d) for d in H["docs"].values()}); ix = {k: i for i, k in enumerate(names)}
    return np.array([ix[key(H["docs"][a])] for a in C["ids"]]), names


def heldout(C, blk, theta, beta, test, s=None):
    """sum over held-out documents of the log-likelihood of their peaks, and the number of peaks"""
    ll = 0.0; npk = 0; mix = np.zeros(len(MIX) + 1)     # with s (the blocks' own maps): the model blended with it at each MIX share, then s alone
    for d in test:
        f = C["foci"][d]
        if len(f) == 0 or len(C["auth"][d]) == 0: continue
        p = (theta[C["auth"][d]] @ beta[:, blk[d], :][:, f]).mean(0)
        ll += float(np.log(p).sum()); npk += len(f)
        if s is not None:
            for j, lam in enumerate(MIX): mix[j] += float(np.log(lam * p + (1 - lam) * s[blk[d], f]).sum())
            mix[-1] += float(np.log(s[blk[d], f]).sum())
    return (ll, npk) if s is None else (ll, npk, mix)


def main():
    global S, ALPHA, ETA, CS, RHO, WARM
    if sys.argv[1] == "sim":
        return sim()
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("corpus"); ap.add_argument("K", type=int); ap.add_argument("out")
    ap.add_argument("--blocks", default="measure_dir"); ap.add_argument("--lam", type=float, default=200.0); ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--iters", type=int, default=1000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--model", default="blocks"); ap.add_argument("--flat-bg", action="store_true"); ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--window", default="block", help="block: each block's own map; measure: pooled over the measure's directions, smoothed, with a floor")
    ap.add_argument("--gpu-update", action="store_true", help="update theta and beta on the device")
    ap.add_argument("--tol", type=float, default=1e-6); ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--cut", type=int, default=2020); ap.add_argument("--years", default="data_all/paper_years.json")
    ap.add_argument("--warm", type=int, default=0); ap.add_argument("--count-scale", type=float, default=1.0); ap.add_argument("--flat-share", type=float, default=0.0)
    ap.add_argument("--eta", type=float, default=ETA); ap.add_argument("--alpha", type=float, default=ALPHA); ap.add_argument("--S", type=float, default=S)
    ap.add_argument("--wsigma", type=float, default=1.5); ap.add_argument("--wfloor", type=float, default=0.02)
    a = ap.parse_args()
    S, ALPHA, ETA, CS, RHO, WARM = a.S, a.alpha, a.eta, a.count_scale, a.flat_share, a.warm
    SET = {"warm": WARM, "S": S, "alpha": ALPHA, "eta": ETA, "count_scale": CS, "flat_share": RHO, "window": a.window, "model": a.model, "flat_background": a.flat_bg, "seed": a.seed}
    H = json.load(open(a.corpus)); C = load_corpus(a.corpus); V = len(C["xyz"]); blk, names = blocks_of(H, C, a.blocks); B = len(names)
    lam = 0.0 if a.blocks == "none" else a.lam; flat = C["nodes"].index("background") if a.flat_bg else None
    W = dict(gpu_update=a.gpu_update)
    if a.window == "measure":
        ms = [x.split(" | ")[0] for x in names]; ijk = np.rint((C["xyz"] - C["xyz"].min(0)) / 4).astype(int)
        W = dict(gpu_update=a.gpu_update, wgroup=[ms.index(x) for x in ms], grid=(ijk, tuple(ijk.max(0) + 1)), wsigma=a.wsigma, wfloor=a.wfloor)
    print(f"{len(C['ids'])} documents, {len(C['nodes'])} nodes, K {a.K}, blocks {a.blocks} ({B}), lambda {lam}", flush=True)
    if a.cmd == "cv":
        tot = 0.0; npk = 0; folds = []
        for k in map(int, a.folds.split(",")):
            tr = [d for d in range(len(C["ids"])) if C["fold"][d] != k and len(C["toks"][d])]; te = [d for d in range(len(C["ids"])) if C["fold"][d] == k]
            t0 = time.time()
            th, be, s, n, ll, it, phi = fit([C["toks"][d] for d in tr], [C["auth"][d] for d in tr], blk[tr], len(C["nodes"]), C["parents"], a.K, B, V, lam,
                                       a.iters, a.seed, tol=a.tol, restarts=a.restarts, device=a.device, model=a.model, flat=flat, **W)
            l, p, mx = heldout(C, blk, th, be, te, s); tot += l; npk += p; mixtot = mx if k == 0 else mixtot + mx
            folds.append({"fold": k, "ll_per_peak": l / p, "peaks": p, "iterations": it, "own_map": mx[-1] / p, "blend": dict(zip(map(str, MIX), (mx[:-1] / p).tolist()))})
            print(f"K{a.K} fold {k}: {it} iterations, {time.time() - t0:.0f}s, ll/peak {l / p:.5f}; block's own map alone {mx[-1] / p:.5f}; best blend {MIX[int(np.argmax(mx[:-1]))]} -> {mx[:-1].max() / p:.5f}", flush=True)
        # the same peaks under each block's own training-free reference: uniform over voxels
        res = {"K": a.K, "settings": SET, "blocks": a.blocks, "lambda": lam, "ll_per_peak": tot / npk, "peaks": npk, "uniform": float(-np.log(V)), "own_map": float(mixtot[-1] / npk),
               "blend": dict(zip(map(str, MIX), (mixtot[:-1] / npk).tolist())), "folds": folds}
        json.dump(res, open(a.out, "w"), indent=1); print(f"K{a.K} {a.blocks}: {npk} held-out peaks; ll/peak {tot / npk:.5f}; own map {mixtot[-1] / npk:.5f}; blends {res['blend']}", flush=True)
    elif a.cmd == "future":
        # Train on papers published before --cut, score the contrasts of papers from --cut on. Each held-out contrast
        # is filed under (condition, measure) for each of its conditions, and the cells are grouped by what the training
        # years held: the condition unseen; the condition seen (>= 8 contrasts) but never in this measure; this cell
        # seen 1 to 4 times; 5 or more. Scores as in cellcv: the model, the condition's class in its place, the window.
        import collections
        Y = json.load(open(a.years)); yr = np.array([Y[H["docs"][x]["group"]] for x in C["ids"]])
        base = lambda i: C["nodes"][i].split(" | ")[0]; is_c = lambda i: C["nodes"][i].startswith("c:all:") and not C["nodes"][i][6:].startswith(("class ", "any condition"))
        meas = [H["docs"][x]["measure"] for x in C["ids"]]; dc = [sorted({base(i) for i in C["auth"][d] if is_c(i)}) for d in range(len(C["ids"]))]
        nix = {x: i for i, x in enumerate(C["nodes"])}
        def klass(i):
            j = nix[base(i)]
            while not C["nodes"][j].startswith("c:all:class ") and C["parents"][j]: j = C["parents"][j][0]
            return j
        tr = [d for d in range(len(C["ids"])) if yr[d] < a.cut and len(C["toks"][d])]; te = [d for d in range(len(C["ids"])) if yr[d] >= a.cut]
        cm = collections.defaultdict(collections.Counter)
        for d in tr:
            for c in dc[d]: cm[c][meas[d]] += 1
        t0 = time.time()
        th, be, s, n, ll, it, phi = fit([C["toks"][d] for d in tr], [C["auth"][d] for d in tr], blk[tr], len(C["nodes"]), C["parents"], a.K, B, V, lam,
                                        a.iters, a.seed, restarts=a.restarts, device=a.device, model=a.model, flat=flat, **W)
        acc = collections.defaultdict(lambda: np.zeros(5 + len(MIX)))
        for d in te:
            f = C["foci"][d]
            if len(f) == 0: continue
            bb = be[:, blk[d], :][:, f]; A1 = C["auth"][d]; A2 = [klass(i) if is_c(i) else i for i in A1]; pm = (th[A1] @ bb).mean(0); pw = s[blk[d], f]
            v = np.array([np.log(pm).sum(), np.log((th[A2] @ bb).mean(0)).sum(), np.log(pw).sum(), len(f), 1.0] + [np.log(lam_ * pm + (1 - lam_) * pw).sum() for lam_ in MIX])
            for c in dc[d]: acc[c, meas[d]] += v
        def kind(c, m):
            tot = sum(cm[c].values()); k = cm[c][m]
            return "condition unseen" if tot == 0 else "condition seen, measure new" if k == 0 and tot >= 8 else "condition barely seen, measure new" if k == 0 else "cell seen 1-4 times" if k < 5 else "cell seen 5+ times"
        cells = [{"condition": c[6:], "measure": m, "kind": kind(c, m), "train_cell": cm[c][m], "train_condition": sum(cm[c].values()), "contrasts": int(v[4]), "peaks": int(v[3]),
                  "model": v[0] / v[3], "class": v[1] / v[3], "window": v[2] / v[3], "blend": dict(zip(map(str, MIX), (v[5:] / v[3]).tolist()))} for (c, m), v in acc.items() if v[3]]
        json.dump({"K": a.K, "cut": a.cut, "settings": SET, "train_contrasts": len(tr), "test_contrasts": len(te), "iterations": it, "cells": cells}, open(a.out, "w"), indent=1)
        print(f"cut {a.cut}: {len(tr)} training contrasts, {len(te)} held out; {it} iterations, {time.time() - t0:.0f}s", flush=True)
        for kd in ("condition unseen", "condition barely seen, measure new", "condition seen, measure new", "cell seen 1-4 times", "cell seen 5+ times", None):
            cs = [c for c in cells if kd is None or c["kind"] == kd]
            if not cs: continue
            P = sum(c["peaks"] for c in cs); g = lambda fn: sum(fn(c) * c["peaks"] for c in cs) / P
            bl = max(((l, g(lambda c: c["blend"][l])) for l in map(str, MIX)), key=lambda kv: kv[1])
            print(f"  {kd or 'all':36s} {len(cs):5d} cells {P:6d} peaks: model - window {g(lambda c: c['model'] - c['window']):+.4f}; class - window {g(lambda c: c['class'] - c['window']):+.4f}; "
                  f"best blend {bl[0]} {bl[1] - g(lambda c: c['window']):+.4f}; cells where the model wins {np.mean([c['model'] > c['window'] for c in cs]):.2f}", flush=True)
    elif a.cmd == "cellcv":
        # Hold out whole condition x measure cells: for each condition with >= 3 measures of >= MINC contrasts, every
        # contrast of one of those measures (a different one per round). Fit on the rest; score the held-out peaks under
        # the model (the documents' own authors), under the condition's class in place of the condition, and under the
        # measure's own map alone (the window). Tests whether a condition's other measures predict the held-out one.
        import collections
        MINC = 8; base = lambda i: C["nodes"][i].split(" | ")[0]; is_c = lambda i: C["nodes"][i].startswith("c:all:") and not C["nodes"][i][6:].startswith(("class ", "any condition"))
        meas = [H["docs"][x]["measure"] for x in C["ids"]]; dc = [sorted({base(i) for i in C["auth"][d] if is_c(i)}) for d in range(len(C["ids"]))]
        cm = collections.defaultdict(collections.Counter)
        for d, cs in enumerate(dc):
            for c in cs: cm[c][meas[d]] += 1
        elig = {c: sorted(m for m, k in v.items() if k >= MINC) for c, v in cm.items()}; elig = {c: v for c, v in elig.items() if len(v) >= 3}
        nix = {x: i for i, x in enumerate(C["nodes"])}
        def klass(i):
            j = nix[base(i)]
            while not C["nodes"][j].startswith("c:all:class ") and C["parents"][j]: j = C["parents"][j][0]
            return j
        rng = np.random.default_rng(a.seed); order = {c: list(rng.permutation(v)) for c, v in sorted(elig.items())}; cells = []
        for rnd in range(a.rounds):
            pick = {c: v[rnd % len(v)] for c, v in order.items()}
            te = [d for d in range(len(C["ids"])) if any(pick.get(c) == meas[d] for c in dc[d])]; tes = set(te)
            tr = [d for d in range(len(C["ids"])) if d not in tes and len(C["toks"][d])]; t0 = time.time()
            th, be, s, n, ll, it, phi = fit([C["toks"][d] for d in tr], [C["auth"][d] for d in tr], blk[tr], len(C["nodes"]), C["parents"], a.K, B, V, lam,
                                            a.iters, a.seed, restarts=a.restarts, device=a.device, model=a.model, flat=flat, **W)
            acc = collections.defaultdict(lambda: np.zeros(5 + len(MIX)))
            for d in te:
                f = C["foci"][d]
                if len(f) == 0: continue
                bb = be[:, blk[d], :][:, f]; A1 = C["auth"][d]; A2 = [klass(i) if is_c(i) else i for i in A1]
                pm = (th[A1] @ bb).mean(0); pw = s[blk[d], f]
                v = np.array([np.log(pm).sum(), np.log((th[A2] @ bb).mean(0)).sum(), np.log(pw).sum(), len(f), 1.0] + [np.log(lam * pm + (1 - lam) * pw).sum() for lam in MIX])
                for c in dc[d]:
                    if pick.get(c) == meas[d]: acc[c] += v
            for c, v in acc.items():
                if v[3]: cells.append({"round": rnd, "condition": c[6:], "measure": pick[c], "contrasts": int(v[4]), "peaks": int(v[3]), "other_contrasts": int(sum(cm[c].values()) - cm[c][pick[c]]),
                                       "model": v[0] / v[3], "class": v[1] / v[3], "window": v[2] / v[3], "blend": dict(zip(map(str, MIX), (v[5:] / v[3]).tolist()))})
            print(f"round {rnd}: {len(te)} held-out contrasts in {len(acc)} cells, {it} iterations, {time.time() - t0:.0f}s", flush=True)
            json.dump({"K": a.K, "settings": SET, "model": a.model, "flat_background": a.flat_bg, "min_contrasts": MINC, "cells": cells}, open(a.out, "w"), indent=1)
        P = np.array([c["peaks"] for c in cells]); g = lambda k: float(sum(c[k] * c["peaks"] for c in cells) / P.sum())
        print(f"{len(cells)} cells, {P.sum()} peaks; ll/peak model {g('model'):.4f}, class {g('class'):.4f}, window {g('window'):.4f}; "
              f"cells where model beats window {np.mean([c['model'] > c['window'] for c in cells]):.2f}, beats class {np.mean([c['model'] > c['class'] for c in cells]):.2f}", flush=True)
        for lam in map(str, MIX):
            print(f"  blend {lam}: ll/peak {float(sum(c['blend'][lam] * c['peaks'] for c in cells) / P.sum()):.4f}; cells beating the window {np.mean([c['blend'][lam] > c['window'] for c in cells]):.2f}", flush=True)
    else:
        ok = [d for d in range(len(C["ids"])) if len(C["toks"][d])]
        th, be, s, n, ll, it, phi = fit([C["toks"][d] for d in ok], [C["auth"][d] for d in ok], blk[ok], len(C["nodes"]), C["parents"], a.K, B, V, lam,
                                   a.iters, a.seed, restarts=a.restarts, device=a.device, verbose=True, model=a.model, flat=flat, **W)
        if phi is not None: np.save(f"{a.out}_phi.npy", phi.astype(np.float32))
        for nm, x in (("theta", th), ("beta", be), ("s", s), ("n", n)): np.save(f"{a.out}_{nm}.npy", x.astype(np.float32) if nm != "n" else x)
        np.save(f"{a.out}_xyz.npy", C["xyz"])
        json.dump({"K": a.K, "blocks": names, "block_mode": a.blocks, "model": a.model, "flat_background": a.flat_bg, "gpu_update": a.gpu_update, "window": a.window, "wsigma": a.wsigma, "wfloor": a.wfloor, "lambda": lam, "seed": a.seed, "restarts": a.restarts, "iterations": it, "ll_tokens": ll, "nodes": C["nodes"]}, open(a.out + ".json", "w"))
        print(json.dumps({"K": a.K, "blocks": B, "iterations": it, "ll_tokens": ll}), flush=True)


def sim():
    """known networks with different maps per block; authors share mixtures across blocks; recover the maps"""
    rng = np.random.default_rng(0); K, B, V, A, D = 4, 3, 240, 12, 1500
    beta = np.full((K, B, V), 1e-4)
    for k in range(K):
        for b in range(B):
            c = rng.integers(0, V - 20); beta[k, b, c:c + 20] += 1.0
    support = np.ones((B, V)); support[1, : V // 2] = 0; support[2, V // 2:] = 0     # block 1 cannot show the first half, block 2 the second
    beta *= support[None]; beta /= beta.sum(2, keepdims=True)
    theta = rng.dirichlet(np.full(K, 0.3), A); toks, auth, blk = [], [], []
    for d in range(D):
        a = rng.choice(A, rng.integers(1, 3), replace=False); b = rng.integers(B); t = []
        for _ in range(60):
            k = rng.choice(K, p=theta[rng.choice(a)]); t.append(rng.choice(V, p=beta[k, b]))
        toks.append(np.array(t)); auth.append(a); blk.append(b)
    th, be, s, n, ll, it, phi = fit(toks, auth, np.array(blk), A, [[] for _ in range(A)], K, B, V, 20.0, 500, 0, restarts=3)
    r = np.corrcoef(np.r_[be.reshape(K, -1), beta.reshape(K, -1)])[:K, K:]; match = r.max(1)
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(-r); rt = np.corrcoef(th[:, ri].ravel(), theta[:, ci].ravel())[0, 1]
    leak = float(max(be[:, 1, : V // 2].sum(1).max(), be[:, 2, V // 2:].sum(1).max()))
    plain = fit(toks, auth, np.zeros(D, int), A, [[] for _ in range(A)], K, 1, V, 0.0, 500, 0, restarts=3)
    print(f"sim: {it} iterations; matched network map r {np.round(r[ri, ci], 3).tolist()}; author mixture r {rt:.3f}; "
          f"mass a network puts where its block has no data (should be near 0): {leak:.4f}; train ll blocks {ll:.0f} vs one map per network {plain[4]:.0f}")
    assert r[ri, ci].min() > 0.9 and rt > 0.9 and leak < 0.02, "recovery failed"
    print("SIM OK")
    # windowed model: one map per network (two bumps: one in each half), blocks 1 and 2 each see one half only
    phi0 = np.full((K, V), 1e-4)
    for k in range(K):
        phi0[k, 10 + 25 * k:25 + 25 * k] += 1.0; phi0[k, V // 2 + 100 - 25 * k:V // 2 + 115 - 25 * k] += 1.0
    w0 = rng.random((B, V)) + 0.5; w0[1, : V // 2] = 1e-6; w0[2, V // 2:] = 1e-6; w0 /= w0.sum(1, keepdims=True)
    bw = phi0[:, None, :] * w0[None]; bw /= bw.sum(2, keepdims=True); toks, auth, blk = [], [], []
    for d in range(D):
        a = rng.choice(A, rng.integers(1, 3), replace=False); b = rng.integers(B); t = []
        for _ in range(60):
            k = rng.choice(K, p=theta[rng.choice(a)]); t.append(rng.choice(V, p=bw[k, b]))
        toks.append(np.array(t)); auth.append(a); blk.append(b)
    th, be, s, n, ll, it, ph = fit(toks, auth, np.array(blk), A, [[] for _ in range(A)], K, B, V, 0.0, 500, 0, restarts=3, model="window")
    # phi is identified only relative to the windows (the fit's window is the block's own map, which already holds the
    # average network), so recovery is judged on the maps each block sees
    r = np.corrcoef(np.r_[be.reshape(K, -1), bw.reshape(K, -1)])[:K, K:]; ri, ci = linear_sum_assignment(-r)
    rt = np.corrcoef(th[:, ri].ravel(), theta[:, ci].ravel())[0, 1]; leak = float(max(be[:, 1, : V // 2].sum(1).max(), be[:, 2, V // 2:].sum(1).max()))
    print(f"sim window: {it} iterations; network map r {np.round(r[ri, ci], 3).tolist()}; author mixture r {rt:.3f}; leak {leak:.4f}")
    assert r[ri, ci].min() > 0.9 and rt > 0.9 and leak < 0.02, "window recovery failed"
    print("SIM WINDOW OK")
    # flat background: an extra author on every document draws from the block's own window; K + 1 networks, network 0 flat
    toks, auth, blk = [], [], []
    for d in range(D):
        a = list(rng.choice(A, rng.integers(1, 3), replace=False)) + [A]; b = rng.integers(B); t = []
        for _ in range(60):
            x = rng.choice(a); t.append(rng.choice(V, p=w0[b]) if x == A else rng.choice(V, p=bw[rng.choice(K, p=theta[x]), b]))
        toks.append(np.array(t)); auth.append(np.array(a)); blk.append(b)
    th, be, s, n, ll, it, ph = fit(toks, auth, np.array(blk), A + 1, [[] for _ in range(A + 1)], K + 1, B, V, 0.0, 500, 0, restarts=3, model="window", flat=A)
    r = np.corrcoef(np.r_[be[1:].reshape(K, -1), bw.reshape(K, -1)])[:K, K:]; ri, ci = linear_sum_assignment(-r)
    t1 = th[:A, 1:] / th[:A, 1:].sum(1, keepdims=True); rt = np.corrcoef(t1[:, ri].ravel(), theta[:, ci].ravel())[0, 1]
    print(f"sim flat background: {it} iterations; network map r {np.round(r[ri, ci], 3).tolist()}; author mixture r {rt:.3f}; "
          f"mean loading of the other authors on the flat network {th[:A, 0].mean():.3f}; background on it {th[A, 0]:.2f}")
    # the fit's window is the block's observed map, which holds the average network as well as the true window, so the
    # other authors put some loading on the flat network; it must stay a minority
    assert r[ri, ci].min() > 0.9 and rt > 0.9 and th[:A, 0].mean() < 0.25, "flat background recovery failed"
    print("SIM FLAT OK")


if __name__ == "__main__":
    main()
