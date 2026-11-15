"""Uncertainty for the additive regression (MODEL_PLAN.md, addition 5), and the test of whether it is honest.

fit: refits the kept regression (checks.py additive, window from peaks counted once, lam 20, family penalty 5, labels split
  by direction, direction by meaning) DRAWS times on a training set, each time on papers resampled with replacement
  (whole papers, so everything a paper reports moves together). Two modes:
    boot  the resampled fit as is: the spread of the penalised estimate
    post  as boot, and the ridge penalty shrinks toward a draw from the prior it stands for, N(0, 1 / (2 penalty)) per
          coefficient, instead of toward zero. A label with no data then varies as much as the penalty says it could; a
          label with much data is unaffected.
  Draw 0 is the ordinary fit. For every draw it stores the share of predicted peaks in each atlas region (Harvard-Oxford,
  and JHU tracts for white matter measures) for every distinct label row, and how often each paper was drawn.
  Training sets: --test paper --part 0..4 (a fifth of the papers held out), --test cell --part 0..2 (a measure held out for
  each eligible condition; the cells of checks.py --tests cell --seed 100), --test all (nothing held out).
  --save-coef also writes OUT_coef.npz: every draw's coefficients and windows, which export_site_hyp.py turns into intervals.
  usage: uncertainty.py fit CORPUS OUT.npz --test paper --part 0 --mode post [--draws 50] [--seed 0]

score: is an interval that should hold 80% of the time right 80% of the time? For every held-out set of contrasts with the
  same labels and block (a unit), and every region: the observed number of peaks in the region against the predictive
  distribution: a draw of the fit, then for each paper of the unit a beta-binomial count with the draw's share as mean and
  within-paper correlation RHO (RHO = 0: binomial). RHO is chosen on the training papers only, from the draws that left a
  paper out (its out-of-bag predictions): the smallest value whose 80% interval covers at least 80% there, for regions
  expecting at least one peak (--rho-by likelihood: by the summed log predictive probability, which the many near-empty regions
  dominate and which under-covers where peaks are expected). Reports the coverage of the central
  50 / 80 / 95% intervals (randomised for discreteness: Czado, Gneiting & Held 2009) by how many training contrasts
  share the unit's labels, by measure, and for regions expected to hold at least one peak; and how wide the interval of
  a cell's expected share is.
  usage: uncertainty.py score CORPUS FIT.npz OUT.json [--rho R]      (a fit with nothing held out: the correlation only)
"""
import argparse, collections, json, os, re, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks as K, pc_model as P
from task_networks.corpus import load_corpus

LAM, VARIANT, PEN = 20.0, "full+dirsplit+meaning", {"fam": 5.0}
WM = ["fa", "diffusivity", "white matter"]
JHU = ["/data2/james/construct-claims/atlases/JHU/JHU-ICBM-labels-2mm.nii.gz", "/data2/james/construct-claims/atlases/JHU/JHU-labels.xml"]
MEAS = ["grey matter", "cortical thickness", "white matter", "fa", "diffusivity", "alff", "falff", "reho", "degree centrality", "glucose metabolism", "perfusion", "molecular binding", "task activation", "eeg meg"]
GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.45]


def setup(corpus):
    H = json.load(open(corpus)); C = load_corpus(corpus); blk, names = P.blocks_of(H, C, "measure_dir"); B = len(names); D = len(C["ids"])
    K.FIT.update(names=names, wfrom="foci", opt="lbfgs", outer=15, pen=PEN)
    Phi = K.basis(C["xyz"]); rows, cols, vals, F = K.features(C, H, blk, B, VARIANT, names)
    X = torch.zeros(D, F); X[torch.tensor(rows), torch.tensor(cols)] = torch.tensor(vals, dtype=torch.float32)
    rk = collections.defaultdict(list)
    for r_, c_, v_ in zip(rows, cols, vals): rk[r_].append((c_, round(v_, 6)))
    gk = {}; gid = np.array([gk.setdefault(tuple(sorted(rk[d])), len(gk)) for d in range(D)]); K.FIT["gid"] = gid
    pidx = {g: i for i, g in enumerate(sorted({H["docs"][x]["group"] for x in C["ids"]}))}; pap = np.array([pidx[H["docs"][x]["group"]] for x in C["ids"]])
    nodes = C["nodes"]; meas = [H["docs"][x]["measure"] for x in C["ids"]]
    base = lambda i: nodes[i].split(" | ")[0]; is_c = lambda i: nodes[i].startswith("c:all:") and not nodes[i][6:].startswith(("class ", "any condition"))
    dc = [sorted({base(i) for i in C["auth"][d] if is_c(i)}) for d in range(D)]
    return dict(H=H, C=C, blk=np.asarray(blk), names=names, B=B, D=D, Phi=Phi, X=X, F=F, gid=gid, pap=pap, meas=meas, dc=dc)


def split(S, test, part):
    D, meas, dc = S["D"], S["meas"], S["dc"]
    if test == "all": te = []
    elif test == "paper": te = [d for d in range(D) if S["C"]["fold"][d] == part]
    else:      # as checks.py --tests cell --seed 100
        cm = collections.defaultdict(collections.Counter)
        for d, cs in enumerate(dc):
            for c in cs: cm[c][meas[d]] += 1
        elig = {c: sorted(m for m, k in v.items() if k >= 8) for c, v in cm.items()}; elig = {c: v for c, v in elig.items() if len(v) >= 3}
        rng = np.random.default_rng(100); order = {c: list(rng.permutation(v)) for c, v in sorted(elig.items())}
        pick = {c: v[part % len(v)] for c, v in order.items()}; te = [d for d in range(D) if any(pick.get(c) == meas[d] for c in dc[d])]
    tes = set(te); return np.array([d for d in range(D) if d not in tes], dtype=int), np.array(te, dtype=int)


def regions(S, cache):
    if os.path.exists(cache): z = np.load(cache, allow_pickle=True); return z["glab"], z["wlab"], list(z["GN"]), list(z["WN"])
    import xml.etree.ElementTree as ET, nibabel as nib
    src = open(Path(__file__).resolve().parent / "export_site.py").read(); a0 = src.index('A = os.path.expanduser("~/nilearn_data'); a1 = src.index("voc = {v[\"condition\"]")
    xyz = S["C"]["xyz"]; ns = {"os": os, "re": re, "np": np, "nib": nib, "ET": ET, "xyz": xyz, "V": len(xyz), "jhu": JHU, "collections": collections}
    exec(src[a0:a1], ns); np.savez(cache, glab=ns["glab"], wlab=ns["wlab"], GN=np.array(ns["GN"]), WN=np.array(ns["WN"]))
    return ns["glab"], ns["wlab"], ns["GN"], ns["WN"]


def fit(a):
    S = setup(a.corpus); dev = torch.device("cuda:0"); tr, te = split(S, a.test, a.part); pap, gid, blk, Phi = S["pap"], S["gid"], S["blk"], S["Phi"]
    glab, wlab, GN, WN = regions(S, a.regions); V = Phi.shape[0]; npap = int(pap.max()) + 1; trp = np.unique(pap[tr])
    oh = lambda lab, n: torch.zeros(V, n, device=dev).scatter_(1, torch.tensor(lab.astype(np.int64), device=dev)[:, None], 1.0)
    Rg, Rw = oh(glab, len(GN)), oh(wlab, len(WN)); G = int(gid.max()) + 1; first = np.unique(gid, return_index=True)[1]
    Xd = S["X"][first].to(dev); bb = torch.tensor(blk[first], device=dev); Phit = torch.tensor(Phi, device=dev)
    penv = np.array([PEN.get(g, LAM) for g in K.FIT["groups"]]); Pg = np.zeros((a.draws + 1, G, len(GN)), np.float32); Pw = np.zeros((a.draws + 1, G, len(WN)), np.float32)
    inbag = np.zeros((a.draws + 1, npap), np.uint8); ll = []; coef, win = [], []
    for b in range(a.draws + 1):
        t0 = time.time(); rng = np.random.default_rng([a.seed, ["paper", "cell", "all"].index(a.test), a.part, b])
        if b == 0: trb = tr; K.FIT["A0"] = None; inbag[0, trp] = 1
        else:
            cnt = np.bincount(rng.choice(trp, len(trp)), minlength=npap); inbag[b] = np.minimum(cnt, 255); trb = np.repeat(tr, cnt[pap[tr]])
            K.FIT["A0"] = (rng.standard_normal((S["F"], Phi.shape[1])) / np.sqrt(2 * penv)[:, None]).astype(np.float32) if a.mode == "post" else None
        K.fit_additive(S["C"], blk, S["B"], Phi, S["X"], list(trb), LAM, dev); ll.append(K.FIT["train_ll"])
        if a.save_coef: coef.append(K.FIT["A"].astype(np.float16)); win.append(K.FIT["logw"].astype(np.float16))
        A = torch.tensor(K.FIT["A"], device=dev); logw = torch.tensor(K.FIT["logw"], dtype=torch.float32, device=dev)
        with torch.no_grad():
            for k in range(0, G, 400):
                p = torch.softmax(logw[bb[k:k + 400]] + Xd[k:k + 400] @ A @ Phit.T, 1); Pg[b, k:k + 400] = (p @ Rg).cpu().numpy(); Pw[b, k:k + 400] = (p @ Rw).cpu().numpy()
        print(f"{a.test} {a.part} {a.mode} draw {b}: train ll {ll[-1]:.4f} {time.time() - t0:.0f}s", flush=True)
    np.savez(a.out, Pg=Pg, Pw=Pw, inbag=inbag, tr=tr, te=te, ll=np.array(ll), test=a.test, part=a.part, mode=a.mode)
    if a.save_coef: np.savez(a.out.replace(".npz", "_coef.npz"), A=np.stack(coef), logw=np.stack(win))      # draws x labels x bumps; draws x blocks x voxels


def betabin_logpmf(y, n, pi, rho):
    from scipy.special import betaln, gammaln
    pi = np.clip(pi, 1e-7, 1 - 1e-7); lc = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
    if rho == 0: return lc + y * np.log(pi) + (n - y) * np.log1p(-pi)
    k = (1 - rho) / rho; return lc + betaln(y + pi * k, n - y + (1 - pi) * k) - betaln(pi * k, (1 - pi) * k)


def score(a):
    S = setup(a.corpus); Z = np.load(a.fit); tr, te, inbag = Z["tr"], Z["te"], Z["inbag"]; glab, wlab, GN, WN = regions(S, a.regions)
    C, gid, pap, meas = S["C"], S["gid"], S["pap"], S["meas"]; nb = inbag.shape[0] - 1; rng = np.random.default_rng(0)
    wm = lambda d: meas[d] in WM
    def counts(d):
        f = np.asarray(C["foci"][d], dtype=int); return np.bincount((wlab if wm(d) else glab)[f], minlength=len(WN if wm(d) else GN)).astype(float)
    PG, PW = Z["Pg"], Z["Pw"]; Pi = lambda d: (PW if wm(d) else PG)[:, gid[d], :]            # draws x regions for contrast d's labels
    # 1. within-paper correlation from out-of-bag predictions of training papers
    units = collections.defaultdict(list)
    for d in tr:
        if len(C["foci"][d]): units[gid[d], pap[d]].append(d)
    keys = list(units); sel = [keys[i] for i in rng.permutation(len(keys))[:a.oob]]; prof = np.zeros(len(GRID)); used = 0; cv = np.zeros(len(GRID)); cn = np.zeros(len(GRID))
    for g, p in sel:
        oob = np.nonzero(inbag[1:, p] == 0)[0] + 1
        if len(oob) < 5: continue
        ds = units[g, p]; y = sum(counts(d) for d in ds); n = y.sum(); pi = Pi(ds[0])[oob]; used += 1
        sm = pi[rng.integers(0, len(oob), 200)]; inf_ = n * pi.mean(0) >= 1
        for i, rho in enumerate(GRID):
            lp = betabin_logpmf(y[None], n, pi, rho); m = lp.max(0); prof[i] += (m + np.log(np.exp(lp - m).mean(0))).sum()
            if inf_.any():       # coverage of the 80% interval for regions expecting at least one peak, on these out-of-bag papers
                q = sm[:, inf_]; k = (1 - rho) / rho if rho else 0; q = np.nan_to_num(rng.beta(np.maximum(q * k, 1e-8), np.maximum((1 - q) * k, 1e-8))) if rho else q
                t = rng.binomial(int(n), np.clip(q, 0, 1)); u = (t < y[inf_]).mean(0) + rng.random(inf_.sum()) * (t == y[inf_]).mean(0); cv[i] += (np.abs(u - 0.5) < 0.4).sum(); cn[i] += inf_.sum()
    cover = cv / np.maximum(cn, 1); print("out-of-bag 80% coverage by rho:", dict(zip(GRID, np.round(cover, 3))), flush=True)
    rho_hat = (next((r for r, c in zip(GRID, cover) if c >= 0.8), GRID[int(np.argmax(cover))]) if a.rho_by == "coverage" else GRID[int(np.argmax(prof))]) if a.rho is None else a.rho
    print("within-paper correlation, log predictive probability per unit relative to rho 0:", {r: round(float(v - prof[0]) / max(used, 1), 3) for r, v in zip(GRID, prof)}, "-> rho", rho_hat, f"({used} out-of-bag units)", flush=True)
    if not len(te):      # nothing held out (--test all): the correlation is all there is to report
        json.dump({"fit": a.fit, "rho": rho_hat, "rho_oob_coverage80": dict(zip(map(str, GRID), cover.tolist())), "draws": nb, "oob_units": used}, open(a.out, "w"), indent=1); return
    # 2. held-out units
    ntr = collections.Counter(gid[d] for d in tr if len(C["foci"][d])); hu = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in te:
        if len(C["foci"][d]): hu[gid[d]][pap[d]].append(d)
    rec = []; SIM = a.sims          # one row per unit x region: pit at rho 0, pit at rho_hat, expected count, stratum, measure
    wid = []
    for g, byp in hu.items():
        d0 = next(iter(byp.values()))[0]; pi = Pi(d0)[1:]; ys = [sum(counts(d) for d in ds) for ds in byp.values()]; y = sum(ys); ns = [int(v.sum()) for v in ys]; n = sum(ns)
        s = pi[rng.integers(0, nb, SIM)]; row = []
        for rho in (0.0, rho_hat):
            tot = np.zeros_like(s)
            for nc in ns:
                if rho > 0:
                    k = (1 - rho) / rho; q = np.nan_to_num(rng.beta(np.maximum(s * k, 1e-8), np.maximum((1 - s) * k, 1e-8)))
                else: q = s
                tot += rng.binomial(nc, np.clip(q, 0, 1))
            row.append((tot < y).mean(0) + rng.random(len(y)) * (tot == y).mean(0))
        k_ = ntr[g]; st = "0" if k_ == 0 else "1-4" if k_ < 5 else "5-19" if k_ < 20 else "20+"; e = n * pi.mean(0)
        for r in range(len(y)): rec.append((row[0][r], row[1][r], e[r], st, meas[d0], n, y[r], len(ns)))
        top = int(np.argmax(pi.mean(0))); lo, hi = np.quantile(pi[:, top], [0.1, 0.9]); wid.append((st, float((PW if wm(d0) else PG)[0, g, top]), float(lo), float(hi)))
    U0 = np.array([r[0] for r in rec]); U1 = np.array([r[1] for r in rec]); E = np.array([r[2] for r in rec]); ST = np.array([r[3] for r in rec]); ME = np.array([r[4] for r in rec])
    def cov(u, m):
        u = u[m]; return {"n": int(m.sum()), "50": round(float((np.abs(u - 0.5) < 0.25).mean()), 3), "80": round(float((np.abs(u - 0.5) < 0.4).mean()), 3), "95": round(float((np.abs(u - 0.5) < 0.475).mean()), 3),
                          "below": round(float((u < 0.1).mean()), 3), "above": round(float((u > 0.9).mean()), 3)} if m.sum() else None
    out = {"fit": a.fit, "rho": rho_hat, "rho_oob_coverage80": dict(zip(map(str, GRID), cover.tolist())), "rho_profile": dict(zip(map(str, GRID), ((prof - prof[0]) / max(used, 1)).tolist())), "units": len(hu), "draws": nb}
    for nm, U in (("cov_rho0", U0), ("cov_rho", U1)):
        o = {"all regions": cov(U, E >= 0), "regions expecting >= 1 peak": cov(U, E >= 1), "regions expecting >= 3 peaks": cov(U, E >= 3)}
        for st in ("0", "1-4", "5-19", "20+"): o[f"training contrasts {st}, expecting >= 1"] = cov(U, (ST == st) & (E >= 1))
        for m in sorted(set(ME)): o[f"{m}, expecting >= 1"] = cov(U, (ME == m) & (E >= 1))
        out[nm] = o
    W = collections.defaultdict(list)
    for st, p0, lo, hi in wid: W[st].append((lo / max(p0, 1e-9), hi / max(p0, 1e-9), hi - lo))
    out["top_region_interval_relative_to_estimate"] = {st: {"units": len(v), "low": round(float(np.median([x[0] for x in v])), 3), "high": round(float(np.median([x[1] for x in v])), 3)} for st, v in W.items()}
    np.save(a.out.replace(".json", "_pit.npy"), np.array([(r[0], r[1], r[2], ("0", "1-4", "5-19", "20+").index(r[3]), r[5], MEAS.index(r[4]), r[6], r[7]) for r in rec], dtype=np.float32))
    json.dump(out, open(a.out, "w"), indent=1)
    for nm in ("cov_rho0", "cov_rho"):
        print(nm, "; ".join(f"{k}: 50% {v['50']}, 80% {v['80']}, 95% {v['95']} (n {v['n']}; below {v['below']}, above {v['above']})" for k, v in out[nm].items() if v and not k.split(",")[0] in set(ME)), flush=True)
    print("top region, 80% interval of the expected share relative to the estimate:", out["top_region_interval_relative_to_estimate"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("corpus"); ap.add_argument("paths", nargs="+")
    ap.add_argument("--test", default="paper"); ap.add_argument("--part", type=int, default=0); ap.add_argument("--mode", default="post"); ap.add_argument("--draws", type=int, default=50); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--save-coef", action="store_true")
    ap.add_argument("--regions", default="data_all/region_labels.npz"); ap.add_argument("--rho", type=float, default=None); ap.add_argument("--rho-by", default="coverage"); ap.add_argument("--oob", type=int, default=3000); ap.add_argument("--sims", type=int, default=400)
    a = ap.parse_args()
    if a.cmd == "fit": a.out = a.paths[0]; fit(a)
    else: a.fit, a.out = a.paths; score(a)
