"""Pools uncertainty.py score outputs over folds. usage: unc_report.py FITS_DIR OUT.json
Coverage of the central 50 / 80 / 95% predictive intervals for region peak counts, per test (paper, cell) and mode (boot,
post), without (rho 0) and with the within-paper correlation, by stratum; standard errors treat region-units as independent."""
import glob, json, sys, numpy as np
from uncertainty import MEAS
d, outp = sys.argv[1:3]; MODES = sys.argv[3].split(",") if len(sys.argv) > 3 else ("boot", "post"); out = {}
def cov(u):
    if not len(u): return None
    c = lambda h: round(float((np.abs(u - 0.5) < h).mean()), 3)
    return {"n": int(len(u)), "50": c(0.25), "80": c(0.4), "95": c(0.475), "below": round(float((u < 0.1).mean()), 3), "above": round(float((u > 0.9).mean()), 3), "se80": round(float(np.sqrt(0.16 / len(u))), 3)}
for test in ("paper", "cell"):
    for mode in MODES:
        fs = sorted(glob.glob(f"{d}/{test}?_{mode}_score_pit.npy"))
        if not fs: continue
        R = np.concatenate([np.load(f) for f in fs]); rho = [json.load(open(f.replace("_pit.npy", ".json")))["rho"] for f in fs]
        wid = [json.load(open(f.replace("_pit.npy", ".json")))["top_region_interval_relative_to_estimate"] for f in fs]
        o = {"parts": len(fs), "rho": rho, "top_region_interval": wid[0]}
        for nm, col in (("binomial", 0), ("with paper correlation", 1)):
            u, E, st, n, me = R[:, col], R[:, 2], R[:, 3], R[:, 4], R[:, 5]; y, npp = R[:, 6], R[:, 7]; o["observed over expected peaks"] = {k: round(float(y[m].sum() / E[m].sum()), 3) for k, m in (("all", E >= 0), ("expecting < 0.2", E < 0.2), ("0.2-1", (E >= 0.2) & (E < 1)), ("1-3", (E >= 1) & (E < 3)), (">= 3", E >= 3))}
            t = {"all regions": cov(u), "expecting >= 1": cov(u[E >= 1]), "expecting >= 3": cov(u[E >= 3]), "expecting < 0.2": cov(u[E < 0.2])}
            for i, s in enumerate(("0", "1-4", "5-19", "20+")): t[f"training contrasts {s}, expecting >= 1"] = cov(u[(st == i) & (E >= 1)])
            for lo, hi in ((0, 10), (10, 30), (30, 100), (100, 1e9)): t[f"unit peaks {lo}-{hi}, expecting >= 1"] = cov(u[(n >= lo) & (n < hi) & (E >= 1)])
            for k, m in (("one paper", npp == 1), ("2-3 papers", (npp > 1) & (npp < 4)), ("4+ papers", npp >= 4)): t[f"unit of {k}, expecting >= 1"] = cov(u[m & (E >= 1)])
            for i, m in enumerate(MEAS): t[f"{m}, expecting >= 1"] = cov(u[(me == i) & (E >= 1)])
            o[nm] = t
        out[f"{test} {mode}"] = o
        print(f"\n== {test} {mode}: rho {rho}; top region interval {wid[0]}; observed/expected {o.get('observed over expected peaks')}")
        for nm in ("binomial", "with paper correlation"):
            print(" ", nm)
            for k, v in o[nm].items():
                if v: print(f"    {k:48s} n {v['n']:7d}  50% {v['50']:.3f}  80% {v['80']:.3f} (+-{v['se80']:.3f})  95% {v['95']:.3f}  below {v['below']:.3f} above {v['above']:.3f}")
json.dump(out, open(outp, "w"), indent=1)
