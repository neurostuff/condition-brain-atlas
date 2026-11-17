"""Peak statistics of every published cell in the real and shuffled-diagnosis refits (null_atlas.py with SAVE_COEF=1).

The planted-signal simulation (plant_atlas.py) showed that the sensitive statistic is the log ratio of a cell's predicted
peak density against its comparison, not the share of peaks that moves. Per fit and cell (condition x measure x reported
direction, >= 1 contrast), against the measure's usual pattern (window + block) and against the class (window + block +
class labels):
  max_u, max_c  the largest log ratio over grid points (whole-brain search);
  ru, rc        per atlas region, log(cell's share of predicted peaks in the region / comparison's share) (known location;
                Harvard-Oxford for grey matter measures, JHU for white matter ones, labelled as export_site_hyp.py does).
usage: null_peak.py CORPUS JHU.nii.gz JHU.xml FIT_DIR OUT.npz   (reads FIT_DIR/<tag>_coef.npz and <tag>_cells.json)"""
import collections, glob, json, os, re, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks as K
from task_networks.corpus import load_corpus

corpus, jhu_img, jhu_xml, fd, outp = sys.argv[1:6]; WM = ["fa", "diffusivity", "white matter"]
C = load_corpus(corpus); xyz = C["xyz"]; V = len(xyz); Phi = K.basis(xyz)
import xml.etree.ElementTree as ET, nibabel as nib                 # region labels per grid point, the exporter's code
src = open(Path(__file__).resolve().parent / "export_site.py").read(); a0 = src.index('A = os.path.expanduser("~/nilearn_data'); a1 = src.index("voc = {v[\"condition\"]")
ns = {"os": os, "re": re, "np": np, "nib": nib, "ET": ET, "xyz": xyz, "V": V, "jhu": [jhu_img, jhu_xml], "collections": collections}
exec(src[a0:a1], ns); GN, glab, WN, wlab = ns["GN"], ns["glab"], ns["WN"], ns["wlab"]
dev = torch.device("cuda:0"); RM = max(len(GN), len(WN))
oh = lambda lab: torch.zeros(V, RM, device=dev).scatter_(1, torch.tensor(lab.astype(np.int64), device=dev)[:, None], 1.0)
Rg, Rw = oh(glab), oh(wlab); Phit = torch.tensor(Phi, dtype=torch.float32, device=dev)
out = {"GN": np.array(GN), "WN": np.array(WN)}
for f in sorted(glob.glob(f"{fd}/*_coef.npz")):
    tag = os.path.basename(f)[:-9]; Z = np.load(f); cells = json.load(open(f"{fd}/{tag}_cells.json")); nn = int(Z["n_nodes"])
    At = torch.tensor(Z["A"].astype(np.float32), device=dev); Lw = torch.tensor(Z["logw"], device=dev); NC = len(cells)
    def sp(lists):
        r = np.repeat(np.arange(len(lists)), [len(x) for x in lists]); k = np.concatenate([np.array(x, np.int64) for x in lists])
        return torch.sparse_coo_tensor(np.stack([r, k]), np.ones(len(r), np.float32), (len(lists), At.shape[0])).coalesce().to(dev)
    bb = torch.tensor([x[4] for x in cells], device=dev); wm = torch.tensor([x[1] in WM for x in cells], device=dev)[:, None]
    Mf, Mc, Mb = sp([[nn + x[4]] + x[5] for x in cells]), sp([[nn + x[4]] + x[6] for x in cells]), sp([[nn + x[4]] for x in cells])
    res = {k: [] for k in ("max_u", "max_c", "ru", "rc")}
    with torch.no_grad():
        cf, cc, cb = torch.sparse.mm(Mf, At), torch.sparse.mm(Mc, At), torch.sparse.mm(Mb, At)
        for s in range(0, NC, 400):
            e = slice(s, s + 400); lw = Lw[bb[e]]
            lp, lc, lb = (torch.log_softmax(lw + x[e] @ Phit.T, 1) for x in (cf, cc, cb))
            res["max_u"].append((lp - lb).max(1).values); res["max_c"].append((lp - lc).max(1).values)
            sh = lambda l: torch.where(wm[e], l.exp() @ Rw, l.exp() @ Rg).clamp_min(1e-12).log()
            p_, c_, b_ = sh(lp), sh(lc), sh(lb); res["ru"].append(p_ - b_); res["rc"].append(p_ - c_)
    for k, v in res.items(): out[f"{tag}/{k}"] = torch.cat(v).cpu().numpy().astype(np.float32 if k.startswith("max") else np.float16)
    out[f"{tag}/key"] = np.array([f"{x[0]}|{x[1]}|{x[2]}" for x in cells]); out[f"{tag}/n"] = np.array([x[3] for x in cells])
    print(tag, NC, "cells; median max log ratio vs usual %.3f vs class %.3f" % (np.median(out[f"{tag}/max_u"]), np.median(out[f"{tag}/max_c"])), flush=True)
np.savez_compressed(outp, **out)
