"""How much of a cell's predicted peak probability falls near given coordinates, relative to the measure's window.
Fits checks.py's regression on all contrasts and, for each (condition, block), prints the ratio
    P(model; within RADIUS mm of any seed) / P(window; same voxels)
using a real contrast that carries only that condition (so the label row is the one the fit used).
usage: check_cell.py CORPUS VARIANT PEN "condition" "x,y,z;x,y,z" "block;block" [RADIUS]
"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks as K, pc_model as P
from task_networks.corpus import load_corpus

corpus, variant, pen, cond, seeds, blocks = sys.argv[1:7]; rad = float(sys.argv[7]) if len(sys.argv) > 7 else 10.0
H = json.load(open(corpus)); C = load_corpus(corpus); blk, names = P.blocks_of(H, C, "measure_dir"); B = len(names); D = len(C["ids"]); nodes = C["nodes"]
K.FIT.update(names=names, wfrom="foci", opt="lbfgs", outer=15, pen={k: float(v) for k, v in (x.split("=") for x in pen.split(",") if x)})
Phi = K.basis(C["xyz"]); rows, cols, vals, F = K.features(C, H, blk, B, variant, names)
X = torch.zeros(D, F); X[torch.tensor(rows), torch.tensor(cols)] = torch.tensor(vals, dtype=torch.float32)
K.fit_additive(C, blk, B, Phi, X, list(range(D)), 20.0, torch.device("cuda:0")); A = K.FIT["A"]; logw = K.FIT["logw"]
S = np.array([[float(v) for v in s.split(",")] for s in seeds.split(";")]); near = (((C["xyz"][:, None] - S[None]) ** 2).sum(-1) <= rad ** 2).any(1)
print(f"{variant} pen {pen or '-'}: {near.sum()} voxels within {rad} mm of the seeds")
for bn in blocks.split(";"):
    b = names.index(bn); ds = [d for d in range(D) if blk[d] == b and len(C["auth"][d]) == 1 and nodes[C["auth"][d][0]].split(" | ")[0] == "c:all:" + cond]
    if not ds: print(f"  {bn}: no single-condition contrast"); continue
    lg = logw[b] + (X[ds[0]].numpy() @ A) @ Phi.T; p = np.exp(lg - lg.max()); p /= p.sum(); w = np.exp(logw[b]); w /= w.sum()
    print(f"  {cond} | {bn}: {len(ds)} single-condition contrasts, {sum(len(C['foci'][d]) for d in ds)} peaks; model share {p[near].sum():.4f}, window share {w[near].sum():.4f}, ratio {p[near].sum() / w[near].sum():.2f}")
