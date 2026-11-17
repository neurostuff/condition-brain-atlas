"""ALE (NiMARE, cluster-level FWE, voxel p < 0.001) on the planted condition's contrasts of each plant_atlas.py draw: does
a standard coordinate-based meta-analysis of just those k studies find the blob? Detection: a significant voxel within
12 mm of the blob centre. Also records whether any significant cluster appears anywhere.
usage: ale_plant.py PLANT_DIR N_ITERS CORES WORKER NWORKERS   (reads PLANT_DIR/coords/*.json, writes PLANT_DIR/ale/*.json)"""
import glob, json, os, sys
from pathlib import Path
import numpy as np, nibabel as nib
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cog_rep import ale
d, n_iters, cores, w, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); Path(d, "ale").mkdir(exist_ok=True)
files = sorted(glob.glob(f"{d}/coords/*.json"))
for i, fp in enumerate(files):
    if i % nw != w: continue
    out = f"{d}/ale/{os.path.basename(fp)}"
    if os.path.exists(out): continue
    c = json.load(open(fp)); ex = {p: (np.asarray(v, float), 30) for p, v in c["experiments"].items()}; stem = f"{d}/ale/{os.path.basename(fp)[:-5]}"
    r = ale(ex, stem, n_iters, cores); img = nib.load(stem + "_z.nii.gz"); z = img.get_fdata(); sig = np.argwhere(z > 1.645)
    xyzs = (img.affine @ np.c_[sig, np.ones(len(sig))].T).T[:, :3] if len(sig) else np.zeros((0, 3)); dist = np.linalg.norm(xyzs - np.array(c["centre"]), axis=1) if len(sig) else np.array([])
    json.dump({"file": os.path.basename(fp), "n_clusters": r["n_clusters"], "clusters": r["clusters"][:5], "detected": bool(len(dist) and dist.min() <= 12), "nearest_sig_mm": float(dist.min()) if len(dist) else None}, open(out, "w"))
    os.remove(stem + "_z.nii.gz"); print(os.path.basename(fp), r["n_clusters"], "clusters; detected", bool(len(dist) and dist.min() <= 12), flush=True)
