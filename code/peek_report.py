"""Quick look at a pc_model.py fit: did the conditions collapse onto single networks, and what do the maps look like.
usage: peek_report.py FIT_PREFIX OUT_PREFIX [TITLE]     writes OUT_PREFIX.json and OUT_PREFIX.png (fast: no nilearn)

Collapse: among conditions with >= MIN_MASS contrast-equivalents, the share whose largest loading is >= 0.95, and the
effective number of networks per condition (exp of the entropy of its loadings).
Maps: per network and block, the part above the block's own map: beta[k, b] - r * s[b], r = min over voxels of
beta / s (see network_report_mm.py). Drawn as maximum-intensity projections (sagittal, axial), threshold 0.3 of the max.
"""
import json, os, re, sys
import xml.etree.ElementTree as ET
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_MASS = 8.0
COLS = ["grey matter | decrease", "cortical thickness | decrease", "fa | decrease", "reho | decrease"]
fit, out = sys.argv[1:3]; title = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(fit)
M = json.load(open(fit + ".json")); nodes, names = M["nodes"], M["blocks"]; B = len(names)
be, s, n, th, xyz = (np.load(f"{fit}_{x}.npy").astype(float) for x in ("beta", "s", "n", "theta", "xyz"))
K = len(th[0]); V = len(xyz); be = be.reshape(K, B, V); s = s.reshape(B, V)
ok = s > np.median(s, 1, keepdims=True)
r = np.array([[(be[k, b][ok[b]] / s[b][ok[b]]).min() for b in range(B)] for k in range(K)]); own = np.clip(be - r[:, :, None] * s[None], 0, None)
ix = {x: i for i, x in enumerate(nodes)}; per = float(n.sum()) / 17285
# a condition = its own author plus its direction children ("c:all:x | less", "c:all:x | more"); loadings from pooled counts
grp = {}
for x in nodes:
    if x.startswith("c:all:") and not x[6:].startswith(("class ", "any condition")): grp.setdefault(x.split(" | ")[0], []).append(ix[x])
cond = list(grp); CN = np.array([n[grp[c]].sum(0) for c in cond]); mass = CN.sum(1) / per; big = mass >= MIN_MASS
T = (CN + 1e-9) / (CN + 1e-9).sum(1, keepdims=True)
neff = np.exp(-(T * np.log(np.clip(T, 1e-12, 1))).sum(1)); mx = T.max(1)
g = CN.sum(0); g = g / g.sum()
dirn = {d: sum(n[i] for x, i in ix.items() if x.startswith("c:all:") and x.endswith(" | " + d)) for d in ("less", "more")}
has_dir = bool(np.sum(dirn["less"]) + np.sum(dirn["more"]))
special = {x: int(np.argmax(n[ix[x]])) for x in ("background", "m:scope:restricted") if x in ix}
if M.get("flat_background"): special["flat (each measure's own map)"] = 0
stats = {"conditions": len(cond), "well_supported": int(big.sum()), "one_network_share": float((mx[big] >= 0.95).mean()),
         "median_largest_loading": float(np.median(mx[big])), "median_effective_networks": float(np.median(neff[big])),
         "background_network": special.get("background"), "restricted_scope_network": special.get("m:scope:restricted")}
print(title); print(json.dumps(stats, indent=1))
A = os.path.expanduser("~/nilearn_data/fsl/data/atlases/")
def atlas(img, xml):
    a = nib.load(A + "HarvardOxford/" + img); nm = {int(l.get("index")) + 1: l.text for l in ET.parse(A + xml).getroot().iter("label")}
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(a.affine), xyz)).astype(int); inb = np.all((ijk >= 0) & (ijk < np.array(a.shape)), 1)
    lab = np.zeros(V, int); lab[inb] = np.asarray(a.dataobj)[tuple(ijk[inb].T)]
    return lab, nm
cl, cn = atlas("HarvardOxford-cort-maxprob-thr25-2mm.nii.gz", "HarvardOxford-Cortical.xml")
sl, sn = atlas("HarvardOxford-sub-maxprob-thr25-2mm.nii.gz", "HarvardOxford-Subcortical.xml")
skip = re.compile(r"Cerebral (White Matter|Cortex)|Ventricle", re.I)
region = np.array([cn[c] if c else (re.sub(r"^(Left|Right) ", "", sn[q]) if q and not skip.search(sn[q]) else "white matter" if q else "cerebellum / outside atlas") for c, q in zip(cl, sl)])
rn, rcode = np.unique(region, return_inverse=True)
def regions(v, k=4):
    w = np.bincount(rcode, v, len(rn)); w = w / w.sum(); return [[rn[i], round(float(w[i]), 2)] for i in np.argsort(-w)[:k]]
cols = [names.index(c) for c in COLS if c in names] if B > 1 else [0]
res = []
for k in np.argsort(-g):
    o = np.argsort(-(T[:, k] * big))[:400]
    cs = sorted(((T[i, k] * mass[i], T[i, k], mass[i], cond[i][6:]) for i in o if big[i]), reverse=True)[:6]
    res.append({"network": int(k), "size": float(g[k]), "is": [a for a, v in special.items() if v == k],
                "conditions": [[c, round(float(t), 2), round(float(m), 0)] for _, t, m, c in cs],
                "more_share": float(dirn["more"][k] / max(dirn["more"][k] + dirn["less"][k], 1e-9)) if has_dir else None,
                "regions": {} if own[k].sum() == 0 else {names[b]: regions(own[k, b]) for b in cols}})
json.dump({"title": title, "stats": stats, "networks": res}, open(out + ".json", "w"), indent=1)
for x in res:
    if has_dir: print(f"\n   [more-in-patients share of this network's direction-specific mass: {x['more_share']:.2f}; overall {dirn['more'].sum() / (dirn['more'].sum() + dirn['less'].sum()):.2f}]", end="")
    print(f"\nnetwork {x['network']}: {x['size']:.1%} {x['is'] or ''}\n   conditions (loading, contrasts): " + "; ".join(f"{c} {t} ({m:.0f})" for c, t, m in x["conditions"]))
    for b, v in x["regions"].items(): print(f"   {b}: " + "; ".join(f"{a} {w:.0%}" for a, w in v))
# figure
lo = xyz.min(0); step = 4; ijk = np.rint((xyz - lo) / step).astype(int); shp = ijk.max(0) + 1
mask = np.zeros(shp, bool); mask[tuple(ijk.T)] = True
def draw(ax, v):
    vol = np.zeros(shp); vol[tuple(ijk.T)] = v / max(v.max(), 1e-30); vol[vol < 0.3] = 0
    sag = np.r_[vol.max(0).T]; axi = vol.max(2).T; ms = mask.max(0).T; ma = mask.max(2).T
    H = max(sag.shape[0], axi.shape[0]); pad = lambda a: np.pad(a.astype(float), ((0, H - a.shape[0]), (0, 2)))
    img = np.c_[pad(sag), pad(axi)]; m = np.c_[pad(ms), pad(ma)]
    ax.imshow(np.where(m > 0, 0.93, 1.0), origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(np.ma.masked_equal(img, 0), origin="lower", cmap="magma_r", vmin=0.2, vmax=1, interpolation="nearest")
    ax.contour(m, [0.5], colors="0.45", linewidths=0.5); ax.axis("off")
if B > 1:
    fig, axes = plt.subplots(K, len(cols), figsize=(len(cols) * 3.3, K * 1.7)); axes = np.atleast_2d(axes)
    for j, x in enumerate(res):
        for i, b in enumerate(cols):
            draw(axes[j, i], own[x["network"], b]); t = names[b].replace(" | decrease", " ↓")
            if i == 0: t = f"{j + 1}. net {x['network']}, {x['size']:.0%}{' ' + '/'.join(x['is']) if x['is'] else ''}: " + "; ".join(f"{c} {l}" for c, l, _ in x["conditions"][:3]) + "\n" + t
            axes[j, i].set_title(t[:170], fontsize=7, loc="left")
else:
    nc = 4; nr = int(np.ceil(K / nc)); fig, axes = plt.subplots(nr, nc, figsize=(nc * 4.2, nr * 2.3)); axes = np.atleast_2d(axes)
    for a in axes.ravel(): a.axis("off")
    for j, x in enumerate(res):
        ax = axes[j // nc, j % nc]; draw(ax, own[x["network"], 0])
        ax.set_title((f"{j + 1}. net {x['network']}, {x['size']:.0%}{' ' + '/'.join(x['is']) if x['is'] else ''}\n" + "; ".join(f"{c} {l}" for c, l, _ in x["conditions"][:3]))[:150], fontsize=7, loc="left")
fig.suptitle(f"{title}: {stats['one_network_share']:.0%} of {stats['well_supported']} well-supported conditions sit on one network (median effective networks {stats['median_effective_networks']:.1f})", fontsize=10, y=1.0)
fig.savefig(out + ".png", dpi=100, bbox_inches="tight"); print("figure", out + ".png")
