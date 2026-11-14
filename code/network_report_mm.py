"""What a multimodal fit's networks look like (pc_model.py fit): each network's own map in each measure x direction block.
usage: network_report_mm.py FIT_PREFIX OUT_PREFIX      writes OUT_PREFIX.json and OUT_PREFIX.png

beta[k, b] = (counts + LAM * s[b] + ETA) / norm, so the block's base map s[b] is in every network. The part a network
learned itself is beta[k, b] - r * s[b] with r = min over voxels of beta / s (a voxel the network has no data at);
1 - r is the share of the map that comes from data, and LAM / r - LAM its token count.
Per network: size (share of condition mass), conditions with the largest lift, and per block the data share, peak
equivalents and the Harvard-Oxford regions holding most of the learned part.
"""
import collections, json, os, re, sys
import xml.etree.ElementTree as ET
import numpy as np, nibabel as nib

MIN_MASS = 8.0          # contrast-equivalents a condition needs to be listed
MIN_PEAKS = 15          # peak-equivalents a network needs in a block for its map to be drawn
COLS = ["grey matter | decrease", "cortical thickness | decrease", "fa | decrease", "reho | decrease", "alff | decrease", "glucose metabolism | decrease"]
fit, out = sys.argv[1:3]
M = json.load(open(fit + ".json")); nodes, names, lam = M["nodes"], M["blocks"], M["lambda"]; B = len(names)
be, s, n, xyz = np.load(fit + "_beta.npy").astype(float), np.load(fit + "_s.npy").astype(float), np.load(fit + "_n.npy"), np.load(fit + "_xyz.npy")
K = len(be); V = len(xyz); be = be.reshape(K, B, V)
ok = s > np.median(s, 1, keepdims=True)
r = np.array([[(be[k, b][ok[b]] / s[b][ok[b]]).min() for b in range(B)] for k in range(K)]); r = np.clip(r, 1e-9, 1)
own = np.clip(be - r[:, :, None] * s[None], 0, None)
tokens = lam / r - lam; TOK_PER_PEAK = 54.0; peaks = tokens / TOK_PER_PEAK
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
ix = {x: i for i, x in enumerate(nodes)}
cond = {x[6:]: n[ix[x]] for x in nodes if x.startswith("c:all:") and not x[6:].startswith(("class ", "any condition"))}
g = np.sum(list(cond.values()), 0); g = g / g.sum(); tok_per_contrast = float(n.sum()) / 17285
res = []
for k in np.argsort(-g):
    cs = sorted(((v[k] / v.sum(), v.sum() / tok_per_contrast, c) for c, v in cond.items() if v.sum() / tok_per_contrast >= MIN_MASS), reverse=True)[:6]
    blocks = {}
    for b in np.argsort(-peaks[k]):
        if peaks[k, b] < MIN_PEAKS: continue
        rs = collections.Counter()
        for reg, w in zip(region, own[k, b]): rs[reg] += w
        t = own[k, b].sum()
        blocks[names[b]] = {"peaks": round(float(peaks[k, b]), 0), "data_share": round(float(1 - r[k, b]), 2),
                            "regions": [[a, round(w / t, 2)] for a, w in rs.most_common(4)]}
    res.append({"network": int(k), "size": float(g[k]), "conditions": [[c, round(float(sh), 3), round(float(sh / g[k]), 1), round(float(m), 1)] for sh, m, c in cs], "blocks": blocks})
json.dump({"K": K, "blocks": names, "networks": res}, open(out + ".json", "w"), indent=1)
for x in res:
    print(f"\nnetwork {x['network']}: {x['size']:.1%} of condition mass")
    print("   conditions: " + "; ".join(f"{c} {sh:.0%} ({l}x, {m:.0f})" for c, sh, l, m in x["conditions"]))
    for nm, v in list(x["blocks"].items())[:8]:
        print(f"   {nm} [{v['peaks']:.0f} peaks]: " + "; ".join(f"{a} {w:.0%}" for a, w in v["regions"]))
# does a measure's map stay inside its own support? share of each block's learned mass in white matter
wm = region == "white matter"
print("\nshare of learned map mass in white matter, per block (all networks pooled by peaks):")
for b in np.argsort(-peaks.sum(0)):
    o = (own[:, b] * tokens[:, b, None] / np.maximum(own[:, b].sum(1, keepdims=True), 1e-12)).sum(0)
    print(f"   {names[b]:36s} {o[wm].sum() / o.sum():.2f}   (base map {s[b][wm].sum():.2f})")
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import plotting
    from nimare.utils import get_template
    from scipy.ndimage import maximum_filter
    tmpl = get_template("mni152_2mm", mask="brain"); ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(tmpl.affine), xyz)).astype(int)
    cb = [names.index(c) for c in COLS]; fig, axes = plt.subplots(K, len(cb), figsize=(len(cb) * 3.4, K * 1.75))
    for j, x in enumerate(res):
        k = x["network"]
        for i, b in enumerate(cb):
            ax = axes[j, i]
            if peaks[k, b] < MIN_PEAKS:
                ax.axis("off"); ax.text(0.5, 0.5, f"little data ({peaks[k, b]:.0f} peaks)", ha="center", va="center", fontsize=7, color="0.55", transform=ax.transAxes)
            else:
                vol = np.zeros(tmpl.shape, np.float32); vol[tuple(ijk.T)] = own[k, b] / own[k, b].max(); vol = maximum_filter(vol, size=2)
                plotting.plot_glass_brain(nib.Nifti1Image(vol, tmpl.affine), threshold=0.3, display_mode="xz", axes=ax, colorbar=False, cmap="magma_r", vmax=1.0,
                                          plot_abs=False, annotate=False)
            t = f"{names[b].replace(' | decrease', ' ↓')}  ({peaks[k, b]:.0f})" if j else f"{names[b].replace(' | decrease', ' ↓ in patients')}  ({peaks[k, b]:.0f} peaks)"
            if i == 0:
                t = f"{j + 1}. net {k}, {x['size']:.0%}: " + "; ".join(c for c, *_ in x["conditions"][:3]) + "\n" + t
            ax.set_title(t[:150], fontsize=7, loc="left")
    fig.savefig(out + ".png", dpi=100, bbox_inches="tight"); print("figure", out + ".png")
except Exception as e:  # noqa: BLE001
    print("no figure:", repr(e))
