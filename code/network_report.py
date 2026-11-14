"""What a fit's networks look like: size, where each sits, which measures and conditions load on it, and a figure.
usage: network_report.py CORPUS FIT_PREFIX OUT_PREFIX      writes OUT_PREFIX.json and OUT_PREFIX.png

Per network: its share of all condition and measure mass; the Harvard-Oxford regions holding most of its map; its
strongest peaks; the direction split of the condition mass on it; the measures and the conditions that put the
largest share of their own mass on it (a condition = its cross-measure author plus its measure-specific authors;
only conditions with >= MIN_MASS contrast-equivalents; lift = share / the network's size).
"""
import collections, json, os, re, sys
import xml.etree.ElementTree as ET
import numpy as np, nibabel as nib

MIN_MASS = 8.0
corpus, fit, out = sys.argv[1:4]
H = json.load(open(corpus)); L = H["labels"]; P = H["parents"]
be, xyz, n = np.load(fit + "_beta.npy").astype(float), np.load(fit + "_xyz.npy"), np.load(fit + "_n.npy")
nodes = json.load(open(fit + ".json"))["nodes"]; K = len(be); ix = {x: i for i, x in enumerate(nodes)}
A = os.path.expanduser("~/nilearn_data/fsl/data/atlases/")
def atlas(img, xml, off):
    a = nib.load(A + "HarvardOxford/" + img); names = {int(l.get("index")) + 1: l.text for l in ET.parse(A + xml).getroot().iter("label")}
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(a.affine), xyz)).astype(int); ok = np.all((ijk >= 0) & (ijk < np.array(a.shape)), 1)
    lab = np.zeros(len(xyz), int); lab[ok] = np.asarray(a.dataobj)[tuple(ijk[ok].T)]
    return lab, names
cl, cn = atlas("HarvardOxford-cort-maxprob-thr25-2mm.nii.gz", "HarvardOxford-Cortical.xml", 0)
sl, sn = atlas("HarvardOxford-sub-maxprob-thr25-2mm.nii.gz", "HarvardOxford-Subcortical.xml", 0)
skip = re.compile(r"Cerebral (White Matter|Cortex)|Ventricle", re.I)
region = np.array([cn[c] if c else (re.sub(r"^(Left|Right) ", "", sn[s]) if s and not skip.search(sn[s]) else "white matter / other" if s else "cerebellum / outside atlas") for c, s in zip(cl, sl)])
content = np.array([x.startswith(("c:", "m:measure")) for x in nodes]); g = n[content].sum(0); g /= g.sum()
b = ix["background"]; shared = int(np.argmax(n[b] / n.sum(0)))
tok = sum(len(d["foci"]) for d in H["docs"].values()) and float(n[[ix[x] for x in nodes if x.startswith("c:")]].sum() + n[[ix[x] for x in nodes if x.startswith("m:")]].sum() + n[b].sum()) / len(H["docs"])
cond = collections.defaultdict(lambda: np.zeros(K))          # (direction, condition) -> counts over its own and its measure-specific authors
for x in nodes:
    if not x.startswith("c:") or "class " in x or "any condition" in x: continue
    d, rest = x[2:5], x[6:]; c = rest.rsplit(" | ", 1)[0]
    cond[d, c] += n[ix[x]]
dmass = {d: sum(v for (dd, _), v in cond.items() if dd == d) for d in ("dec", "inc")}
meas = {x[10:]: n[ix[x]] for x in nodes if x.startswith("m:measure:")}
res = []
for k in np.argsort(-g):
    v = be[k]; rs = collections.Counter()
    for r, w in zip(region, v): rs[r] += w
    pk = []
    for t in np.argsort(-v)[:800]:
        if all(np.linalg.norm(xyz[t] - xyz[q]) >= 20 for q in pk): pk.append(t)
        if len(pk) == 4: break
    left = v[xyz[:, 0] < -4].sum() / max(v[np.abs(xyz[:, 0]) > 4].sum(), 1e-12)
    cs = sorted(((v2[k] / v2.sum(), v2.sum() / tok, d, c) for (d, c), v2 in cond.items() if v2.sum() / tok >= MIN_MASS), reverse=True)[:7]
    ms = sorted(((v2[k] / v2.sum(), m) for m, v2 in meas.items()), reverse=True)
    res.append({"network": int(k), "size": float(g[k]), "shared": bool(k == shared), "background_share": float(n[b, k] / n[:, k].sum()),
                "regions": [[r, round(w / v.sum(), 3)] for r, w in rs.most_common(5)], "left_share": float(left),
                "peaks": [[int(c) for c in xyz[t]] for t in pk], "decrease_share": float(dmass["dec"][k] / (dmass["dec"][k] + dmass["inc"][k])),
                "measures": [[m, round(float(s), 3), round(float(s / g[k]), 2)] for s, m in ms[:4]],
                "conditions": [[("less: " if d == "dec" else "more: ") + c, round(float(s), 3), round(float(s / g[k]), 2), round(float(mass), 1)] for s, mass, d, c in cs]})
json.dump({"K": K, "tokens_per_contrast": tok, "overall_decrease_share": float(dmass["dec"].sum() / (dmass["dec"].sum() + dmass["inc"].sum())), "networks": res}, open(out + ".json", "w"), indent=1)
for r in res:
    print(f"\nnetwork {r['network']}: {r['size']:.1%} of mass{' [SHARED]' if r['shared'] else ''}; shared-author share {r['background_share']:.2f}; less-in-patients share {r['decrease_share']:.2f}; left {r['left_share']:.2f}; peaks {r['peaks']}")
    print("   regions: " + "; ".join(f"{a} {w:.0%}" for a, w in r["regions"]))
    print("   measures: " + "; ".join(f"{m} {s:.0%} ({l}x)" for m, s, l in r["measures"]))
    print("   conditions: " + "; ".join(f"{c} {s:.0%} ({l}x, {m:.0f})" for c, s, l, m in r["conditions"]))
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import plotting
    from nimare.utils import get_template
    tmpl = get_template("mni152_2mm", mask="brain"); ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(tmpl.affine), xyz)).astype(int)
    cols = 3; rows = int(np.ceil(K / cols)); fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.2, rows * 2.5)); axes = np.atleast_2d(axes)
    for j, r in enumerate(res):
        k = r["network"]; vol = np.zeros(tmpl.shape, np.float32); vol[tuple(ijk.T)] = be[k] / be[k].max()
        from scipy.ndimage import zoom, maximum_filter
        vol = maximum_filter(vol, size=2)
        ax = axes[j // cols, j % cols]
        plotting.plot_glass_brain(nib.Nifti1Image(vol, tmpl.affine), threshold=0.3, display_mode="lyrz", axes=ax, colorbar=False, cmap="magma_r", vmax=1.0, plot_abs=False,
                                  title=None, annotate=False)
        top = "; ".join(c.replace("less: ", "↓").replace("more: ", "↑") for c, *_ in r["conditions"][:2])
        ax.set_title(f"{j + 1}. network {k}: {r['size']:.0%}{' (shared)' if r['shared'] else ''}  {r['measures'][0][0]}  |  {top}"[:95], fontsize=8, loc="left")
    for j in range(len(res), rows * cols): axes[j // cols, j % cols].axis("off")
    fig.savefig(out + ".png", dpi=110, bbox_inches="tight"); print("figure", out + ".png")
except Exception as e:  # noqa: BLE001
    print("no figure:", repr(e))
